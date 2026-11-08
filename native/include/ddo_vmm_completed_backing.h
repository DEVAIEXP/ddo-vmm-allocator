#pragma once
#include "ddo_vmm_span_ledger.h"
#include <mutex>

namespace ddo_vmm_detail {
enum class CompletedRelease { unavailable, released, failed };
struct ArenaPoolRelease {
    CompletedRelease status = CompletedRelease::unavailable;
    size_t bytes = 0;
    size_t backings = 0;
};
constexpr size_t arena_pool_release_limit = 16;

// Caller serializes the ledger and filters device/context/retiring ownership.
// Query only: no pending consumer can turn this transaction into a CUDA wait.
template<class Query, class Release>
CompletedRelease release_completed_backing(SpanLedger& ledger, Query query, Release release) {
    const auto snapshot = ledger.snapshot();
    if (!snapshot.mapped || snapshot.leased || snapshot.reserved_count || snapshot.quarantined)
        return CompletedRelease::unavailable;
    for (const auto& fence : ledger.pending_fences()) {
        const auto completion = query(fence);
        ledger.observe(fence, completion);
        if (completion == SpanCompletion::failed) {
            ledger.quarantine_idle_backing();
            return CompletedRelease::failed;
        }
        if (completion != SpanCompletion::complete) return CompletedRelease::unavailable;
    }
    if (!ledger.begin_backing_release()) return CompletedRelease::unavailable;
    if (!release()) {
        ledger.fail_backing_release();
        return CompletedRelease::failed;
    }
    if (!ledger.commit_backing_release()) return CompletedRelease::failed;
    return CompletedRelease::released;
}

inline thread_local bool completed_pool_recovery_active = false;

// Shared CPU-testable coordinator. The native adapter supplies driver queries,
// checked physical release, diagnostics, and empty-owner retirement callbacks.
template<class Owner, class Query, class Release, class Observe, class Retire>
ArenaPoolRelease release_completed_pool(
    std::mutex& mutex, std::vector<std::shared_ptr<Owner>>& owners,
    int device, uintptr_t context, size_t required,
    Query query, Release release, Observe observe, Retire retire) {
    ArenaPoolRelease result;
    if (completed_pool_recovery_active || device < 0 || !context || !required) return result;
    struct Scope {
        Scope() { completed_pool_recovery_active = true; }
        ~Scope() { completed_pool_recovery_active = false; }
    } scope;
    while (result.bytes < required && result.backings < arena_pool_release_limit) {
        std::shared_ptr<Owner> retired;
        {
            std::unique_lock lock(mutex, std::try_to_lock);
            if (!lock.owns_lock()) return result;
            for (auto it = owners.begin(); it != owners.end(); ++it) {
                const auto& owner = *it;
                if constexpr (requires { owner->segmented; })
                    if (owner->segmented) continue; // Separate checked physical-unit ownership.
                if (owner->retiring || owner->device != device
                    || reinterpret_cast<uintptr_t>(owner->context) != context) continue;
                const auto bytes = owner->ledger.snapshot().mapped;
                if (bytes > SIZE_MAX - result.bytes) {
                    result.status = CompletedRelease::failed;
                    return result;
                }
                CompletedRelease outcome;
                try {
                    outcome = release_completed_backing(owner->ledger,
                        [&](const SpanFenceOwner& fence) { return query(*owner, fence); },
                        [&] { owner->retiring = true; return release(*owner); });
                } catch (...) {
                    // A throwing callback cannot establish completed cleanup.
                    owner->ledger.fail_backing_release();
                    owner->ledger.quarantine_idle_backing();
                    owner->retiring = false;
                    result.status = CompletedRelease::failed;
                    return result;
                }
                if (outcome == CompletedRelease::unavailable) continue;
                observe(*owner, outcome, bytes);
                if (outcome == CompletedRelease::failed) {
                    owner->retiring = false;
                    result.status = CompletedRelease::failed;
                    return result;
                }
                retired = owner;
                owners.erase(it);
                result.bytes += bytes;
                ++result.backings;
                result.status = CompletedRelease::released;
                break;
            }
        }
        if (!retired) break;
        // Registry/VA destruction must run after dropping the pool lock.
        retire(*retired);
    }
    return result;
}
}
