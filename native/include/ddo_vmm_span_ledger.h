#pragma once
#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
#include <utility>

namespace ddo_vmm_detail {

// An eventual CUDA owner can derive from this type. The ledger holds strong
// references but never queries, waits for, or destroys CUDA handles itself.
struct SpanFence { virtual ~SpanFence() = default; };
using SpanFenceOwner = std::shared_ptr<const SpanFence>;
enum class SpanState { available, reserved, live, pending, quarantined };
enum class SpanCompletion { complete, not_ready, failed };

// One mapped backing, externally serialized. No physical allocation or CUDA
// policy. Generation must be unique when a backing owner is replaced.
class SpanLedger {
public:
    struct Lease {
        const SpanLedger* owner;
        uint64_t generation, id;
        size_t offset, size, requested;
        std::vector<SpanFenceOwner> dependencies;
    };
    struct Snapshot {
        size_t mapped = 0, requested = 0, leased = 0, available = 0,
               pending = 0, quarantined = 0, live_count = 0, reserved_count = 0;
    };
    struct Geometry {
        uint64_t generation = 0;
        size_t largest_available = 0, largest_reusable = 0,
               available_runs = 0, reusable_runs = 0, spans = 0;
        bool eligible = false, fits_available = false, fits_reusable = false,
             whole_backing_releasable = false;
    };

    // CPU diagnostic prototype only. Pending intervals retain their fences;
    // reusable is a virtual fit, not proof of completed work or physical headroom.
    // Caller must hold its ledger lock and check context/retiring eligibility.
    Geometry geometry(size_t requested, int device) const noexcept {
        Geometry result;
        result.generation = generation_;
        result.whole_backing_releasable = releasable();
        const auto size = aligned_size(requested);
        result.eligible = !released_ && !release_pending_ && device == device_
            && next_id_ != UINT64_MAX && size.has_value();
        if (!result.eligible) return result;
        size_t available = 0, reusable = 0, end = 0;
        for (const auto& span : spans_) {
            ++result.spans;
            if (end != span.offset) { available = 0; reusable = 0; }
            if (span.state == SpanState::available) {
                if (!available) ++result.available_runs;
                available += span.size;
            } else available = 0;
            if (span.state == SpanState::available || span.state == SpanState::pending) {
                if (!reusable) ++result.reusable_runs;
                reusable += span.size;
            } else reusable = 0;
            if (available > result.largest_available) result.largest_available = available;
            if (reusable > result.largest_reusable) result.largest_reusable = reusable;
            end = span.offset + span.size;
        }
        result.fits_available = result.largest_available >= *size;
        result.fits_reusable = result.largest_reusable >= *size;
        return result;
    }

    SpanLedger(size_t mapped, int device, uint64_t generation)
        : mapped_(mapped), device_(device), generation_(generation) {
        if (!mapped || mapped % alignment || device < 0 || !generation)
            throw std::invalid_argument("invalid span backing identity/size");
        spans_.push_back({0, mapped, 0, 0, SpanState::available, {}});
    }
    SpanLedger(const SpanLedger&) = delete;
    SpanLedger& operator=(const SpanLedger&) = delete;

    std::optional<Lease> reserve(size_t requested, int device) {
        if (released_ || release_pending_ || device != device_ || next_id_ == UINT64_MAX) return {};
        const auto size = aligned_size(requested);
        if (!size) return {};
        auto best = spans_.end();
        for (auto it = spans_.begin(); it != spans_.end(); ++it) {
            if ((it->state == SpanState::available || it->state == SpanState::pending)
                && it->size >= *size && (best == spans_.end() || it->size < best->size))
                best = it;
        }
        if (best == spans_.end()) return {};
        // All throwing metadata copies/insertion happen before changing the
        // selected span. Failure leaves its partition and ownership untouched.
        Lease lease{this, generation_, next_id_, best->offset, *size, requested,
                    best->dependencies};
        if (best->size > *size)
            spans_.insert(std::next(best), {best->offset + *size, best->size - *size,
                          0, 0, best->state, best->dependencies});
        best->size = *size;
        best->requested = requested;
        best->id = next_id_++;
        best->state = SpanState::reserved;
        return lease;
    }

    std::optional<size_t> fit_size(size_t requested, int device) const {
        if (released_ || release_pending_ || device != device_ || next_id_ == UINT64_MAX) return {};
        const auto size = aligned_size(requested);
        if (!size) return {};
        std::optional<size_t> best;
        for (const auto& span : spans_)
            if ((span.state == SpanState::available || span.state == SpanState::pending)
                && span.size >= *size && (!best || span.size < *best)) best = span.size;
        return best;
    }

    // Opt-in fallback only: the legacy single-span best fit remains preferred.
    // Reserve a contiguous idle prefix while retaining every predecessor fence.
    std::optional<Lease> reserve_joined(size_t requested, int device) {
        if (auto single = reserve(requested, device)) return single;
        if (released_ || release_pending_ || device != device_ || next_id_ == UINT64_MAX) return {};
        const auto size = aligned_size(requested);
        if (!size) return {};
        const auto [first, fit] = joined_run(*size);
        if (!fit) return {};
        return reserve_run(first, *size, requested);
    }

    // Growable shared owner placement: the lowest-address contiguous run of
    // idle spans (available or pending, predecessor fences retained) that fits,
    // so freed low space is reused before untouched VA at the end. Existing
    // best-fit methods are unchanged and still used by every other owner.
    std::optional<Lease> reserve_lowest(size_t requested, int device) {
        if (released_ || release_pending_ || device != device_ || next_id_ == UINT64_MAX) return {};
        const auto size = aligned_size(requested);
        if (!size) return {};
        const auto [first, fit] = lowest_run(*size);
        if (!fit) return {};
        return reserve_run(first, *size, requested);
    }
    std::optional<size_t> fit_size_lowest(size_t requested, int device) const {
        if (released_ || release_pending_ || device != device_ || next_id_ == UINT64_MAX) return {};
        const auto size = aligned_size(requested);
        if (!size) return {};
        const auto fit = lowest_run(*size).second;
        return fit ? std::optional<size_t>(fit) : std::nullopt;
    }

    std::optional<size_t> fit_size_joined(size_t requested, int device) const {
        if (auto single = fit_size(requested, device)) return single;
        if (released_ || release_pending_ || device != device_ || next_id_ == UINT64_MAX) return {};
        const auto size = aligned_size(requested);
        if (!size) return {};
        const auto fit = joined_run(*size).second;
        return fit ? std::optional<size_t>(fit) : std::nullopt;
    }

    std::vector<SpanFenceOwner> pending_fences() const {
        std::vector<SpanFenceOwner> result;
        for (const auto& span : spans_) {
            if (span.state != SpanState::pending) continue;
            for (const auto& fence : span.dependencies) {
                bool found = false;
                for (const auto& existing : result) found |= existing == fence;
                if (!found) result.push_back(fence);
            }
        }
        return result;
    }

    // Distinct fences of pending spans intersecting one range. Empty when the
    // range is invalid. Read-only; the caller must hold the ledger lock.
    std::vector<SpanFenceOwner> range_pending_fences(size_t offset, size_t bytes) const {
        std::vector<SpanFenceOwner> result;
        if (released_ || release_pending_ || offset > mapped_ || bytes > mapped_ - offset) return result;
        const auto end = offset + bytes;
        for (const auto& span : spans_) {
            if (span.state != SpanState::pending) continue;
            if (span.offset + span.size <= offset || span.offset >= end) continue;
            for (const auto& fence : span.dependencies) {
                bool found = false;
                for (const auto& existing : result) found |= existing == fence;
                if (!found) result.push_back(fence);
            }
        }
        return result;
    }

    bool publish(const Lease& lease, bool dependencies_enqueued) {
        auto it = find(lease, SpanState::reserved);
        if (it == spans_.end()) return false;
        it->state = dependencies_enqueued ? SpanState::live : SpanState::quarantined;
        return dependencies_enqueued;
    }

    // Valid only before any submission/publication. After partial submission
    // the caller must quarantine, never roll back as though no work happened.
    bool cancel_unsubmitted(const Lease& lease) {
        auto it = find(lease, SpanState::reserved);
        if (it == spans_.end()) return false;
        it->state = it->dependencies.empty() ? SpanState::available : SpanState::pending;
        it->requested = 0;
        it->id = 0;
        coalesce();
        return true;
    }

    bool retire(const Lease& lease, SpanFenceOwner consumer_completion) {
        auto it = find(lease, SpanState::live);
        if (it == spans_.end()) return false;
        if (!consumer_completion) {
            it->state = SpanState::quarantined;
            return false;
        }
        // Caller proves this fence covers predecessor waits + all consumer
        // work. Construct first: metadata failure keeps the live lease intact.
        std::vector<SpanFenceOwner> dependencies{std::move(consumer_completion)};
        it->dependencies.swap(dependencies);
        it->requested = 0;
        it->id = 0;
        it->state = SpanState::pending;
        coalesce();
        return true;
    }

    void observe(const SpanFenceOwner& fence, SpanCompletion proof) {
        if (!fence || proof == SpanCompletion::not_ready) return;
        for (auto& span : spans_) {
            if (span.state != SpanState::pending) continue;
            for (auto it = span.dependencies.begin(); it != span.dependencies.end(); ++it) {
                if (*it != fence) continue;
                if (proof == SpanCompletion::failed) span.state = SpanState::quarantined;
                else {
                    span.dependencies.erase(it);
                    if (span.dependencies.empty()) span.state = SpanState::available;
                }
                break;
            }
        }
        coalesce();
    }

    Snapshot snapshot() const {
        Snapshot result;
        result.mapped = released_ ? 0 : mapped_;
        if (released_) return result;
        if (release_pending_) {
            result.pending = mapped_;
            return result;
        }
        for (const auto& span : spans_) {
            switch (span.state) {
            case SpanState::reserved:
                ++result.reserved_count;
                result.requested += span.requested;
                result.leased += span.size;
                break;
            case SpanState::live:
                ++result.live_count;
                result.requested += span.requested;
                result.leased += span.size;
                break;
            case SpanState::available: result.available += span.size; break;
            case SpanState::pending: result.pending += span.size; break;
            case SpanState::quarantined: result.quarantined += span.size; break;
            }
        }
        return result;
    }

    // Candidate-only physical accounting: clip states to one resident unit.
    // Requested bytes/counts remain whole-lease gauges in snapshot().
    std::optional<Snapshot> range_snapshot(size_t offset, size_t bytes) const {
        if (released_ || offset > mapped_ || bytes > mapped_ - offset) return {};
        Snapshot result; result.mapped = bytes;
        if (release_pending_) { result.pending = bytes; return result; }
        const auto end = offset + bytes;
        for (const auto& span : spans_) {
            const auto left = std::max(offset, span.offset);
            const auto right = std::min(end, span.offset + span.size);
            if (left >= right) continue;
            const auto size = right - left;
            switch (span.state) {
            case SpanState::reserved: case SpanState::live: result.leased += size; break;
            case SpanState::available: result.available += size; break;
            case SpanState::pending: result.pending += size; break;
            case SpanState::quarantined: result.quarantined += size; break;
            }
        }
        if (result.leased + result.available + result.pending + result.quarantined != bytes) return {};
        return result;
    }

    // Read-only proof under the same external serialization as reserve/retire.
    // A byte total is insufficient: every intersecting span must be available.
    bool dead_range(size_t offset, size_t bytes, int device, uint64_t generation) const {
        if (released_ || release_pending_ || device != device_ || generation != generation_
            || !bytes || offset > mapped_ || bytes > mapped_ - offset) return false;
        const auto end = offset + bytes;
        size_t cursor = offset;
        for (const auto& span : spans_) {
            const auto right = span.offset + span.size;
            if (right <= offset || span.offset >= end) continue;
            if (span.offset > cursor || span.state != SpanState::available
                || !span.dependencies.empty()) return false;
            cursor = right < end ? right : end;
        }
        return cursor == end;
    }

    bool owns_reservation(const Lease& lease, int device) const {
        if (released_ || release_pending_ || device != device_ || lease.owner != this || lease.generation != generation_)
            return false;
        for (const auto& span : spans_)
            if (span.state == SpanState::reserved && span.id == lease.id
                && span.offset == lease.offset && span.size == lease.size
                && span.requested == lease.requested) return true;
        return false;
    }

    bool releasable() const {
        if (released_ || release_pending_) return false;
        for (const auto& span : spans_)
            if (span.state != SpanState::available || !span.dependencies.empty()) return false;
        return true;
    }
    bool release_claimed() const { return release_pending_ && !released_; }
    // Reserve ownership under the caller's lock; actual native destruction
    // happens outside it. No new allocation can enter while release is pending.
    bool begin_backing_release() {
        if (!releasable()) return false;
        release_pending_ = true;
        return true;
    }
    bool commit_backing_release() {
        if (!release_pending_ || released_) return false;
        release_pending_ = false;
        released_ = true;
        spans_.clear();
        return true;
    }
    bool fail_backing_release() {
        if (!release_pending_) return false;
        release_pending_ = false;
        // Partial native destruction cannot be treated as reusable backing.
        for (auto& span : spans_) span.state = SpanState::quarantined;
        return true;
    }
    bool quarantine_idle_backing() {
        if (released_ || release_pending_) return false;
        for (const auto& span : spans_)
            if (span.state == SpanState::live || span.state == SpanState::reserved) return false;
        for (auto& span : spans_) span.state = SpanState::quarantined;
        return true;
    }

private:
    static constexpr size_t alignment = 512;
    struct Span {
        size_t offset, size, requested;
        uint64_t id;
        SpanState state;
        std::vector<SpanFenceOwner> dependencies;
    };
    std::list<Span> spans_;
    size_t mapped_;
    int device_;
    uint64_t generation_, next_id_ = 1;
    bool released_ = false, release_pending_ = false;

    static std::optional<size_t> aligned_size(size_t requested) {
        const size_t normalized = requested ? requested : alignment;
        if (normalized > std::numeric_limits<size_t>::max() - (alignment - 1)) return {};
        return ((normalized + alignment - 1) / alignment) * alignment;
    }
    auto find(const Lease& lease, SpanState state) -> std::list<Span>::iterator {
        if (released_ || lease.owner != this || lease.generation != generation_) return spans_.end();
        for (auto it = spans_.begin(); it != spans_.end(); ++it)
            if (it->id == lease.id && it->offset == lease.offset && it->size == lease.size
                && it->requested == lease.requested && it->state == state) return it;
        return spans_.end();
    }
    // Commit a reservation of `size` bytes starting at `first`, a run of idle
    // spans already proven to cover it. Shared by joined and lowest placement.
    std::optional<Lease> reserve_run(std::list<Span>::const_iterator first_const, size_t size,
                                     size_t requested) {
        auto first = spans_.erase(first_const, first_const); // const_iterator -> iterator.
        auto last = first;
        size_t remaining = size;
        std::vector<SpanFenceOwner> dependencies;
        for (;;) {
            for (const auto& fence : last->dependencies) {
                bool found = false;
                for (const auto& retained : dependencies) found |= retained == fence;
                if (!found) dependencies.push_back(fence);
            }
            if (remaining <= last->size) break;
            remaining -= last->size;
            ++last;
        }
        Lease lease{this, generation_, next_id_, first->offset, size, requested, dependencies};
        // Allocate/copy every replacement before touching the production list.
        // Splice/erase then commit without throwing or losing a predecessor.
        std::list<Span> replacement;
        replacement.push_back({first->offset, size, requested, next_id_,
                               SpanState::reserved, std::move(dependencies)});
        if (remaining < last->size)
            replacement.push_back({last->offset + remaining, last->size - remaining,
                                   0, 0, last->state, last->dependencies});
        const auto after = std::next(last);
        spans_.splice(first, replacement);
        spans_.erase(first, after);
        ++next_id_;
        return lease;
    }

    // Lowest-address contiguous run of idle spans covering size.
    auto lowest_run(size_t size) const
        -> std::pair<std::list<Span>::const_iterator, size_t> {
        auto first = spans_.end();
        size_t total = 0, end = 0;
        for (auto it = spans_.begin(); it != spans_.end(); ++it) {
            if (it->state != SpanState::available && it->state != SpanState::pending) {
                first = spans_.end(); total = 0; continue;
            }
            if (first == spans_.end() || end != it->offset) { first = it; total = 0; }
            total += it->size;
            end = it->offset + it->size;
            if (total >= size) return {first, total};
        }
        return {spans_.end(), 0};
    }
    auto joined_run(size_t size) const
        -> std::pair<std::list<Span>::const_iterator, size_t> {
        auto best = spans_.end();
        size_t best_size = 0;
        auto first = spans_.end();
        size_t total = 0, end = 0;
        auto consider = [&]() {
            if (total >= size && (!best_size || total < best_size)) {
                best = first;
                best_size = total;
            }
        };
        for (auto it = spans_.begin(); it != spans_.end(); ++it) {
            if (it->state != SpanState::available && it->state != SpanState::pending) {
                consider();
                first = spans_.end();
                total = 0;
                continue;
            }
            if (first == spans_.end() || end != it->offset) {
                consider();
                first = it;
                total = 0;
            }
            total += it->size;
            end = it->offset + it->size;
        }
        consider();
        return {best, best_size};
    }

    void coalesce() {
        for (auto it = spans_.begin(); it != spans_.end();) {
            auto next = std::next(it);
            if (next == spans_.end()) break;
            if ((it->state == SpanState::available || it->state == SpanState::pending)
                && it->state == next->state && it->dependencies == next->dependencies
                && it->offset + it->size == next->offset) {
                it->size += next->size;
                spans_.erase(next);
            } else ++it;
        }
    }
};
} // namespace ddo_vmm_detail
