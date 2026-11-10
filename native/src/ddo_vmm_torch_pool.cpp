#include "ddo_vmm.h"
#include "ddo_vmm_execution_recheck.h"
#include "ddo_vmm_torch_pool.h"
#include "ddo_vmm_span_ledger.h"
#include "ddo_vmm_stream_dependency.h"
#include "ddo_vmm_allocation_diagnostics.h"
#include "ddo_vmm_allocation_failure_trace.h"
#include "ddo_vmm_allocation_attempt.h"
#include "ddo_vmm_owner_observation.h"
#ifdef DDO_VMM_POOL_GEOMETRY_DIAGNOSTICS
#include "ddo_vmm_pool_geometry.h"
#endif
#include <cstdio>
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
#include "ddo_vmm_capacity_transaction.h"
#endif
#ifdef DDO_VMM_TIMING_COUNTERS
#include "ddo_vmm_timing_counters.h"
#if defined(_WIN32)
// Defined in ddo_vmm.cpp (same library); DXGI and system memory observation.
bool query_windows_memory(int device_ordinal, ddo_vmm_windows_memory_info* out_info);
#endif
#endif
#ifdef DDO_VMM_SEGMENTED_DRIVER_CANDIDATE
#define DDO_VMM_SEGMENTED_POOL_CPU 1
#include "cuda_segment_driver.h"
#include "segmented_driver_storage.h"
#endif
#ifdef DDO_VMM_SEGMENTED_POOL_CPU
#include "ddo_vmm_segmented_pool_cpu.h"
#endif
#ifdef DDO_VMM_POOL_LIFETIME_TRACE
#include "ddo_vmm_pool_lifetime_trace.h"
#include "ddo_vmm_pool_dependency_trace.h"
#endif
#include <cuda.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
using namespace ddo_vmm_detail;
#ifdef DDO_VMM_ALLOCATION_FAILURE_TRACE
void allocation_failure_sink(const char* stage, size_t bytes, int device,
                             uintptr_t stream, int status, bool exception, const char* error) noexcept {
    std::fprintf(stderr, "DDO allocation failure: stage=%s bytes=%zu device=%d stream=0x%llx status=%d exception=%d error=%s\n",
                 stage, bytes, device, static_cast<unsigned long long>(stream), status, exception ? 1 : 0, error);
    std::fflush(stderr);
}
using PoolAllocationAttempt = AllocationFailureTrace;
#else
constexpr AllocationFailureSink allocation_failure_sink = nullptr;
using PoolAllocationAttempt = NoopAllocationFailureTrace;
#endif
struct ContextScope {
    bool active;
    explicit ContextScope(CUcontext context) : active(cuCtxPushCurrent(context) == CUDA_SUCCESS) {}
    ~ContextScope() { if (active) { CUcontext previous = nullptr; cuCtxPopCurrent(&previous); } }
};
struct CudaFence final : SpanFence {
    CUevent event = nullptr;
    CUcontext context = nullptr;
    StreamIdentity stream;
    ~CudaFence() override {
        ContextScope scope(context);
        if (scope.active && event) cuEventDestroy(event);
    }
};
using RegionOwner = std::unique_ptr<ddo_vmm_region, decltype(&ddo_vmm_region_destroy)>;
struct Backing {
    RegionOwner region;
    std::unique_ptr<SpanLedger> local_ledger;
    SpanLedger& ledger;
    CUcontext context;
    uintptr_t address;
    int device;
    bool retiring = false;
#ifdef DDO_VMM_SEGMENTED_POOL_CPU
    std::shared_ptr<SegmentedPoolStorage> segmented;
    int segmented_error = 0;
    size_t next_reclaim_unit = 0;
    bool shared = false; // Growable shared owner (experimental; see growable_reserve).
    Backing(uintptr_t base, size_t extent, int ordinal, CUcontext ctx, uint64_t generation)
        : region(nullptr, &ddo_vmm_region_destroy), local_ledger(std::make_unique<SpanLedger>(extent, ordinal, generation)), ledger(*local_ledger),
          context(ctx), address(base), device(ordinal) {}
    Backing(std::shared_ptr<SegmentedPoolStorage> storage, SpanLedger& shared,
            uintptr_t base, int ordinal, CUcontext ctx)
        : region(nullptr, &ddo_vmm_region_destroy), ledger(shared), context(ctx),
          address(base), device(ordinal), segmented(std::move(storage)) {}
#endif
    Backing(RegionOwner owner, int ordinal, CUcontext ctx, uint64_t generation)
        : region(std::move(owner)), local_ledger(std::make_unique<SpanLedger>(ddo_vmm_region_mapped_size(region.get()), ordinal, generation)), ledger(*local_ledger),
          context(ctx), address(ddo_vmm_region_address(region.get())), device(ordinal) {}
    ~Backing() {
        // Only explicit completed retirement destroys backing. Process teardown
        // or any incomplete error path retains mappings rather than guessing.
        if (region) (void)region.release();
    }
};
struct Record { std::shared_ptr<Backing> backing; SpanLedger::Lease lease; };
std::mutex pool_mutex;
std::vector<std::shared_ptr<Backing>> backings;
std::unordered_map<void*, Record> records;
uint64_t next_generation = 1;
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
bool auto_segmented = false, allocation_started = false;
// Growable shared owner (experimental, cold-configured). 0 disables. When set,
// a request that fits no existing owner first goes to one shared segmented
// owner per (device, context) with this VA reservation (no physical up front);
// request-sized owners remain the fallback. Protected by pool_mutex.
uint64_t growable_reserve = 0;
std::atomic<uint64_t> growable_created{0}, growable_fallbacks{0}, growable_trim_calls{0},
    growable_trim_released_units{0};
// One request owns at most one dead-page recovery, including recursive admission.
thread_local bool auto_recovery_used = false;
thread_local int auto_recovery_error = 0;
thread_local size_t auto_request_depth = 0;
struct AutoRequestScope {
    AutoRequestScope() {
        if (!auto_request_depth++) { auto_recovery_used = false; auto_recovery_error = 0; }
    }
    ~AutoRequestScope() { --auto_request_depth; }
};
// Capacity transaction report. Every field starts unavailable (UINT64_MAX);
// a value is written only when it was observed. Never report a missing
// observation as zero. Field order is the experimental export order.
constexpr uint64_t unobserved = UINT64_MAX;
enum CapacityTermination : uint64_t {
    term_satisfied = 1, term_retry_refused = 2, term_no_release = 3, term_partial_release = 4,
    term_rollback_retained = 5, term_completion_error = 6, term_driver_error = 7, term_quota_consumed = 8,
};
struct CapacityReport {
    uint64_t request_bytes = unobserved, deficit_bytes = unobserved, owner_new = unobserved,
        operation = unobserved, status_first = unobserved, status_barrier = unobserved,
        status_continuation = unobserved, rollback_fresh = unobserved, rollback_released = unobserved,
        rollback_retained = unobserved, t1_pool_busy = unobserved, t1_owners = unobserved,
        t1_owners_busy = unobserved, t1_scanned = unobserved, t1_released = unobserved,
        t1_pending = unobserved, t1_live = unobserved, t1_quarantined = unobserved,
        t1_discarded = unobserved, t1_excluded = unobserved, t2_evictable = unobserved,
        t2_busy_arenas = unobserved, t2_evictions = unobserved, t2_bytes = unobserved,
        t2_status = unobserved, t3_status = unobserved, t3_fences = unobserved,
        t3_released = unobserved, retry_status = unobserved, termination = unobserved;
};
constexpr size_t capacity_report_fields = 31; // Field 0: a record is available.
bool capacity_report_recorded = false;         // Protected by pool_mutex.
CapacityReport last_capacity_report;           // Protected by pool_mutex.
// Clean capacity refusal of the current request, set by publish_segmented
// after a successful rollback and lease cancellation. Thread-local.
struct PendingCapacityRefusal {
    bool pending = false;
    std::shared_ptr<Backing> owner;
    size_t lease_offset = 0, lease_size = 0;
    SegmentedRefusal refusal;
};
thread_local PendingCapacityRefusal pending_capacity_refusal;
thread_local bool request_created_owner = false;
// During the single retry a retained rollback is reported by the transaction.
thread_local bool capacity_transaction_active = false;
thread_local bool retry_rollback_retained = false;
uint64_t status_value(int status) { return static_cast<uint64_t>(static_cast<uint32_t>(status)); }
CapacityReport begin_capacity_report(size_t bytes, size_t unit, const SegmentedRefusal& refusal) {
    CapacityReport r;
    r.request_bytes = bytes;
    if (unit && refusal.missing <= SIZE_MAX / unit) r.deficit_bytes = refusal.missing * unit;
    r.owner_new = request_created_owner ? 1 : 0;
    r.operation = static_cast<uint64_t>(refusal.operation);
    if (refusal.operation == 3) {
        if (refusal.access_observed) r.status_first = status_value(refusal.access_first);
        if (refusal.barrier_observed) r.status_barrier = status_value(refusal.access_barrier);
        if (refusal.continuation_observed) r.status_continuation = status_value(refusal.access_continuation);
    } else r.status_first = status_value(refusal.status);
    r.rollback_fresh = refusal.fresh; r.rollback_released = refusal.released;
    r.rollback_retained = refusal.retained;
    return r;
}
void publish_capacity_report(const CapacityReport& r) {
    // Caller must not hold pool_mutex.
    {
        std::scoped_lock lock(pool_mutex);
        last_capacity_report = r; capacity_report_recorded = true;
    }
    auto v = [](uint64_t value) { return static_cast<unsigned long long>(value); };
    std::fprintf(stderr, "DDO segmented capacity transaction: request=%llu deficit=%llu owner_new=%llu operation=%llu"
        " status_first=%llu status_barrier=%llu status_continuation=%llu rollback_fresh=%llu rollback_released=%llu"
        " rollback_retained=%llu t1_pool_busy=%llu t1_owners=%llu t1_owners_busy=%llu t1_scanned=%llu t1_released=%llu"
        " t1_pending=%llu t1_live=%llu t1_quarantined=%llu t1_discarded=%llu t1_excluded=%llu t2_evictable=%llu"
        " t2_busy_arenas=%llu t2_evictions=%llu t2_bytes=%llu t2_status=%llu t3_status=%llu t3_fences=%llu"
        " t3_released=%llu retry_status=%llu termination=%llu (unobserved=%llu)\n",
        v(r.request_bytes), v(r.deficit_bytes), v(r.owner_new), v(r.operation), v(r.status_first),
        v(r.status_barrier), v(r.status_continuation), v(r.rollback_fresh), v(r.rollback_released),
        v(r.rollback_retained), v(r.t1_pool_busy), v(r.t1_owners), v(r.t1_owners_busy), v(r.t1_scanned),
        v(r.t1_released), v(r.t1_pending), v(r.t1_live), v(r.t1_quarantined), v(r.t1_discarded),
        v(r.t1_excluded), v(r.t2_evictable), v(r.t2_busy_arenas), v(r.t2_evictions), v(r.t2_bytes),
        v(r.t2_status), v(r.t3_status), v(r.t3_fences), v(r.t3_released), v(r.retry_status),
        v(r.termination), v(unobserved));
    if (r.termination == term_satisfied) {
        std::fprintf(stderr,
            "DDO allocation recovery: status=recovered requested_bytes=%llu; "
            "allocation succeeded after releasing unused GPU backing; execution can continue.\n",
            v(r.request_bytes));
    } else {
        std::fprintf(stderr,
            "DDO allocation recovery: status=failed requested_bytes=%llu termination=%llu; "
            "allocation was not satisfied; see the capacity transaction details above.\n",
            v(r.request_bytes), v(r.termination));
    }
    std::fflush(stderr);
}
#ifdef DDO_VMM_TIMING_COUNTERS
// One Windows memory query in export order; false when unsupported or failed.
bool query_memory_fields([[maybe_unused]] int device,
                         [[maybe_unused]] std::array<uint64_t, memory_fields>& out) {
#if defined(_WIN32)
    ddo_vmm_windows_memory_info info{};
    if (!query_windows_memory(device, &info)) return false;
    out = {info.local_budget_bytes, info.local_usage_bytes, info.nonlocal_budget_bytes,
           info.nonlocal_usage_bytes, info.available_physical_bytes, info.available_commit_bytes};
    return true;
#else
    // Linux/WSL: device capacity from the current context; no WDDM demotion
    // or non-local segment, so those fields stay unavailable.
    size_t free_bytes = 0, total_bytes = 0;
    if (cuMemGetInfo(&free_bytes, &total_bytes) != CUDA_SUCCESS || free_bytes > total_bytes) return false;
    out = {total_bytes, total_bytes - free_bytes, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
    return true;
#endif
}
// Observation only; caller holds no pool lock. Unavailable (not recorded) when
// the platform has no WDDM/system query or the query fails.
void record_windows_memory(int device, bool unsatisfied, uint64_t status) {
    std::array<uint64_t, memory_fields> values{};
    if (query_memory_fields(device, values)) timing_record_memory(unsatisfied, status, values);
}
// Throttled sample from the allocation path; caller holds no pool lock.
// Returns true with `values` filled when this call took a successful sample.
bool sample_memory(int device, std::array<uint64_t, memory_fields>& values) {
    if (!memory_sampler.claim(timing_now_ns())) return false;
    if (!query_memory_fields(device, values)) return false;
    memory_sampler.record(values, timing_now_ns());
    return true;
}
void sample_composition(int device, const std::array<uint64_t, memory_fields>& memory);
// Budget-aware idle activation retention (experimental, cold-configured).
// policy 0 = fixed (today's cache_limit()), 1 = budget. Protected by pool_mutex
// after configuration; configuration itself is cold-only.
int retention_policy = 0;
uint64_t retention_headroom_bytes = 512ull * 1024 * 1024;
uint64_t retention_cap_bytes = 4096ull * 1024 * 1024;
constexpr uint64_t retention_max_observation_age_ns = 2ull * 1000 * 1000 * 1000;
// Decision counters: kept, reduced (discard work done), fallback to fixed
// because no fresh observation, bytes released by budget reduction.
std::atomic<uint64_t> retention_kept{0}, retention_reduced{0}, retention_fallbacks{0},
    retention_released_bytes{0};
// Bytes of retained idle physical backing the budget allows, or false when no
// fresh observation exists (caller falls back to the fixed limit).
bool budget_allowed_idle(uint64_t idle_physical, uint64_t& allowed) {
    uint64_t budget = 0, usage = 0;
    if (!memory_sampler.latest(timing_now_ns(), retention_max_observation_age_ns, budget, usage)) return false;
    const uint64_t target = budget > retention_headroom_bytes ? budget - retention_headroom_bytes : 0;
    // Usage already includes the retained idle backing; remove it to see the rest.
    const uint64_t others = usage > idle_physical ? usage - idle_physical : 0;
    allowed = target > others ? target - others : 0;
    if (allowed > retention_cap_bytes) allowed = retention_cap_bytes;
    return true;
}
#endif
std::shared_ptr<Backing> create_segmented_owner(int, CUcontext, size_t, int&);
int trim_segmented_idle(bool);
#endif
#ifdef DDO_VMM_SEGMENTED_DRIVER_CANDIDATE
void unregister_segmented_storage(const std::shared_ptr<SegmentedPoolStorage>&);
#endif
#ifdef DDO_VMM_POOL_GEOMETRY_DIAGNOSTICS
void geometry_failure(size_t bytes, size_t mapped, int device, CUcontext context,
                      cudaStream_t stream, int status, int reason) noexcept {
    PoolGeometryFailure<> snapshot;
    try {
        // Refuse to observe a busy pool instead of waiting or touching CUDA.
        std::unique_lock lock(pool_mutex, std::try_to_lock);
        if (lock.owns_lock())
            snapshot.capture(backings, bytes, device, reinterpret_cast<uintptr_t>(context));
    } catch (...) {
        snapshot.available = false;
        snapshot.count = 0;
    }
    // The output lock cannot block another thread while holding pool_mutex.
    (void)snapshot.write(stderr, bytes, mapped, device, reinterpret_cast<uintptr_t>(context),
                         reinterpret_cast<uintptr_t>(stream), status, reason);
    std::fflush(stderr);
}
#endif
std::atomic<uint64_t> eviction_attempts{0}, eviction_wait_ns{0}, failures{0};
#ifdef DDO_VMM_POOL_LIFETIME_TRACE
PoolLifetimeTrace<> lifetime_trace;
PoolDependencyTrace<> dependency_trace;
void trace_dependency(uint64_t action, uintptr_t pointer, uintptr_t backing,
    uint64_t generation, uint64_t lease, const CudaFence& fence,
    StreamIdentity consumer, CUstream stream, CUresult status) {
    dependency_trace.record(action, pointer, backing, generation, lease,
        fence.stream.id, consumer.id, (fence.stream.known ? 1 : 0) | (consumer.known ? 2 : 0),
        reinterpret_cast<uintptr_t>(stream), reinterpret_cast<uintptr_t>(fence.event),
        static_cast<uint64_t>(status), lifetime_trace.total_sequence());
}
void trace_lease(uint64_t action, const Backing& owner, const SpanLedger::Lease& lease,
                 cudaStream_t stream, CUevent fence = nullptr, uint64_t status = 0) {
    lifetime_trace.record(action, owner.address + lease.offset, owner.address,
        owner.ledger.snapshot().mapped, lease.offset, lease.size, lease.requested,
        lease.generation, lease.id, reinterpret_cast<uintptr_t>(stream),
        reinterpret_cast<uintptr_t>(fence), status);
}
#endif

size_t cache_limit() {
    char* owned = nullptr;
#ifdef _WIN32
    size_t length = 0;
    if (_dupenv_s(&owned, &length, "DDO_VMM_TORCH_CACHE_MIB") != 0) return 0;
    const char* value = owned;
#else
    const char* value = std::getenv("DDO_VMM_TORCH_CACHE_MIB");
#endif
    char* end = nullptr;
    const auto amount = value ? std::strtoull(value, &end, 10) : 0;
    const bool valid = value && end != value && *end == '\0'
        && amount <= std::numeric_limits<size_t>::max() / (1024ULL * 1024ULL);
    std::free(owned);
    return valid ? static_cast<size_t>(amount) * 1024ULL * 1024ULL : 0;
}
StreamIdentity identity(cudaStream_t stream, int device) {
    unsigned long long id = 0;
    if (cuStreamGetId(reinterpret_cast<CUstream>(stream), &id) != CUDA_SUCCESS) return {};
    return {id, device, true};
}
bool idle(const Backing& owner) {
    const auto s = owner.ledger.snapshot();
    return s.mapped && !s.leased && !s.quarantined;
}
void reap(Backing& owner) {
    ContextScope scope(owner.context);
    for (const auto& dependency : owner.ledger.pending_fences()) {
        const auto fence = std::static_pointer_cast<const CudaFence>(dependency);
        const auto status = scope.active ? cuEventQuery(fence->event) : CUDA_ERROR_INVALID_CONTEXT;
#ifdef DDO_VMM_POOL_LIFETIME_TRACE
        trace_dependency(3, 0, owner.address, 0, 0, *fence, {}, nullptr, status);
#endif
        if (status == CUDA_SUCCESS) owner.ledger.observe(dependency, SpanCompletion::complete);
        else if (status != CUDA_ERROR_NOT_READY) {
            owner.ledger.observe(dependency, SpanCompletion::failed);
            failures.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

// Select/mark under the ledger lock, then prove completion and release outside
// it. Pending bytes remain visible and unavailable throughout the transaction.
bool evict_one(bool over_limit_only, [[maybe_unused]] uint64_t reason = 0,
               [[maybe_unused]] size_t requested = 0) {
    std::shared_ptr<Backing> selected;
    std::vector<SpanFenceOwner> fences;
    {
        std::scoped_lock lock(pool_mutex);
        size_t idle_bytes = 0;
        for (const auto& owner : backings) {
#ifdef DDO_VMM_SEGMENTED_POOL_CPU
            if (owner->segmented) continue; // Never send segmented VA to monolithic retirement.
#endif
            if (owner->retiring || !idle(*owner)) continue;
            const auto mapped = owner->ledger.snapshot().mapped;
            if (mapped > SIZE_MAX - idle_bytes) return false;
            idle_bytes += mapped;
        }
        if (over_limit_only && idle_bytes <= cache_limit()) return false;
        for (const auto& owner : backings) {
#ifdef DDO_VMM_SEGMENTED_POOL_CPU
            if (owner->segmented) continue;
#endif
            if (owner->retiring || !idle(*owner)) continue;
            // Cache trimming must not wait for outstanding consumer work.
            // Query completion only; allocation-pressure eviction below still
            // waits for pending events before releasing physical backing.
            if (over_limit_only) {
                reap(*owner);
                if (!idle(*owner)) continue; // Failed query quarantined it.
            }
            const auto mapped = owner->ledger.snapshot().mapped;
            // In-flight idle bytes remain mapped, accounted and reusable via
            // stream dependencies. They are not permission to grow budgets.
            if (over_limit_only && !owner->ledger.releasable()) continue;
            if (!selected || mapped < selected->ledger.snapshot().mapped) selected = owner;
        }
        if (!selected) return false;
        fences = selected->ledger.pending_fences();
        selected->retiring = true;
#ifdef DDO_VMM_POOL_LIFETIME_TRACE
        lifetime_trace.record(3, 0, selected->address, selected->ledger.snapshot().mapped,
            reason, 0, requested, 0, 0, 0, 0, 0);
#endif
    }
    eviction_attempts.fetch_add(1, std::memory_order_relaxed);
    const auto started = std::chrono::steady_clock::now();
    bool completed = true;
    {
        ContextScope scope(selected->context);
        completed = scope.active;
        for (const auto& dependency : fences) {
            const auto fence = std::static_pointer_cast<const CudaFence>(dependency);
            const auto status = scope.active ? cuEventSynchronize(fence->event) : CUDA_ERROR_INVALID_CONTEXT;
            if (status != CUDA_SUCCESS) completed = false;
#ifdef DDO_VMM_POOL_LIFETIME_TRACE
            std::scoped_lock lock(pool_mutex);
            trace_dependency(4, 0, selected->address, 0, 0, *fence, {}, nullptr, status);
#endif
        }
    }
    eviction_wait_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
    {
        std::scoped_lock lock(pool_mutex);
        for (const auto& fence : fences)
            selected->ledger.observe(fence, completed ? SpanCompletion::complete : SpanCompletion::failed);
        if (!completed || !selected->ledger.begin_backing_release()) {
#ifdef DDO_VMM_POOL_LIFETIME_TRACE
            lifetime_trace.record(5, 0, selected->address, selected->ledger.snapshot().mapped,
                0, 0, 0, 0, 0, 0, 0, 1);
#endif
            selected->ledger.quarantine_idle_backing();
            selected->retiring = false;
            failures.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }
    // Checked eviction is required; the legacy void destructor does not report
    // partial unmap/release failures. Never label such backing reusable.
    const bool unmapped = ddo_vmm_region_evict(selected->region.get()) == DDO_VMM_SUCCESS;
    if (unmapped) selected->region.reset();
    {
        std::scoped_lock lock(pool_mutex);
#ifdef DDO_VMM_POOL_LIFETIME_TRACE
        lifetime_trace.record(unmapped ? 4 : 5, 0, selected->address,
            selected->ledger.snapshot().mapped, 0, 0, 0, 0, 0, 0, 0, unmapped ? 0 : 1);
#endif
        if (unmapped) {
            selected->ledger.commit_backing_release();
            for (auto it = backings.begin(); it != backings.end(); ++it)
                if (*it == selected) { backings.erase(it); break; }
        } else {
            selected->ledger.fail_backing_release();
            selected->retiring = false;
            failures.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return unmapped;
}

void* publish_locked(const std::shared_ptr<Backing>& owner, size_t bytes, cudaStream_t stream,
                     PoolAllocationAttempt& attempt, std::optional<SpanLedger::Lease> prepared = {}) {
    attempt.stage("reserve_lease");
#ifdef DDO_VMM_NATIVE_JOINED_SPANS
    auto lease = prepared ? std::move(prepared) : owner->ledger.reserve_joined(bytes, owner->device);
#else
    auto lease = prepared ? std::move(prepared) : owner->ledger.reserve(bytes, owner->device);
#endif
    if (!lease) return nullptr;
    for (const auto& dependency : lease->dependencies) {
        attempt.stage("reuse_dependency");
        const auto fence = std::static_pointer_cast<const CudaFence>(dependency);
#ifdef DDO_VMM_POOL_LIFETIME_TRACE
        bool ordered = false;
        if (fence->context == owner->context) {
            const auto consumer = identity(stream, owner->device);
            auto observed_wait = [&](CUstream target, CUevent event, unsigned int flags) {
                const auto status = cuStreamWaitEvent(target, event, flags);
                trace_dependency(2, owner->address + lease->offset, owner->address,
                    lease->generation, lease->id, *fence, consumer, target, status);
                return status;
            };
            ordered = enqueue_reuse_dependency(fence->event, reinterpret_cast<CUstream>(stream),
                observed_wait, fence->stream, consumer);
            if (!fence->event || same_stream(fence->stream, consumer))
                trace_dependency(1, owner->address + lease->offset, owner->address,
                    lease->generation, lease->id, *fence, consumer,
                    reinterpret_cast<CUstream>(stream), CUDA_SUCCESS);
        } else {
            trace_dependency(5, owner->address + lease->offset, owner->address,
                lease->generation, lease->id, *fence, {},
                reinterpret_cast<CUstream>(stream), CUDA_ERROR_INVALID_CONTEXT);
        }
        if (!ordered) {
#else
        if (fence->context != owner->context || !enqueue_reuse_dependency(fence->event,
            reinterpret_cast<CUstream>(stream), cuStreamWaitEvent, fence->stream, identity(stream, owner->device))) {
#endif
            owner->ledger.publish(*lease, false);
            failures.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
    }
    attempt.stage("pointer_range");
    if (lease->offset > UINTPTR_MAX - owner->address) {
        owner->ledger.publish(*lease, false);
        return nullptr;
    }
    void* pointer = reinterpret_cast<void*>(owner->address + lease->offset);
    try {
        attempt.stage("record_insert");
        const auto [entry, inserted] = records.emplace(pointer, Record{owner, *lease});
        if (!inserted) { owner->ledger.publish(*lease, false); return nullptr; }
        attempt.stage("lease_publish");
        if (!owner->ledger.publish(*lease, true)) { records.erase(entry); return nullptr; }
#ifdef DDO_VMM_POOL_LIFETIME_TRACE
        trace_lease(1, *owner, *lease, stream);
#endif
    } catch (...) {
        // Submission may already exist. Preserve backing + predecessor owners.
        owner->ledger.publish(*lease, false);
        throw;
    }
    return pointer;
}
#ifdef DDO_VMM_SEGMENTED_POOL_CPU
// Reserve under canonical serialization; driver materialization is outside it.
// The owner stays excluded from fits/reclaim/close across the unlocked interval.
void* publish_segmented(const std::shared_ptr<Backing>& owner, size_t bytes, cudaStream_t stream,
        PoolAllocationAttempt& attempt, std::unique_lock<std::mutex>& lock) {
    if (owner->retiring) return nullptr;
    // The growable shared owner places at the lowest fitting address so freed,
    // still-resident space is reused before untouched VA at its end.
    auto lease = owner->shared ? owner->ledger.reserve_lowest(bytes, owner->device)
                               : owner->ledger.reserve_joined(bytes, owner->device);
    if (!lease) return nullptr;
    owner->retiring = true;
    if (!owner->segmented->materialize(*lease, owner->ledger, lock)) {
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
        std::fprintf(stderr, "DDO segmented materialization refusal: requested=%llu lease=%llu charged=%llu driver_status=%d recovery_used=%d recovery_error=%d\n",
            static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(lease->size),
            static_cast<unsigned long long>(owner->segmented->physical_charge()),
            owner->segmented->error(), auto_recovery_used ? 1 : 0, auto_recovery_error);
#endif
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
        const auto refusal = owner->segmented->last_refusal();
#endif
        if (owner->segmented->error()) {
            owner->segmented_error = owner->segmented->error();
            attempt.stage("segmented_materialize_driver_error", owner->segmented_error);
            owner->ledger.publish(*lease, false);
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
            if (refusal.capacity && !refusal.rolled_back) {
                // Rollback stopped on a Driver failure: ownership is retained
                // exactly as before and no capacity work may run.
                if (capacity_transaction_active) retry_rollback_retained = true;
                else {
                    auto report = begin_capacity_report(bytes, owner->segmented->unit_bytes(), refusal);
                    report.termination = term_rollback_retained;
                    lock.unlock(); publish_capacity_report(report); lock.lock();
                }
            }
#endif
        }
        else {
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
            if (auto_recovery_error) attempt.stage("segmented_recovery_error", auto_recovery_error);
            if (refusal.capacity && refusal.rolled_back) {
                pending_capacity_refusal = {true, owner, lease->offset, lease->size, refusal};
                attempt.stage("segmented_capacity_refusal", refusal.status);
            }
#endif
            owner->ledger.cancel_unsubmitted(*lease); owner->retiring = false;
        }
        return nullptr;
    }
    void* pointer = nullptr;
    try { pointer = publish_locked(owner, bytes, stream, attempt, *lease); }
    catch (...) { owner->ledger.publish(*lease, false); throw; }
    owner->retiring = false;
    return pointer;
}
struct DeadPageRecovery { size_t scanned = 0, released = 0; int error = 0; };
// Preserve the actual query error instead of reporting a successful zero-byte
// recovery. Pending events remain pending; no synchronization is introduced.
int reap_segmented(Backing& owner) {
    ContextScope scope(owner.context);
    for (const auto& dependency : owner.ledger.pending_fences()) {
        const auto fence = std::static_pointer_cast<const CudaFence>(dependency);
        const auto status = scope.active ? cuEventQuery(fence->event) : CUDA_ERROR_INVALID_CONTEXT;
        if (status == CUDA_SUCCESS) owner.ledger.observe(dependency, SpanCompletion::complete);
        else if (status != CUDA_ERROR_NOT_READY) {
            owner.ledger.observe(dependency, SpanCompletion::failed);
            owner.segmented_error = static_cast<int>(status);
            owner.retiring = true;
            failures.fetch_add(1, std::memory_order_relaxed);
            return owner.segmented_error;
        }
    }
    return 0;
}
// Separate fixed work bounds; do not repurpose/increase the 16-backing service.
DeadPageRecovery reclaim_segmented(const std::shared_ptr<Backing>& owner,
        size_t scan_limit = 64, size_t release_limit = 16) {
    DeadPageRecovery result;
    std::unique_lock lock(pool_mutex, std::try_to_lock);
    if (!lock || !owner->segmented || owner->retiring) return result;
    if (const int error = reap_segmented(*owner)) { result.error = error; return result; }
    owner->retiring = true;
    const auto pages = owner->segmented->page_count();
    const auto first = owner->next_reclaim_unit;
    for (size_t n = 0; n < pages && result.scanned < scan_limit && result.released < release_limit; ++n) {
        const auto i = (first + n) % pages;
        owner->next_reclaim_unit = (i + 1) % pages;
        ++result.scanned;
        if (owner->segmented->discard(i, owner->ledger, lock)) ++result.released;
        else if (owner->segmented->error()) {
            owner->segmented_error = result.error = owner->segmented->error(); return result;
        }
    }
    owner->retiring = false;
    return result;
}
// CPU candidate only: a single bounded recovery per failed allocation, separate
// from whole-backing recovery. Never wait for a pool lock or pending consumer.
DeadPageRecovery recover_segmented_context(int device, CUcontext context) {
    DeadPageRecovery total;
    static thread_local bool recovering = false;
    if (recovering) return total;
    struct RecoveryScope {
        bool& flag;
        explicit RecoveryScope(bool& value) : flag(value) { flag = true; }
        ~RecoveryScope() { flag = false; }
    } guard(recovering);
    std::vector<std::shared_ptr<Backing>> selected;
    static size_t next_owner = 0; // Protected by pool_mutex, independent of page cursors.
    {
        std::unique_lock lock(pool_mutex, std::try_to_lock);
        if (!lock) return total;
        for (size_t n = 0; n < backings.size(); ++n) {
            const auto& owner = backings[(next_owner + n) % backings.size()];
            if (owner->segmented && !owner->retiring && owner->device == device && owner->context == context)
                selected.push_back(owner);
            if (selected.size() == 16) break;
        }
    }
    for (const auto& owner : selected) {
        if (total.scanned == 64 || total.released == 16) break;
        {
            std::unique_lock lock(pool_mutex, std::try_to_lock);
            if (!lock) return total;
            const auto it = std::find(backings.begin(), backings.end(), owner);
            if (it != backings.end()) next_owner = (static_cast<size_t>(it - backings.begin()) + 1) % backings.size();
        }
        const auto part = reclaim_segmented(owner, 64 - total.scanned, 16 - total.released);
        total.scanned += part.scanned; total.released += part.released;
        if (part.error) { total.error = part.error; return total; }
    }
    return total;
}
struct SegmentedPoolReport {
    size_t owners = 0, virtual_bytes = 0, physical_charge = 0;
    size_t live_bytes = 0, pending_bytes = 0, quarantined_bytes = 0;
    bool available = false;
};
SegmentedPoolReport segmented_report_locked() {
    SegmentedPoolReport result;
    for (const auto& owner : backings) {
        if (!owner->segmented) continue;
        const auto spans = owner->ledger.snapshot();
        auto add = [](size_t& total, size_t value) {
            if (value > SIZE_MAX - total) return false;
            total += value; return true;
        };
        if (!add(result.owners, 1) || !add(result.virtual_bytes, spans.mapped)
            || !add(result.physical_charge, owner->segmented->physical_charge())
            || !add(result.live_bytes, spans.leased) || !add(result.pending_bytes, spans.pending)
            || !add(result.quarantined_bytes, spans.quarantined)) return {};
    }
    result.available = true; return result;
}
#ifdef DDO_VMM_TIMING_COUNTERS
// Composition row for one successful memory sample. Caller holds no lock; the
// arena observation and the pool are try-locked separately, never nested.
void sample_composition(int device, const std::array<uint64_t, memory_fields>& memory) {
    constexpr uint64_t unavailable = UINT64_MAX;
    std::array<uint64_t, composition_fields> row{};
    row.fill(unavailable);
    row[0] = memory[0]; row[1] = memory[1]; row[2] = memory[3];
    uint64_t flags = 0;
    const auto arenas = observe_arena_composition(device);
    if (!arenas.registry_observed) flags |= 1;
    if (arenas.busy_arenas) flags |= 2;
    if (!flags) {
        row[3] = arenas.model_bytes; row[4] = arenas.slot_bytes; row[5] = arenas.region_bytes;
        row[9] = arenas.model_arenas; row[10] = arenas.slot_arenas;
    }
    {
        std::unique_lock lock(pool_mutex, std::try_to_lock);
        if (!lock) {
            flags |= 4;
        } else {
            uint64_t physical = 0, live = 0, pending = 0;
            for (const auto& owner : backings) {
                if (owner->device != device) continue;
                const auto spans = owner->ledger.snapshot();
                physical += owner->segmented ? owner->segmented->physical_charge() : spans.mapped;
                live += spans.leased; pending += spans.pending;
            }
            row[6] = physical; row[7] = live; row[8] = pending;
        }
    }
    row[11] = flags;
    composition_sampler.record(row);
}
#endif
SegmentedPoolReport segmented_report() {
    std::unique_lock lock(pool_mutex, std::try_to_lock);
    return lock ? segmented_report_locked() : SegmentedPoolReport{};
}
bool close_segmented(const std::shared_ptr<Backing>& owner) {
    std::unique_lock lock(pool_mutex, std::try_to_lock);
    if (!lock || !owner->segmented || owner->retiring) return false;
    if (reap_segmented(*owner)) return false;
    if (!owner->ledger.begin_backing_release()) return false;
    // Freeze the ledger for close, then release already-proven dead pages using
    // a read-only proof captured before the frozen phase.
    owner->retiring = true;
    if (!owner->segmented->close_va(lock)) {
        owner->segmented_error = owner->segmented->error();
        owner->ledger.fail_backing_release(); return false;
    }
    if (!owner->ledger.commit_backing_release()) return false;
    for (auto it = backings.begin(); it != backings.end(); ++it)
        if (*it == owner) { backings.erase(it); break; }
#ifdef DDO_VMM_SEGMENTED_DRIVER_CANDIDATE
    unregister_segmented_storage(owner->segmented);
#endif
    return true;
}
#endif
}

#ifdef DDO_VMM_SEGMENTED_DRIVER_CANDIDATE
namespace {
using DriverStorage = SegmentedDriverStorage<ddo_dead_pages::CudaSegmentDriver>;
// Retention precedes all Driver work, including metadata-publication failures.
std::vector<std::shared_ptr<DriverStorage>> driver_storage_owners;
void unregister_segmented_storage(const std::shared_ptr<SegmentedPoolStorage>& storage) {
    // Caller holds pool_mutex, and checked close has already released all VA.
    for (auto it = driver_storage_owners.begin(); it != driver_storage_owners.end(); ++it)
        if (*it == storage) { driver_storage_owners.erase(it); break; }
}
int segmented_admission(int device, size_t bytes) noexcept {
    const auto initial = ddo_vmm_pressure_prepare_cuda_allocation(device, bytes);
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
    if (auto_segmented && initial == DDO_VMM_OUT_OF_MEMORY && !auto_recovery_used) {
        auto_recovery_used = true;
        CUcontext context = nullptr;
        if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context) return DDO_VMM_CUDA_ERROR;
        const auto recovery = recover_segmented_context(device, context);
        if (recovery.error) { auto_recovery_error = recovery.error; return DDO_VMM_CUDA_ERROR; }
        if (recovery.released) return static_cast<int>(ddo_vmm_pressure_prepare_cuda_allocation(device, bytes));
    }
#endif
    return static_cast<int>(initial);
}
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
// Access OOM is no longer recovered inside the Driver adapter. The pool runs
// the capacity transaction after rollback (see run_capacity_transaction).
std::shared_ptr<Backing> create_segmented_owner(int device, CUcontext context, size_t extent, int& error) {
    const auto page = ddo_vmm_device_granularity(device);
    if (!page || !extent || extent % page || extent / page > 4096) { error = -1; return {}; }
    uint64_t generation;
    {
        std::scoped_lock lock(pool_mutex);
        if (next_generation == UINT64_MAX) { error = -1; return {}; }
        generation = next_generation++;
    }
    auto storage = std::make_shared<DriverStorage>(pool_mutex, context, device,
        extent, page, generation, &segmented_admission);
    {
        std::scoped_lock lock(pool_mutex); driver_storage_owners.push_back(storage);
    }
    if (!storage->open()) { error = storage->error(); return {}; }
    auto owner = std::make_shared<Backing>(storage, storage->ledger(), storage->address(), device, context);
    std::scoped_lock lock(pool_mutex); backings.push_back(owner);
    return owner;
}
#ifdef DDO_VMM_TIMING_COUNTERS
// Close idle owners that hold no physical backing (VA hygiene, as the fixed
// policy does below its limit). Query-only completion; never waits.
int close_empty_idle(const std::vector<std::shared_ptr<Backing>>& selected) {
    for (const auto& owner : selected) {
        {
            std::unique_lock lock(pool_mutex, std::try_to_lock);
            if (!lock) return 0;
            if (owner->retiring || owner->segmented->physical_charge() != 0) continue;
            if (reap_segmented(*owner)) return 1;
            if (!owner->ledger.releasable()) continue;
        }
        if (!close_segmented(owner)) return 1;
    }
    return 0;
}
// Budget policy: keep retained idle backing up to `allowed`; above it, discard
// completed dead units (existing proof, no waits) until back within budget,
// then close owners left without physical backing. Caller holds no lock.
int reduce_idle_to_budget(std::vector<std::shared_ptr<Backing>> selected,
                          uint64_t idle_physical, uint64_t allowed) {
    if (idle_physical <= allowed) {
        retention_kept.fetch_add(1, std::memory_order_relaxed);
        return close_empty_idle(selected);
    }
    const uint64_t need = idle_physical - allowed;
    uint64_t released = 0;
    for (const auto& owner : selected) {
        if (released >= need) break;
        const uint64_t unit = owner->segmented->unit_bytes();
        if (!unit) continue;
        const uint64_t units = (need - released + unit - 1) / unit;
        const auto part = reclaim_segmented(owner, owner->segmented->page_count(), static_cast<size_t>(units));
        if (part.error) return 1;
        released += static_cast<uint64_t>(part.released) * unit;
    }
    retention_reduced.fetch_add(1, std::memory_order_relaxed);
    retention_released_bytes.fetch_add(released, std::memory_order_relaxed);
    return close_empty_idle(selected);
}
#endif
// Query-only completed close, bounded to 16 owners; no destructor cleanup.
int trim_segmented_idle(bool over_limit) {
    std::vector<std::shared_ptr<Backing>> selected;
    {
        std::unique_lock lock(pool_mutex, std::try_to_lock);
        if (!lock) return 0;
        size_t idle_physical = 0;
        for (const auto& owner : backings) {
            if (owner->retiring || !idle(*owner)) continue;
            // The shared owner is trimmed per unit (trim_shared_units) on the
            // free path; only an explicit trim may close it as a whole.
            if (over_limit && owner->shared) continue;
            const auto count = owner->segmented ? owner->segmented->physical_charge() : owner->ledger.snapshot().mapped;
            if (count > SIZE_MAX - idle_physical) return 1;
            idle_physical += count;
            if (owner->segmented && selected.size() < 16) selected.push_back(owner);
        }
#ifdef DDO_VMM_TIMING_COUNTERS
        if (over_limit && retention_policy == 1) {
            uint64_t allowed = 0;
            if (budget_allowed_idle(idle_physical, allowed)) {
                lock.unlock();
                return reduce_idle_to_budget(std::move(selected), idle_physical, allowed);
            }
            retention_fallbacks.fetch_add(1, std::memory_order_relaxed);
        }
#endif
        if (over_limit && idle_physical <= cache_limit()) {
            // Empty physical owners still hold VA; do not retain them as cache.
            selected.erase(std::remove_if(selected.begin(), selected.end(), [](const auto& owner) {
                return owner->segmented->physical_charge() != 0;
            }), selected.end());
        }
    }
    for (const auto& owner : selected) {
        {
            std::unique_lock lock(pool_mutex, std::try_to_lock);
            if (!lock) return 0;
            if (owner->retiring) continue;
            if (reap_segmented(*owner)) return 1;
            if (!owner->ledger.releasable()) continue;
        }
        if (!close_segmented(owner)) return 1;
    }
    return 0;
}
// Unit-level idle trim of the shared owner (free path). Physical not covered by
// live leases above the fixed idle limit triggers one bounded reclaim pass:
// completed fences are observed (query only), then at most 64 units are scanned
// and 16 discarded with the dead-range proof. Pending units are never waited on.
int trim_shared_units() {
    std::vector<std::shared_ptr<Backing>> shared;
    {
        std::unique_lock lock(pool_mutex, std::try_to_lock);
        if (!lock) return 0;
        const uint64_t limit = cache_limit();
        for (const auto& owner : backings) {
            if (!owner->shared || owner->retiring || !owner->segmented) continue;
            const auto spans = owner->ledger.snapshot();
            const uint64_t physical = owner->segmented->physical_charge();
            const uint64_t idle_physical = physical > spans.leased ? physical - spans.leased : 0;
            if (idle_physical > limit) shared.push_back(owner);
        }
    }
    for (const auto& owner : shared) {
        growable_trim_calls.fetch_add(1, std::memory_order_relaxed);
        const auto part = reclaim_segmented(owner, 64, 16);
        if (part.error) return 1;
        growable_trim_released_units.fetch_add(part.released, std::memory_order_relaxed);
    }
    return 0;
}
// One request-scoped capacity transaction after a clean capacity refusal.
// Entry: no pool lock held; the refused lease is cancelled and its owner is
// healthy. Work bounds: W1 Tier 1 releases <= D; W2 scans <= pages of the
// owners snapshotted at entry; W3 owners = that snapshot; W4 arena evictions
// <= evictable pages observed at Tier 2 entry; W5 waits only on the distinct
// fences of the fewest pending units covering the remaining deficit. Busy
// locks are counted and skipped, never waited on. Exactly one retry.
template<class Retry>
void* run_capacity_transaction(int device, CUcontext context, cudaStream_t stream,
        size_t bytes, Retry retry) {
    auto refused = std::move(pending_capacity_refusal);
    pending_capacity_refusal = {};
    const size_t unit = refused.owner->segmented->unit_bytes();
    auto r = begin_capacity_report(bytes, unit, refused.refusal);
#ifdef DDO_VMM_TIMING_COUNTERS
    TimingScope transaction_timing(TimingOp::transaction, bytes);
    // Observation only, with no pool lock held: memory state at the first refusal.
    if (!timing_first_refusal_recorded())
        record_windows_memory(device, false, status_value(refused.refusal.status));
    uint64_t tier_started = timing_now_ns();
    auto tier_done = [&](TimingOp op) {
        const uint64_t now = timing_now_ns();
        timing_counters.add(op, now - tier_started); tier_started = now;
    };
#endif
    auto finish = [&](uint64_t termination, void* pointer = nullptr) {
        r.termination = termination; publish_capacity_report(r);
#ifdef DDO_VMM_TIMING_COUNTERS
        if (termination != term_satisfied) record_windows_memory(device, true, termination);
#endif
        return pointer;
    };
    if (auto_recovery_used) return finish(term_quota_consumed);
    auto_recovery_used = true;
    if (!unit || r.deficit_bytes == unobserved || !r.deficit_bytes) return finish(term_no_release);
    const size_t deficit = static_cast<size_t>(r.deficit_bytes);
    size_t released = 0;
    auto excluded = [&](const std::shared_ptr<Backing>& owner, size_t offset) {
        return owner == refused.owner && offset < refused.lease_offset + refused.lease_size
            && offset + unit > refused.lease_offset;
    };
    struct Candidate { std::shared_ptr<Backing> owner; size_t index; };
    std::vector<Candidate> pending;
    // Tier 1: completed dead units of every snapshotted owner (no waits).
    std::vector<std::shared_ptr<Backing>> owners;
    size_t snapshot_pages = 0;
    {
        std::unique_lock lock(pool_mutex, std::try_to_lock);
        r.t1_pool_busy = lock ? 0 : 1;
        if (lock) {
            for (const auto& owner : backings) {
                if (!owner->segmented || owner->retiring || owner->device != device || owner->context != context)
                    continue;
                owners.push_back(owner);
                snapshot_pages += owner->segmented->page_count();
            }
        }
    }
    if (!r.t1_pool_busy) {
        pending.reserve(snapshot_pages); // No allocation while an owner is excluded.
        r.t1_owners = owners.size();
        r.t1_owners_busy = r.t1_scanned = r.t1_released = r.t1_pending = 0;
        r.t1_live = r.t1_quarantined = r.t1_discarded = r.t1_excluded = 0;
        for (const auto& owner : owners) {
            if (released >= deficit) break;
            std::unique_lock lock(pool_mutex, std::try_to_lock);
            if (!lock || owner->retiring) { ++r.t1_owners_busy; continue; }
            if (reap_segmented(*owner)) { lock.unlock(); return finish(term_completion_error); } // Retained by reap.
            if (owner->segmented->unit_bytes() != unit) continue;
            owner->retiring = true;
            const size_t pages = owner->segmented->page_count();
            for (size_t i = 0; i < pages && released < deficit; ++i) {
                ++r.t1_scanned;
                const size_t offset = i * unit;
                if (excluded(owner, offset)) { ++r.t1_excluded; continue; }
                const auto kind = owner->segmented->unit_kind(i);
                if (kind == SegmentedUnitKind::discarded) { ++r.t1_discarded; continue; }
                if (kind != SegmentedUnitKind::resident) { ++r.t1_quarantined; continue; }
                const auto part = owner->ledger.range_snapshot(offset, unit);
                if (!part || part->quarantined) { ++r.t1_quarantined; continue; }
                if (part->leased) { ++r.t1_live; continue; }
                if (part->pending) { ++r.t1_pending; pending.push_back({owner, i}); continue; }
                if (owner->segmented->discard(i, owner->ledger, lock)) { released += unit; ++r.t1_released; }
                else if (owner->segmented->error()) {
                    // Failed release: the owner stays retiring with its units retained.
                    owner->segmented_error = owner->segmented->error();
                    lock.unlock(); return finish(term_driver_error);
                }
            }
            owner->retiring = false;
        }
    }
#ifdef DDO_VMM_TIMING_COUNTERS
    tier_done(TimingOp::transaction_tier1);
#endif
    // Tier 2: unpinned model-arena pages only, through the pressure lock order.
    if (released < deficit) {
        const auto count = pressure_count_arena_pages(device);
        r.t2_evictable = count.evictable_pages; r.t2_busy_arenas = count.busy_arenas;
        r.t2_evictions = 0; r.t2_bytes = 0; r.t2_status = 1;
        while (released < deficit && r.t2_evictions < count.evictable_pages) {
            const auto eviction = pressure_evict_arena_page(device);
            r.t2_busy_arenas = std::max<uint64_t>(r.t2_busy_arenas, eviction.busy_arenas);
            if (eviction.outcome == ArenaEviction::failed) { r.t2_status = 2; return finish(term_driver_error); }
            if (eviction.outcome != ArenaEviction::evicted) break;
            ++r.t2_evictions; r.t2_bytes += eviction.bytes; released += eviction.bytes; r.t2_status = 0;
        }
    }
#ifdef DDO_VMM_TIMING_COUNTERS
    tier_done(TimingOp::transaction_tier2);
#endif
    // Tier 3 (bounded completion wait): only selected pending units.
    // t3_status: 1 ran, 2 capture active, 3 wait failed, 4 capture unobserved, 5 pool busy.
    if (released < deficit && !pending.empty()) {
        CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_NONE;
        if (cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &capture) != CUDA_SUCCESS) r.t3_status = 4;
        else if (capture != CU_STREAM_CAPTURE_STATUS_NONE) r.t3_status = 2;
        else {
            const size_t needed = (deficit - released + unit - 1) / unit;
            if (pending.size() > needed) pending.resize(needed);
            struct Wait { std::shared_ptr<Backing> owner; SpanFenceOwner fence; };
            std::vector<Wait> waits;
            {
                std::unique_lock lock(pool_mutex, std::try_to_lock);
                if (!lock) r.t3_status = 5;
                else {
                    for (const auto& candidate : pending)
                        for (const auto& fence : candidate.owner->ledger.range_pending_fences(candidate.index * unit, unit)) {
                            bool found = false;
                            for (const auto& wait : waits) found |= wait.fence == fence;
                            if (!found) waits.push_back({candidate.owner, fence});
                        }
                }
            }
            if (r.t3_status != 5) {
                r.t3_status = 1; r.t3_fences = 0; r.t3_released = 0;
                for (const auto& wait : waits) {
                    const auto fence = std::static_pointer_cast<const CudaFence>(wait.fence);
                    CUresult status = CUDA_ERROR_INVALID_CONTEXT;
                    {
                        ContextScope scope(wait.owner->context);
                        if (scope.active && fence->context == wait.owner->context)
                            status = cuEventSynchronize(fence->event);
                    }
                    ++r.t3_fences;
                    std::unique_lock lock(pool_mutex, std::try_to_lock);
                    // A busy pool leaves the fence pending; a later query observes it.
                    if (lock) wait.owner->ledger.observe(wait.fence,
                        status == CUDA_SUCCESS ? SpanCompletion::complete : SpanCompletion::failed);
                    if (status != CUDA_SUCCESS) {
                        if (lock) lock.unlock();
                        r.t3_status = 3; return finish(term_completion_error);
                    }
                }
                for (const auto& candidate : pending) {
                    if (released >= deficit) break;
                    std::unique_lock lock(pool_mutex, std::try_to_lock);
                    if (!lock || candidate.owner->retiring) continue;
                    candidate.owner->retiring = true;
                    if (candidate.owner->segmented->unit_kind(candidate.index) == SegmentedUnitKind::resident
                        && candidate.owner->segmented->discard(candidate.index, candidate.owner->ledger, lock)) {
                        released += unit; ++r.t3_released;
                    } else if (candidate.owner->segmented->error()) {
                        candidate.owner->segmented_error = candidate.owner->segmented->error();
                        lock.unlock(); return finish(term_driver_error);
                    }
                    candidate.owner->retiring = false;
                }
            }
        }
    }
#ifdef DDO_VMM_TIMING_COUNTERS
    tier_done(TimingOp::transaction_tier3);
#endif
    if (!released) return finish(term_no_release);
#ifdef DDO_VMM_TIMING_COUNTERS
    TimingScope retry_timing(TimingOp::transaction_retry, bytes);
#endif
    capacity_transaction_active = true; retry_rollback_retained = false;
    void* pointer = nullptr;
    try { pointer = retry(); } catch (...) { capacity_transaction_active = false; throw; }
    capacity_transaction_active = false;
    if (pointer) { r.retry_status = 1; return finish(term_satisfied, pointer); }
    r.retry_status = 2;
    const bool capacity = pending_capacity_refusal.pending;
    pending_capacity_refusal = {};
    if (retry_rollback_retained) return finish(term_rollback_retained);
    if (!capacity) return finish(term_driver_error);
    return finish(released >= deficit ? term_retry_refused : term_partial_release);
}
#endif
}
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
// Cold opt-in only, applied by install_pytorch_vmm_allocator before the
// allocator is installed; refused after the first allocation.
extern "C" DDO_VMM_API int ddo_exp_segmented_configure_auto(int enabled) noexcept {
    try {
        std::scoped_lock lock(pool_mutex);
        if ((enabled != 0 && enabled != 1) || allocation_started || !backings.empty()
            || !records.empty() || !driver_storage_owners.empty()) return 1;
        auto_segmented = enabled != 0; return 0;
    } catch (...) { return 2; }
}
// Private cold configuration of the growable shared owner (experimental).
// reserve_bytes: VA reservation of the shared owner (0 disables; at most
// 64 GiB; the per-owner unit bound and granularity are checked when it is
// created, and a refused creation falls back to request-sized owners).
// Refused after the first allocation or when any owner exists.
extern "C" DDO_VMM_API int ddo_exp_segmented_configure_growable(uint64_t reserve_bytes) noexcept {
    try {
        std::scoped_lock lock(pool_mutex);
        if (allocation_started || !backings.empty() || !records.empty() || !driver_storage_owners.empty()
            || reserve_bytes > (64ull << 30)) return 1;
        growable_reserve = reserve_bytes; return 0;
    } catch (...) { return 2; }
}
// Growable owner observations (8): reserve bytes, shared owners created, shared
// owners present, request-sized fallback owners created while enabled, unit
// trim passes, units released by unit trim, shared physical bytes, shared live
// bytes. Returns 2 while the pool lock is busy.
extern "C" DDO_VMM_API int ddo_exp_growable_stats_v1(uint64_t* output, size_t count) noexcept {
    if (!output || count != 8) return 1;
    try {
        std::unique_lock lock(pool_mutex, std::try_to_lock);
        if (!lock) return 2;
        uint64_t present = 0, physical = 0, live = 0;
        for (const auto& owner : backings) {
            if (!owner->shared || !owner->segmented) continue;
            ++present; physical += owner->segmented->physical_charge();
            live += owner->ledger.snapshot().leased;
        }
        const uint64_t values[8]{growable_reserve, growable_created.load(), present, growable_fallbacks.load(),
            growable_trim_calls.load(), growable_trim_released_units.load(), physical, live};
        std::copy(std::begin(values), std::end(values), output);
        return 0;
    } catch (...) { return 3; }
}
#ifdef DDO_VMM_TIMING_COUNTERS
// Private cold configuration of idle activation retention (experimental).
// policy 0 = fixed (bootstrap cache limit), 1 = budget-aware. Refused after
// the first allocation or when any owner exists. Headroom and cap in bytes.
extern "C" DDO_VMM_API int ddo_exp_segmented_configure_retention(int policy, uint64_t headroom_bytes,
                                                                  uint64_t cap_bytes) noexcept {
    try {
        std::scoped_lock lock(pool_mutex);
        if ((policy != 0 && policy != 1) || allocation_started || !backings.empty()
            || !records.empty() || !driver_storage_owners.empty()
            || headroom_bytes > (64ull << 30) || cap_bytes > (64ull << 30)) return 1;
        retention_policy = policy; retention_headroom_bytes = headroom_bytes; retention_cap_bytes = cap_bytes;
        return 0;
    } catch (...) { return 2; }
}
// Retention decisions: policy, headroom, cap, kept, reduced, fallbacks to the
// fixed limit (no fresh observation), bytes released by budget reduction.
extern "C" DDO_VMM_API int ddo_exp_retention_stats_v1(uint64_t* output, size_t count) noexcept {
    if (!output || count != 7) return 1;
    try {
        std::unique_lock lock(pool_mutex, std::try_to_lock);
        if (!lock) return 2;
        const uint64_t values[7]{static_cast<uint64_t>(retention_policy), retention_headroom_bytes,
            retention_cap_bytes, retention_kept.load(), retention_reduced.load(),
            retention_fallbacks.load(), retention_released_bytes.load()};
        std::copy(std::begin(values), std::end(values), output);
        return 0;
    } catch (...) { return 3; }
}
#endif
#endif
// Private candidate ABI only. No default allocator selection or installed binding.
extern "C" DDO_VMM_API int ddo_exp_segmented_create(int device, size_t extent) noexcept {
    try {
        CUcontext context = nullptr; CUdevice current = -1;
        if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context
            || cuCtxGetDevice(&current) != CUDA_SUCCESS || current != device) return 1;
        const size_t page = ddo_vmm_device_granularity(device);
        if (!page || !extent || extent % page || extent / page > 4096) return 1;
        uint64_t generation;
        {
            std::scoped_lock lock(pool_mutex);
            if (next_generation == UINT64_MAX) return 1;
            generation = next_generation++;
        }
        auto storage = std::make_shared<DriverStorage>(pool_mutex, context, device,
            extent, page, generation, &segmented_admission);
        {
            std::scoped_lock lock(pool_mutex); driver_storage_owners.push_back(storage);
        }
        if (!storage->open()) return 2;
        auto owner = std::make_shared<Backing>(storage, storage->ledger(), storage->address(), device, context);
        std::scoped_lock lock(pool_mutex); backings.push_back(owner);
        return 0;
    } catch (...) { return 3; }
}
extern "C" DDO_VMM_API int ddo_exp_segmented_recover(int device, uint64_t* output, size_t count) noexcept {
    if (!output || count != 4) return 1;
    try {
        CUcontext context = nullptr; CUdevice current = -1;
        if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context
            || cuCtxGetDevice(&current) != CUDA_SUCCESS || current != device) return 1;
        const auto result = recover_segmented_context(device, context);
        const uint64_t values[4]{1, result.scanned, result.released, static_cast<uint64_t>(result.error)};
        std::copy(std::begin(values), std::end(values), output);
        return result.error ? 2 : 0;
    } catch (...) { return 3; }
}
extern "C" DDO_VMM_API int ddo_exp_segmented_close(int device) noexcept {
    try {
        CUcontext context = nullptr; CUdevice current = -1;
        if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context
            || cuCtxGetDevice(&current) != CUDA_SUCCESS || current != device) return 1;
        std::vector<std::shared_ptr<Backing>> selected;
        {
            std::scoped_lock lock(pool_mutex);
            for (const auto& owner : backings)
                if (owner->segmented && owner->context == context && owner->device == device) selected.push_back(owner);
        }
        for (const auto& owner : selected) {
            if (!close_segmented(owner)) return 2;
            std::scoped_lock lock(pool_mutex);
            driver_storage_owners.erase(std::remove_if(driver_storage_owners.begin(), driver_storage_owners.end(),
                [&](const auto& storage) { return storage.get() == owner->segmented.get(); }), driver_storage_owners.end());
        }
        return 0;
    } catch (...) { return 3; }
}
extern "C" DDO_VMM_API int ddo_exp_segmented_stats(uint64_t* output, size_t count) noexcept {
    if (!output || count != 10) return 1;
    try {
        size_t retained_va = 0, retained_physical = 0;
        std::unique_lock lock(pool_mutex, std::try_to_lock);
        if (!lock) return 2;
        const auto report = segmented_report_locked();
        if (!report.available) return 2;
        for (const auto& owner : driver_storage_owners) {
            if (owner->va_charge() > SIZE_MAX - retained_va || owner->physical_charge() > SIZE_MAX - retained_physical) return 2;
            retained_va += owner->va_charge(); retained_physical += owner->physical_charge();
        }
        const uint64_t values[10]{1, report.owners, report.virtual_bytes, report.physical_charge,
            report.live_bytes, report.pending_bytes, report.quarantined_bytes,
            driver_storage_owners.size(), retained_va, retained_physical};
        std::copy(std::begin(values), std::end(values), output); return 0;
    } catch (...) { return 3; }
}
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
// Experimental CPU copy of the last capacity-transaction record. Field 0 is 1
// when a record exists; every other field is UINT64_MAX when unobserved.
// Returns 1 for invalid arguments and 2 when the pool lock is busy.
#ifdef DDO_VMM_TIMING_COUNTERS
// Experimental CPU copy of the timing counters (55 fields; see
// ddo_vmm_timing_counters.h for the order). No CUDA call; safe at step callbacks.
// Experimental: current Windows memory query plus throttled-sampler peaks
// (13 fields; order in ddo_vmm_timing_counters.h). No pool lock is taken.
extern "C" DDO_VMM_API int ddo_exp_memory_observation_v1(int device, uint64_t* output, size_t count) noexcept {
    if (!output || count != memory_export_count || device < 0) return 1;
    try {
        std::array<uint64_t, memory_fields> current{};
        const bool ok = query_memory_fields(device, current);
#if defined(_WIN32)
        memory_snapshot(output, true, ok ? &current : nullptr);
#else
        memory_snapshot(output, false, nullptr);
#endif
        return 0;
    } catch (...) { return 3; }
}
// Experimental: device-usage composition rows taken with the throttled memory
// samples (31 fields; order in ddo_vmm_timing_counters.h). No CUDA call, no
// pool lock; UINT64_MAX marks what was not observed.
extern "C" DDO_VMM_API int ddo_exp_device_composition_v1(uint64_t* output, size_t count) noexcept {
    if (!output || count != composition_export_count) return 1;
    try { composition_sampler.snapshot(output); return 0; } catch (...) { return 3; }
}
extern "C" DDO_VMM_API int ddo_exp_timing_counters_v1(uint64_t* output, size_t count) noexcept {
    if (!output || count != timing_export_count) return 1;
    try {
#if defined(_WIN32)
        timing_snapshot(output, true);
#else
        timing_snapshot(output, false);
#endif
        return 0;
    } catch (...) { return 3; }
}
#endif
extern "C" DDO_VMM_API int ddo_exp_segmented_last_refusal(uint64_t* output, size_t count) noexcept {
    if (!output || count != capacity_report_fields) return 1;
    try {
        std::unique_lock lock(pool_mutex, std::try_to_lock);
        if (!lock) return 2;
        const auto& r = last_capacity_report;
        const uint64_t values[capacity_report_fields]{capacity_report_recorded ? 1u : 0u,
            r.request_bytes, r.deficit_bytes, r.owner_new, r.operation, r.status_first, r.status_barrier,
            r.status_continuation, r.rollback_fresh, r.rollback_released, r.rollback_retained,
            r.t1_pool_busy, r.t1_owners, r.t1_owners_busy, r.t1_scanned, r.t1_released, r.t1_pending,
            r.t1_live, r.t1_quarantined, r.t1_discarded, r.t1_excluded, r.t2_evictable, r.t2_busy_arenas,
            r.t2_evictions, r.t2_bytes, r.t2_status, r.t3_status, r.t3_fences, r.t3_released,
            r.retry_status, r.termination};
        std::copy(std::begin(values), std::end(values), output);
        return 0;
    } catch (...) { return 3; }
}
#endif
#endif

void* ddo_vmm_pool_malloc(size_t bytes, int device, cudaStream_t stream) noexcept {
#ifdef DDO_VMM_TIMING_COUNTERS
    // Sample before the timed scope and before any pool lock is taken.
    try {
        std::array<uint64_t, memory_fields> sampled{};
        if (sample_memory(device, sampled)) sample_composition(device, sampled);
    } catch (...) {}
    ddo_vmm_detail::TimingScope malloc_timing(ddo_vmm_detail::TimingOp::pool_malloc, bytes);
#endif
    PoolAllocationAttempt attempt(bytes, device, reinterpret_cast<uintptr_t>(stream), allocation_failure_sink);
    try {
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
        AutoRequestScope auto_scope;
        {
            std::scoped_lock lock(pool_mutex); allocation_started = true;
        }
#endif
        attempt.stage("context_validation");
        CUcontext context = nullptr;
        CUcontext stream_context = nullptr;
        CUdevice current_device = -1;
        if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context
            || cuCtxGetDevice(&current_device) != CUDA_SUCCESS || current_device != device
            || cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &stream_context) != CUDA_SUCCESS
            || stream_context != context) {
            failures.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        attempt.stage("allocation_granularity");
        const size_t normalized = bytes ? bytes : 512;
        const size_t granularity = ddo_vmm_device_granularity(device);
        if (!granularity || normalized > SIZE_MAX - (granularity - 1)) return nullptr;
        const size_t mapped = ((normalized + granularity - 1) / granularity) * granularity;
#if defined(_WIN32)
        // Preserve context validation and avoid holding the Torch pool lock while
        // checking execution slots. Reuse does not request more physical bytes.
        if (ddo_vmm_detail::recheck_execution_backing(device) != DDO_VMM_SUCCESS)
            return nullptr;
#endif
        // One pool attempt: fit reuse, then (automatic candidate) a new
        // segmented owner. nullopt continues to the monolithic region route.
        auto pool_attempt = [&]() -> std::optional<void*> {
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
            pending_capacity_refusal = {}; request_created_owner = false;
#endif
            {
                std::unique_lock lock(pool_mutex);
                for (int pass = 0; pass < 2; ++pass) {
                    attempt.stage("pool_fit_search");
                    std::shared_ptr<Backing> best;
                    size_t best_size = SIZE_MAX;
                    for (const auto& owner : backings) {
                        if (owner->retiring || owner->context != context || owner->device != device) continue;
                        if (pass) {
#ifdef DDO_VMM_SEGMENTED_POOL_CPU
                            if (owner->segmented) {
                                if (const int error = reap_segmented(*owner)) {
                                    attempt.stage("segmented_completion_query_error", error);
                                    return std::optional<void*>(nullptr);
                                }
                            } else
#endif
                            reap(*owner);
                        }
#ifdef DDO_VMM_NATIVE_JOINED_SPANS
                        const auto fit = owner->shared ? owner->ledger.fit_size_lowest(bytes, device)
                                                       : owner->ledger.fit_size_joined(bytes, device);
#else
                        const auto fit = owner->ledger.fit_size(bytes, device);
#endif
                        if (fit && (!best || *fit < best_size)) { best = owner; best_size = *fit; }
                    }
                    if (best) {
#ifdef DDO_VMM_SEGMENTED_POOL_CPU
                        if (best->segmented) return std::optional<void*>(publish_segmented(best, bytes, stream, attempt, lock));
#endif
                        auto pointer = publish_locked(best, bytes, stream, attempt);
#ifdef DDO_VMM_ALLOCATION_DIAGNOSTICS
                        if (pointer) allocation_diagnostics.pool_reuse();
#endif
                        return std::optional<void*>(pointer);
                    }
                }
            }
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
            if (auto_segmented) {
                int error = 0;
                std::shared_ptr<Backing> owner;
                // Growable shared owner first: created once per (device, context)
                // when no owner fits; afterwards a request that it cannot fit
                // contiguously falls back to a request-sized owner as before.
                bool shared_exists = false;
                uint64_t reserve = 0;
                {
                    std::scoped_lock lock(pool_mutex);
                    reserve = growable_reserve;
                    for (const auto& candidate : backings)
                        if (candidate->shared && candidate->device == device && candidate->context == context)
                            shared_exists = true;
                }
                if (reserve && mapped <= reserve && !shared_exists) {
                    owner = create_segmented_owner(device, context, static_cast<size_t>(reserve), error);
                    if (owner) {
                        std::scoped_lock lock(pool_mutex);
                        owner->shared = true;
                        growable_created.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        attempt.stage("segmented_shared_owner_create", error);
                        error = 0;
                    }
                }
                if (!owner) {
                    if (reserve) growable_fallbacks.fetch_add(1, std::memory_order_relaxed);
                    owner = create_segmented_owner(device, context, mapped, error);
                }
                if (!owner) { attempt.stage("segmented_owner_create", error); return std::optional<void*>(nullptr); }
                request_created_owner = true;
                std::unique_lock lock(pool_mutex);
                return std::optional<void*>(publish_segmented(owner, bytes, stream, attempt, lock));
            }
#endif
            return std::nullopt;
        };
        if (const auto first = pool_attempt()) {
            void* pointer = *first;
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
            if (!pointer && auto_segmented && pending_capacity_refusal.pending) {
                pointer = run_capacity_transaction(device, context, stream, bytes, [&]() -> void* {
                    const auto again = pool_attempt();
                    return again ? *again : nullptr;
                });
            }
            pending_capacity_refusal = {};
#endif
            return attempt.finish(pointer);
        }
        attempt.stage("pressure_prepare");
        const auto admission = ddo_vmm_pressure_prepare_cuda_allocation(device, mapped);
        if (admission != DDO_VMM_SUCCESS) {
#ifdef DDO_VMM_POOL_LIFETIME_TRACE
            {
                std::scoped_lock lock(pool_mutex);
                lifetime_trace.trigger(1, mapped, device, reinterpret_cast<uintptr_t>(stream), admission);
            }
#endif
            while (evict_one(false, 1, mapped)) {}
            const auto retry_admission = ddo_vmm_pressure_prepare_cuda_allocation(device, mapped);
            attempt.stage("pressure_prepare_after_eviction", retry_admission);
            if (retry_admission != DDO_VMM_SUCCESS) {
#ifdef DDO_VMM_POOL_GEOMETRY_DIAGNOSTICS
                geometry_failure(bytes, mapped, device, context, stream, retry_admission, 1);
#endif
                return nullptr;
            }
        }
        ddo_vmm_region* raw = nullptr;
        attempt.stage("region_create");
        const auto creation = ddo_vmm_region_create(device, normalized, &raw);
        if (creation != DDO_VMM_SUCCESS) {
            attempt.region_failure(creation, ddo_vmm_last_error());
            if (creation != DDO_VMM_OUT_OF_MEMORY) return nullptr;
#ifdef DDO_VMM_SEGMENTED_POOL_CPU
            const auto recovery = recover_segmented_context(device, context);
            if (recovery.error) {
                attempt.stage("segmented_recovery_error", recovery.error); return nullptr;
            }
            if (recovery.released) {
                // Exactly one new allocation attempt, and only after real safe
                // release. Do not chain another segmented/whole recovery here.
                const auto retry = ddo_vmm_region_create(device, normalized, &raw);
                if (retry != DDO_VMM_SUCCESS) { attempt.region_failure(retry, ddo_vmm_last_error()); return nullptr; }
            } else {
#endif
#ifdef DDO_VMM_POOL_LIFETIME_TRACE
            {
                std::scoped_lock lock(pool_mutex);
                lifetime_trace.trigger(2, mapped, device, reinterpret_cast<uintptr_t>(stream), creation);
            }
#endif
            while (evict_one(false, 2, mapped)) {}
            const auto retry_creation = allocation_attempt(
                [&] {
                    const auto result = ddo_vmm_region_create(device, normalized, &raw);
                    if (result != DDO_VMM_SUCCESS) {
                        attempt.region_failure(result, ddo_vmm_last_error());
                    }
                    return result;
                },
                [&] { return ddo_vmm_pressure_evict_one(device); },
                DDO_VMM_OUT_OF_MEMORY, true);
            if (retry_creation != DDO_VMM_SUCCESS) {
#ifdef DDO_VMM_POOL_GEOMETRY_DIAGNOSTICS
                geometry_failure(bytes, mapped, device, context, stream, retry_creation, 2);
#endif
                return nullptr;
            }
#ifdef DDO_VMM_SEGMENTED_POOL_CPU
            }
#endif
        }
        RegionOwner region(raw, &ddo_vmm_region_destroy);
        std::scoped_lock lock(pool_mutex);
        attempt.stage("generation_limit");
        if (next_generation == UINT64_MAX) return nullptr;
        attempt.stage("backing_create");
        auto owner = std::make_shared<Backing>(std::move(region), device, context, next_generation++);
        backings.push_back(owner);
#ifdef DDO_VMM_ALLOCATION_DIAGNOSTICS
        if (allocation_diagnostics.enabled.load(std::memory_order_relaxed)) {
            uint64_t pool_mapped = 0;
            for (const auto& backing : backings) pool_mapped += backing->ledger.snapshot().mapped;
            allocation_diagnostics.pool_backing(pool_mapped);
        }
#endif
        return attempt.finish(publish_locked(owner, bytes, stream, attempt));
    } catch (...) { attempt.exception(); failures.fetch_add(1, std::memory_order_relaxed); return nullptr; }
}

void ddo_vmm_pool_free(void* pointer, cudaStream_t stream) noexcept {
#ifdef DDO_VMM_TIMING_COUNTERS
    ddo_vmm_detail::TimingScope free_timing(ddo_vmm_detail::TimingOp::pool_free);
#endif
    if (!pointer) return;
    try {
        {
            std::scoped_lock lock(pool_mutex);
            auto it = records.find(pointer);
            if (it == records.end()) return;
            auto& record = it->second;
            auto fence = std::make_shared<CudaFence>();
            fence->context = record.backing->context;
            fence->stream = identity(stream, record.backing->device);
            CUcontext current = nullptr;
            CUcontext stream_context = nullptr;
            const bool recorded = cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == fence->context
                && cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &stream_context) == CUDA_SUCCESS
                && stream_context == fence->context
                && cuEventCreate(&fence->event, CU_EVENT_DISABLE_TIMING) == CUDA_SUCCESS
                && cuEventRecord(fence->event, reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS;
            const bool retired = recorded && record.backing->ledger.retire(record.lease, fence);
            if (!retired) {
                record.backing->ledger.retire(record.lease, {});
                failures.fetch_add(1, std::memory_order_relaxed);
            }
#ifdef DDO_VMM_POOL_LIFETIME_TRACE
            trace_lease(2, *record.backing, record.lease, stream, fence->event, retired ? 0 : 1);
#endif
            records.erase(it);
        }
        while (evict_one(true, 3)) {}
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
        if (auto_segmented && trim_segmented_idle(true)) failures.fetch_add(1, std::memory_order_relaxed);
        if (auto_segmented && growable_reserve && trim_shared_units()) failures.fetch_add(1, std::memory_order_relaxed);
#endif
    } catch (...) { failures.fetch_add(1, std::memory_order_relaxed); }
}

ddo_vmm_detail::ArenaPoolRelease ddo_vmm_pool_release_for_arena(
    int device, uintptr_t context_address, size_t required) noexcept {
    ArenaPoolRelease result;
    try {
        CUcontext context = nullptr;
        CUdevice current_device = -1;
        if (cuCtxGetCurrent(&context) != CUDA_SUCCESS
            || reinterpret_cast<uintptr_t>(context) != context_address
            || cuCtxGetDevice(&current_device) != CUDA_SUCCESS || current_device != device) {
            result.status = CompletedRelease::failed;
            return result;
        }
        result = release_completed_pool(pool_mutex, backings, device, context_address, required,
            [&](Backing& owner, const SpanFenceOwner& dependency) {
                const auto fence = std::static_pointer_cast<const CudaFence>(dependency);
                const auto status = fence->context == context
                    ? cuEventQuery(fence->event) : CUDA_ERROR_INVALID_CONTEXT;
#ifdef DDO_VMM_POOL_LIFETIME_TRACE
                trace_dependency(3, 0, owner.address, 0, 0, *fence, {}, nullptr, status);
#else
                (void)owner;
#endif
                return status == CUDA_SUCCESS ? SpanCompletion::complete
                    : status == CUDA_ERROR_NOT_READY ? SpanCompletion::not_ready
                    : SpanCompletion::failed;
            }, [&](Backing& owner) {
                eviction_attempts.fetch_add(1, std::memory_order_relaxed);
                return ddo_vmm_region_evict(owner.region.get()) == DDO_VMM_SUCCESS;
            }, [&]([[maybe_unused]] Backing& owner, [[maybe_unused]] CompletedRelease outcome,
                    [[maybe_unused]] size_t bytes) {
#ifdef DDO_VMM_POOL_LIFETIME_TRACE
                lifetime_trace.record(outcome == CompletedRelease::released ? 4 : 5,
                    0, owner.address, bytes, 5, 0, required, 0, 0, 0, 0,
                    outcome == CompletedRelease::released ? 0 : 1);
#endif
            }, [&](Backing& owner) {
                // Physical eviction was checked. The legacy void destroy only
                // provides a returned-call receipt for empty VA/context cleanup.
                owner.region.reset();
            });
        if (result.status == CompletedRelease::failed)
            failures.fetch_add(1, std::memory_order_relaxed);
    } catch (...) {
        failures.fetch_add(1, std::memory_order_relaxed);
        result.status = CompletedRelease::failed;
    }
    return result;
}

int ddo_vmm_pool_trim_idle() noexcept {
    // Explicit phase boundary only. Never selects leased or quarantined backing.
    // evict_one proves pending completion and checks unmap/release before commit.
    const auto before = failures.load(std::memory_order_relaxed);
    try {
        while (evict_one(false, 4)) {}
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
        if (auto_segmented && trim_segmented_idle(false)) return 1;
#endif
        return failures.load(std::memory_order_relaxed) == before ? 0 : 1;
    } catch (...) {
        failures.fetch_add(1, std::memory_order_relaxed);
        return 1;
    }
}

int ddo_vmm_pool_stats(uint64_t* values, size_t count) noexcept {
    if (!values || count != 15) return 1;
    try {
        uint64_t result[15]{1}; // Enabled, then unique physical/partition gauges.
        std::scoped_lock lock(pool_mutex);
        for (const auto& owner : backings) {
            auto s = owner->ledger.snapshot();
#ifdef DDO_VMM_SEGMENTED_POOL_CPU
            if (owner->segmented) {
                if (owner->retiring || !owner->segmented->physical_snapshot(owner->ledger, s)) return 1;
            }
#endif
            result[1] += s.mapped; result[2] += s.requested; result[3] += s.leased;
            if (owner->retiring) result[5] += s.available + s.pending;
            else { result[4] += s.available; result[5] += s.pending; }
            result[6] += s.quarantined;
            result[7] += s.live_count; result[8] += s.reserved_count; ++result[9];
            if (idle(*owner)) { result[10] += s.mapped; ++result[14]; }
        }
        result[11] = eviction_attempts.load(std::memory_order_relaxed);
        result[12] = eviction_wait_ns.load(std::memory_order_relaxed);
        result[13] = failures.load(std::memory_order_relaxed);
        for (size_t i = 0; i < count; ++i) values[i] = result[i];
        return 0;
    } catch (...) { return 1; }
}

#ifdef DDO_VMM_MAPPING_TRACE
extern "C" DDO_VMM_API int ddo_vmm_owner_observe_v1(
    uint64_t address, uint64_t bytes, uint64_t* output, size_t count) {
    try {
        std::scoped_lock lock(pool_mutex);
        return ddo_vmm_detail::observe_owner(address, bytes, output, count,
            [](uint64_t query, ddo_vmm_detail::OwnerMetadata& result) {
                const auto it = records.find(reinterpret_cast<void*>(static_cast<uintptr_t>(query)));
                if (it == records.end()) return false;
                const auto& record = it->second;
                const auto& backing = *record.backing;
                result = {backing.address, record.lease.generation, record.lease.id,
                    record.lease.requested, record.lease.offset, record.lease.size,
                    static_cast<uint64_t>(backing.device),
                    reinterpret_cast<uintptr_t>(backing.context), backing.retiring ? 1ULL : 0ULL};
                return true;
            });
    } catch (...) { return 1; }
}
#endif

#ifdef DDO_VMM_POOL_LIFETIME_TRACE
// Candidate-only exports. Stream values are raw handles, not proven unique
// identities. Sequence gaps mean overwritten history, not absent allocations.
extern "C" DDO_VMM_API int ddo_vmm_pool_lifetime_trace_enable_v1() {
    std::scoped_lock lock(pool_mutex);
    if (!lifetime_trace.start()) return 1;
    return dependency_trace.start() ? 0 : 1;
}
extern "C" DDO_VMM_API int ddo_vmm_pool_dependency_trace_snapshot_v1(
    uint64_t* report, size_t report_count, uint64_t* rows, size_t row_capacity) {
    std::scoped_lock lock(pool_mutex);
    return dependency_trace.snapshot(report, report_count, rows, row_capacity);
}
extern "C" DDO_VMM_API int ddo_vmm_pool_lifetime_trace_snapshot_v1(
    uint64_t* report, size_t report_count, uint64_t* rows, size_t row_capacity) {
    std::scoped_lock lock(pool_mutex);
    return lifetime_trace.snapshot(report, report_count, rows, row_capacity);
}
#endif
