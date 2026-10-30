#include "ddo_vmm.h"
#include "ddo_vmm_execution_recheck.h"
#include "ddo_vmm_torch_pool.h"
#include "ddo_vmm_span_ledger.h"
#include "ddo_vmm_stream_dependency.h"
#include "ddo_vmm_allocation_diagnostics.h"
#include "ddo_vmm_allocation_failure_trace.h"
#include "ddo_vmm_allocation_attempt.h"
#include "ddo_vmm_owner_observation.h"
#include <cstdio>
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
    SpanLedger ledger;
    CUcontext context;
    uintptr_t address;
    int device;
    bool retiring = false;
    Backing(RegionOwner owner, int ordinal, CUcontext ctx, uint64_t generation)
        : region(std::move(owner)), ledger(ddo_vmm_region_mapped_size(region.get()), ordinal, generation),
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
            if (owner->retiring || !idle(*owner)) continue;
            const auto mapped = owner->ledger.snapshot().mapped;
            if (mapped > SIZE_MAX - idle_bytes) return false;
            idle_bytes += mapped;
        }
        if (over_limit_only && idle_bytes <= cache_limit()) return false;
        for (const auto& owner : backings) {
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
                     PoolAllocationAttempt& attempt) {
    attempt.stage("reserve_lease");
#ifdef DDO_VMM_NATIVE_JOINED_SPANS
    auto lease = owner->ledger.reserve_joined(bytes, owner->device);
#else
    auto lease = owner->ledger.reserve(bytes, owner->device);
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
}

void* ddo_vmm_pool_malloc(size_t bytes, int device, cudaStream_t stream) noexcept {
    PoolAllocationAttempt attempt(bytes, device, reinterpret_cast<uintptr_t>(stream), allocation_failure_sink);
    try {
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
        {
            std::scoped_lock lock(pool_mutex);
            for (int pass = 0; pass < 2; ++pass) {
                attempt.stage("pool_fit_search");
                std::shared_ptr<Backing> best;
                size_t best_size = SIZE_MAX;
                for (const auto& owner : backings) {
                    if (owner->retiring || owner->context != context || owner->device != device) continue;
                    if (pass) reap(*owner);
#ifdef DDO_VMM_NATIVE_JOINED_SPANS
                    const auto fit = owner->ledger.fit_size_joined(bytes, device);
#else
                    const auto fit = owner->ledger.fit_size(bytes, device);
#endif
                    if (fit && (!best || *fit < best_size)) { best = owner; best_size = *fit; }
                }
                if (best) {
                    auto pointer = publish_locked(best, bytes, stream, attempt);
#ifdef DDO_VMM_ALLOCATION_DIAGNOSTICS
                    if (pointer) allocation_diagnostics.pool_reuse();
#endif
                    return attempt.finish(pointer);
                }
            }
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
            if (retry_admission != DDO_VMM_SUCCESS) return nullptr;
        }
        ddo_vmm_region* raw = nullptr;
        attempt.stage("region_create");
        const auto creation = ddo_vmm_region_create(device, normalized, &raw);
        if (creation != DDO_VMM_SUCCESS) {
            attempt.region_failure(creation, ddo_vmm_last_error());
            if (creation != DDO_VMM_OUT_OF_MEMORY) return nullptr;
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
            if (retry_creation != DDO_VMM_SUCCESS) return nullptr;
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
    } catch (...) { failures.fetch_add(1, std::memory_order_relaxed); }
}

int ddo_vmm_pool_trim_idle() noexcept {
    // Explicit phase boundary only. Never selects leased or quarantined backing.
    // evict_one proves pending completion and checks unmap/release before commit.
    const auto before = failures.load(std::memory_order_relaxed);
    try {
        while (evict_one(false, 4)) {}
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
            const auto s = owner->ledger.snapshot();
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
