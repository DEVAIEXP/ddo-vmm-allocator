#include "ddo_vmm.h"
#include "ddo_vmm_torch_pool.h"
#include "ddo_vmm_span_ledger.h"
#include "ddo_vmm_stream_dependency.h"
#include "ddo_vmm_allocation_diagnostics.h"
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
        if (status == CUDA_SUCCESS) owner.ledger.observe(dependency, SpanCompletion::complete);
        else if (status != CUDA_ERROR_NOT_READY) {
            owner.ledger.observe(dependency, SpanCompletion::failed);
            failures.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

// Select/mark under the ledger lock, then prove completion and release outside
// it. Pending bytes remain visible and unavailable throughout the transaction.
bool evict_one(bool over_limit_only) {
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
    }
    eviction_attempts.fetch_add(1, std::memory_order_relaxed);
    const auto started = std::chrono::steady_clock::now();
    bool completed = true;
    {
        ContextScope scope(selected->context);
        completed = scope.active;
        for (const auto& dependency : fences) {
            const auto fence = std::static_pointer_cast<const CudaFence>(dependency);
            if (!scope.active || cuEventSynchronize(fence->event) != CUDA_SUCCESS) completed = false;
        }
    }
    eviction_wait_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
    {
        std::scoped_lock lock(pool_mutex);
        for (const auto& fence : fences)
            selected->ledger.observe(fence, completed ? SpanCompletion::complete : SpanCompletion::failed);
        if (!completed || !selected->ledger.begin_backing_release()) {
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

void* publish_locked(const std::shared_ptr<Backing>& owner, size_t bytes, cudaStream_t stream) {
#ifdef DDO_VMM_NATIVE_JOINED_SPANS
    auto lease = owner->ledger.reserve_joined(bytes, owner->device);
#else
    auto lease = owner->ledger.reserve(bytes, owner->device);
#endif
    if (!lease) return nullptr;
    for (const auto& dependency : lease->dependencies) {
        const auto fence = std::static_pointer_cast<const CudaFence>(dependency);
        if (fence->context != owner->context || !enqueue_reuse_dependency(fence->event,
            reinterpret_cast<CUstream>(stream), cuStreamWaitEvent, fence->stream, identity(stream, owner->device))) {
            owner->ledger.publish(*lease, false);
            failures.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
    }
    if (lease->offset > UINTPTR_MAX - owner->address) {
        owner->ledger.publish(*lease, false);
        return nullptr;
    }
    void* pointer = reinterpret_cast<void*>(owner->address + lease->offset);
    try {
        const auto [entry, inserted] = records.emplace(pointer, Record{owner, *lease});
        if (!inserted) { owner->ledger.publish(*lease, false); return nullptr; }
        if (!owner->ledger.publish(*lease, true)) { records.erase(entry); return nullptr; }
    } catch (...) {
        // Submission may already exist. Preserve backing + predecessor owners.
        owner->ledger.publish(*lease, false);
        throw;
    }
    return pointer;
}
}

void* ddo_vmm_pool_malloc(size_t bytes, int device, cudaStream_t stream) noexcept {
    try {
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
        const size_t normalized = bytes ? bytes : 512;
        const size_t granularity = ddo_vmm_device_granularity(device);
        if (!granularity || normalized > SIZE_MAX - (granularity - 1)) return nullptr;
        const size_t mapped = ((normalized + granularity - 1) / granularity) * granularity;
        {
            std::scoped_lock lock(pool_mutex);
            for (int pass = 0; pass < 2; ++pass) {
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
                    auto pointer = publish_locked(best, bytes, stream);
#ifdef DDO_VMM_ALLOCATION_DIAGNOSTICS
                    if (pointer) allocation_diagnostics.pool_reuse();
#endif
                    return pointer;
                }
            }
        }
        if (ddo_vmm_pressure_prepare_cuda_allocation(device, mapped) != DDO_VMM_SUCCESS) {
            while (evict_one(false)) {}
            if (ddo_vmm_pressure_prepare_cuda_allocation(device, mapped) != DDO_VMM_SUCCESS) return nullptr;
        }
        ddo_vmm_region* raw = nullptr;
        if (ddo_vmm_region_create(device, normalized, &raw) != DDO_VMM_SUCCESS) {
            while (evict_one(false)) {}
            while (ddo_vmm_region_create(device, normalized, &raw) != DDO_VMM_SUCCESS) {
                if (!ddo_vmm_pressure_evict_one(device)) return nullptr;
            }
        }
        RegionOwner region(raw, &ddo_vmm_region_destroy);
        std::scoped_lock lock(pool_mutex);
        if (next_generation == UINT64_MAX) return nullptr;
        auto owner = std::make_shared<Backing>(std::move(region), device, context, next_generation++);
        backings.push_back(owner);
#ifdef DDO_VMM_ALLOCATION_DIAGNOSTICS
        if (allocation_diagnostics.enabled.load(std::memory_order_relaxed)) {
            uint64_t pool_mapped = 0;
            for (const auto& backing : backings) pool_mapped += backing->ledger.snapshot().mapped;
            allocation_diagnostics.pool_backing(pool_mapped);
        }
#endif
        return publish_locked(owner, bytes, stream);
    } catch (...) { failures.fetch_add(1, std::memory_order_relaxed); return nullptr; }
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
            if (!recorded || !record.backing->ledger.retire(record.lease, fence)) {
                record.backing->ledger.retire(record.lease, {});
                failures.fetch_add(1, std::memory_order_relaxed);
            }
            records.erase(it);
        }
        while (evict_one(true)) {}
    } catch (...) { failures.fetch_add(1, std::memory_order_relaxed); }
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
