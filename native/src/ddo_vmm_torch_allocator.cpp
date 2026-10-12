// Optional PyTorch CUDAPluggableAllocator entry points.
//
// The API is intentionally small: PyTorch owns allocation scheduling and calls
// these functions; this DLL owns the CUDA VMM region lifetime.  It is not a
// general-purpose multi-stream caching allocator: callers must still observe
// PyTorch's stream lifetime contract. Cache reuse is stream-ordered here.

#include "ddo_vmm.h"
#include "ddo_vmm_stream_dependency.h"
#include "ddo_vmm_allocation_footprint.h"
#include "ddo_vmm_cache_selection.h"
#ifdef DDO_VMM_NATIVE_SPAN_POOL
#include "ddo_vmm_torch_pool.h"
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr size_t kMinimumTorchAllocationBytes = 512;
// Process-lifetime observation only; no CUDA queries or policy changes.
std::atomic<uint64_t> reset_sync_calls{0}, reset_sync_ns{0}, reset_failures{0};
std::atomic<uint64_t> cache_evictions{0}, cache_evicted_bytes{0}, cache_eviction_sync_ns{0};

struct AllocationRecord {
    std::unique_ptr<ddo_vmm_region, decltype(&ddo_vmm_region_destroy)> region{
        nullptr, &ddo_vmm_region_destroy
    };
    size_t mapped_size = 0;
    ddo_vmm_detail::AllocationFootprint footprint;
    int device_ordinal = -1;
    CUevent completion_event = nullptr;
    ddo_vmm_detail::StreamIdentity completion_stream;
    bool completion_unproven = false;
    bool cache_eviction = false;

    AllocationRecord() = default;
    AllocationRecord(const AllocationRecord&) = delete;
    AllocationRecord& operator=(const AllocationRecord&) = delete;

    AllocationRecord(AllocationRecord&& other) noexcept
        : region(std::move(other.region)),
          mapped_size(other.mapped_size),
          footprint(std::exchange(other.footprint, {})),
          device_ordinal(other.device_ordinal),
          completion_event(std::exchange(other.completion_event, nullptr)),
          completion_stream(std::exchange(other.completion_stream, {})),
          completion_unproven(std::exchange(other.completion_unproven, false)) {
        cache_eviction = std::exchange(other.cache_eviction, false);
        other.mapped_size = 0;
        other.device_ordinal = -1;
    }

    AllocationRecord& operator=(AllocationRecord&& other) noexcept {
        if (this != &other) {
            reset();
            region = std::move(other.region);
            mapped_size = other.mapped_size;
            footprint = std::exchange(other.footprint, {});
            device_ordinal = other.device_ordinal;
            completion_event = std::exchange(other.completion_event, nullptr);
            completion_stream = std::exchange(other.completion_stream, {});
            completion_unproven = std::exchange(other.completion_unproven, false);
            cache_eviction = std::exchange(other.cache_eviction, false);
            other.mapped_size = 0;
            other.device_ordinal = -1;
        }
        return *this;
    }

    ~AllocationRecord() { reset(); }

    void reset() noexcept {
        if (cache_eviction) {
            cache_evictions.fetch_add(1, std::memory_order_relaxed);
            cache_evicted_bytes.fetch_add(mapped_size, std::memory_order_relaxed);
        }
        auto synchronize = [this](CUevent event) {
            const auto start = std::chrono::steady_clock::now();
            const auto status = cuEventSynchronize(event);
            const auto ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count());
            reset_sync_calls.fetch_add(1, std::memory_order_relaxed);
            reset_sync_ns.fetch_add(ns, std::memory_order_relaxed);
            if (cache_eviction) cache_eviction_sync_ns.fetch_add(ns, std::memory_order_relaxed);
            return status;
        };
        if (completion_unproven || !ddo_vmm_detail::drain_completion(
                completion_event, synchronize, cuEventDestroy)) {
            reset_failures.fetch_add(1, std::memory_order_relaxed);
            // CUDA could not prove completion. Fail closed: retain the mapping
            // until process teardown instead of unmapping potentially live work.
            std::fprintf(stderr, "DDO VMM: retaining %zu bytes after CUDA completion failure\n", mapped_size);
            (void)region.release();
            completion_event = nullptr;
        }
        region.reset();
        completion_stream = {};
        completion_unproven = false;
        cache_eviction = false;
        mapped_size = 0;
        footprint = {};
        device_ordinal = -1;
    }
};

std::mutex allocations_mutex;
std::unordered_map<void*, AllocationRecord> allocations;
std::multimap<size_t, AllocationRecord> cached_allocations;
size_t live_mapped_bytes = 0;
size_t cached_mapped_bytes = 0;

std::string environment_value(const char* name) {
#if defined(_WIN32)
    char* raw_value = nullptr;
    size_t raw_length = 0;
    if (_dupenv_s(&raw_value, &raw_length, name) != 0 || raw_value == nullptr) {
        return {};
    }
    std::string value(raw_value, raw_length == 0 ? 0 : raw_length - 1);
    std::free(raw_value);
    return value;
#else
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
#endif
}

size_t cache_limit_bytes() {
    const std::string value = environment_value("DDO_VMM_TORCH_CACHE_MIB");
    if (value.empty()) {
        return 0;
    }
    char* end = nullptr;
    const unsigned long long mebibytes = std::strtoull(value.c_str(), &end, 10);
    constexpr size_t bytes_per_mebibyte = 1024ULL * 1024ULL;
    if (
        end == value.c_str() || *end != '\0'
        || mebibytes > std::numeric_limits<size_t>::max() / bytes_per_mebibyte
    ) {
        return 0;
    }
    return static_cast<size_t>(mebibytes) * bytes_per_mebibyte;
}

void release_records(std::vector<AllocationRecord>& records) {
    for (AllocationRecord& record : records) {
        record.reset();
    }
}

ddo_vmm_detail::StreamIdentity stream_identity(cudaStream_t stream, int device) {
    unsigned long long id = 0;
    // CUDA stream IDs are unique for the process lifetime, including special
    // streams. Unknown identity preserves the original event-wait path.
    if (cuStreamGetId(reinterpret_cast<CUstream>(stream), &id) != CUDA_SUCCESS)
        return {};
    return {id, device, true};
}

bool record_stream_completion(AllocationRecord* record, cudaStream_t stream) {
    if (record == nullptr) {
        return false;
    }
    CUevent event = nullptr;
    const auto identity = stream_identity(stream, record->device_ordinal);
    if (
        cuEventCreate(&event, CU_EVENT_DISABLE_TIMING) == CUDA_SUCCESS
        && cuEventRecord(event, reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS
    ) {
        // The new fence includes the queued reuse dependency and all work on
        // the consumer stream. Only now may the previous fence be retired.
        if (record->completion_event != nullptr) {
            cuEventDestroy(record->completion_event);
        }
        record->completion_event = event;
        record->completion_stream = identity;
        return true;
    }
    if (event != nullptr) {
        cuEventDestroy(event);
    }
    // If event setup failed, synchronize the freeing stream rather than
    // risk releasing a virtual mapping that a queued kernel can still read.
    if (cuStreamSynchronize(reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS) {
        if (record->completion_event != nullptr) {
            cuEventDestroy(record->completion_event);
            record->completion_event = nullptr;
        }
        record->completion_stream = {};
        return true;
    }
    record->completion_unproven = true;
    return false;
}

bool take_cached_allocation(
    size_t minimum_mapped_size,
    int device_ordinal,
    AllocationRecord* out_record
) {
    std::scoped_lock lock(allocations_mutex);
    const auto candidate = ddo_vmm_detail::find_exact_cached_allocation(
        cached_allocations, minimum_mapped_size, device_ordinal);
    if (candidate != cached_allocations.end()) {
        *out_record = std::move(candidate->second);
        cached_mapped_bytes -= out_record->mapped_size;
        cached_allocations.erase(candidate);
        return true;
    }
    return false;
}

std::vector<AllocationRecord> drain_cached_allocations() {
    std::vector<AllocationRecord> evicted;
    std::scoped_lock lock(allocations_mutex);
    evicted.reserve(cached_allocations.size());
    for (auto& [size, record] : cached_allocations) {
        (void)size;
        record.cache_eviction = true;
        evicted.push_back(std::move(record));
    }
    cached_allocations.clear();
    cached_mapped_bytes = 0;
    return evicted;
}

}  // namespace

extern "C" DDO_VMM_API void* ddo_vmm_torch_malloc(
    size_t size_bytes,
    int device_ordinal,
    cudaStream_t stream
) {
#ifdef DDO_VMM_NATIVE_SPAN_POOL
    return ddo_vmm_pool_malloc(size_bytes, device_ordinal, stream);
#else
    // PyTorch may request a zero-byte tensor.  It still needs a unique,
    // non-null allocation record to preserve its allocator bookkeeping.
    const size_t requested_size = size_bytes == 0 ? kMinimumTorchAllocationBytes : size_bytes;
    const size_t granularity = ddo_vmm_device_granularity(device_ordinal);
    if (granularity == 0 || requested_size > std::numeric_limits<size_t>::max() - (granularity - 1)) {
        return nullptr;
    }
    const size_t minimum_mapped_size = ((requested_size + granularity - 1) / granularity) * granularity;

    AllocationRecord record;
    if (!take_cached_allocation(minimum_mapped_size, device_ordinal, &record)) {
        // Ordinary PyTorch allocations pass through this native entry point.
        // When the optional WDDM headroom guard is active, recover unpinned
        // model-arena pages before asking the driver for another mapping.
        // This is allocator-wide coordination, not runner-side residency.
        if (ddo_vmm_pressure_prepare_cuda_allocation(device_ordinal, minimum_mapped_size) != DDO_VMM_SUCCESS) {
            auto evicted = drain_cached_allocations();
            release_records(evicted);
            if (ddo_vmm_pressure_prepare_cuda_allocation(device_ordinal, minimum_mapped_size) != DDO_VMM_SUCCESS) {
                return nullptr;
            }
        }
        ddo_vmm_region* raw_region = nullptr;
        if (ddo_vmm_region_create(device_ordinal, requested_size, &raw_region) != DDO_VMM_SUCCESS) {
            // A full cache must never make a new allocation fail.  Release
            // cached physical backing and retry once before reporting OOM.
            auto evicted = drain_cached_allocations();
            release_records(evicted);
            if (ddo_vmm_region_create(device_ordinal, requested_size, &raw_region) != DDO_VMM_SUCCESS) {
                // Model-weight mappings retain stable virtual ranges. Reclaim
                // one eligible native mapping at a time and retry until no
                // safe page or region remains.
                while (ddo_vmm_pressure_evict_one(device_ordinal)) {
                    if (ddo_vmm_region_create(device_ordinal, requested_size, &raw_region) == DDO_VMM_SUCCESS) {
                        break;
                    }
                }
                if (raw_region == nullptr) {
                    return nullptr;
                }
            }
        }
        record.region = std::unique_ptr<ddo_vmm_region, decltype(&ddo_vmm_region_destroy)>(
            raw_region, &ddo_vmm_region_destroy
        );
        record.mapped_size = ddo_vmm_region_mapped_size(raw_region);
        record.device_ordinal = device_ordinal;
    }
    // The current request, not the original request that created cached backing.
    record.footprint = {size_bytes, minimum_mapped_size, record.mapped_size};
    if (!ddo_vmm_detail::enqueue_reuse_dependency(
            record.completion_event, reinterpret_cast<CUstream>(stream), cuStreamWaitEvent,
            record.completion_stream,
            record.completion_event != nullptr ? stream_identity(stream, device_ordinal)
                                               : ddo_vmm_detail::StreamIdentity{})) {
        // Do not publish the pointer after a failed dependency. The record's
        // destructor drains the retained fence, or retains backing on failure.
        return nullptr;
    }
    const uintptr_t address = ddo_vmm_region_address(record.region.get());
    if (address == 0 || record.mapped_size == 0) {
        return nullptr;
    }
    void* pointer = reinterpret_cast<void*>(address);

    std::scoped_lock lock(allocations_mutex);
    const auto [entry, inserted] = allocations.emplace(
        pointer, std::move(record)
    );
    if (!inserted) {
        return nullptr;
    }
    if (entry->second.mapped_size > std::numeric_limits<size_t>::max() - live_mapped_bytes) {
        allocations.erase(entry);
        return nullptr;
    }
    live_mapped_bytes += entry->second.mapped_size;
    return pointer;
#endif
}

extern "C" DDO_VMM_API void ddo_vmm_torch_free(
    void* pointer,
    size_t /*size_bytes*/,
    int /*device_ordinal*/,
    cudaStream_t stream
) {
#ifdef DDO_VMM_NATIVE_SPAN_POOL
    ddo_vmm_pool_free(pointer, stream);
#else
    if (pointer == nullptr) {
        return;
    }
    AllocationRecord record;
    std::vector<AllocationRecord> evicted;
    {
        std::scoped_lock lock(allocations_mutex);
        const auto found = allocations.find(pointer);
        if (found == allocations.end()) {
            return;
        }
        record = std::move(found->second);
        live_mapped_bytes -= record.mapped_size;
        allocations.erase(found);

        const bool completion_recorded = record_stream_completion(&record, stream);

        const size_t cache_limit = cache_limit_bytes();
        if (completion_recorded && cache_limit != 0 && record.mapped_size <= cache_limit) {
            while (
                !cached_allocations.empty()
                && (cached_mapped_bytes > cache_limit || record.mapped_size > cache_limit - cached_mapped_bytes)
            ) {
                auto candidate = cached_allocations.begin();
                candidate->second.cache_eviction = true;
                cached_mapped_bytes -= candidate->second.mapped_size;
                evicted.push_back(std::move(candidate->second));
                cached_allocations.erase(candidate);
            }
            cached_mapped_bytes += record.mapped_size;
            cached_allocations.emplace(record.mapped_size, std::move(record));
            return;
        }
    }
    // Release outside the lock because CUDA Driver calls may block.
    record.reset();
    release_records(evicted);
#endif
}

extern "C" DDO_VMM_API int ddo_vmm_torch_pool_stats_v1(uint64_t* values, size_t count) {
#ifdef DDO_VMM_NATIVE_SPAN_POOL
    return ddo_vmm_pool_stats(values, count);
#else
    (void)values; (void)count;
    return 2; // Unsupported, not an empty allocator snapshot.
#endif
}

extern "C" DDO_VMM_API int ddo_vmm_torch_trim_idle_v1(void) {
#ifdef DDO_VMM_NATIVE_SPAN_POOL
    return ddo_vmm_pool_trim_idle();
#else
    return 2; // Unsupported builds must not simulate successful release.
#endif
}

extern "C" DDO_VMM_API size_t ddo_vmm_torch_live_allocation_count(void) {
#ifdef DDO_VMM_NATIVE_SPAN_POOL
    uint64_t values[15]{};
    return ddo_vmm_pool_stats(values, 15) == 0 ? static_cast<size_t>(values[7]) : 0;
#else
    std::scoped_lock lock(allocations_mutex);
    return allocations.size();
#endif
}

extern "C" DDO_VMM_API size_t ddo_vmm_torch_live_mapped_bytes(void) {
#ifdef DDO_VMM_NATIVE_SPAN_POOL
    uint64_t values[15]{};
    return ddo_vmm_pool_stats(values, 15) == 0 ? static_cast<size_t>(values[1] - values[10]) : 0;
#else
    std::scoped_lock lock(allocations_mutex);
    return live_mapped_bytes;
#endif
}

extern "C" DDO_VMM_API size_t ddo_vmm_torch_cached_allocation_count(void) {
#ifdef DDO_VMM_NATIVE_SPAN_POOL
    uint64_t values[15]{};
    return ddo_vmm_pool_stats(values, 15) == 0 ? static_cast<size_t>(values[14]) : 0;
#else
    std::scoped_lock lock(allocations_mutex);
    return cached_allocations.size();
#endif
}

extern "C" DDO_VMM_API size_t ddo_vmm_torch_cached_mapped_bytes(void) {
#ifdef DDO_VMM_NATIVE_SPAN_POOL
    uint64_t values[15]{};
    return ddo_vmm_pool_stats(values, 15) == 0 ? static_cast<size_t>(values[10]) : 0;
#else
    std::scoped_lock lock(allocations_mutex);
    return cached_mapped_bytes;
#endif
}

extern "C" DDO_VMM_API int ddo_vmm_torch_release_stats_v1(uint64_t* values, size_t count) {
    if (values == nullptr || count != 6) return 1;
#ifdef DDO_VMM_NATIVE_SPAN_POOL
    return 2; // Pool uses its own versioned release/accounting scope.
#else
    values[0] = reset_sync_calls.load(std::memory_order_relaxed);
    values[1] = reset_sync_ns.load(std::memory_order_relaxed);
    values[2] = reset_failures.load(std::memory_order_relaxed);
    values[3] = cache_evictions.load(std::memory_order_relaxed);
    values[4] = cache_evicted_bytes.load(std::memory_order_relaxed);
    values[5] = cache_eviction_sync_ns.load(std::memory_order_relaxed);
    return 0;
#endif
}

extern "C" DDO_VMM_API int ddo_vmm_torch_allocation_footprint_v1(uint64_t* values, size_t count) {
    if (values == nullptr || count != 6) return 1;
#ifdef DDO_VMM_NATIVE_SPAN_POOL
    return 2; // Per-record mappings are not physical consumption of pooled spans.
#else
    std::scoped_lock lock(allocations_mutex);
    uint64_t totals[6]{};
    for (const auto& [pointer, record] : allocations) {
        (void)pointer;
        const auto& f = record.footprint;
        if (!f.valid() || f.mapped != record.mapped_size) return 1;
        totals[0] += f.requested;
        totals[1] += f.minimum_mapped;
        totals[2] += f.mapped;
        totals[3] += f.granularity_padding();
        totals[4] += f.reuse_excess();
        ++totals[5];
    }
    // Publication is atomic under the ledger lock; failed queries leave output unchanged.
    for (size_t i = 0; i < 6; ++i) values[i] = totals[i];
    return 0;
#endif
}
