// Optional PyTorch CUDAPluggableAllocator entry points.
//
// The API is intentionally small: PyTorch owns allocation scheduling and calls
// these functions; this DLL owns the CUDA VMM region lifetime.  It is not a
// replacement for a stream-aware caching allocator yet.

#include "ddo_vmm.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

struct AllocationRecord {
    std::unique_ptr<ddo_vmm_region, decltype(&ddo_vmm_region_destroy)> region{
        nullptr, &ddo_vmm_region_destroy
    };
    size_t mapped_size = 0;
    int device_ordinal = -1;
    CUevent completion_event = nullptr;

    AllocationRecord() = default;
    AllocationRecord(const AllocationRecord&) = delete;
    AllocationRecord& operator=(const AllocationRecord&) = delete;

    AllocationRecord(AllocationRecord&& other) noexcept
        : region(std::move(other.region)),
          mapped_size(other.mapped_size),
          device_ordinal(other.device_ordinal),
          completion_event(std::exchange(other.completion_event, nullptr)) {
        other.mapped_size = 0;
        other.device_ordinal = -1;
    }

    AllocationRecord& operator=(AllocationRecord&& other) noexcept {
        if (this != &other) {
            reset();
            region = std::move(other.region);
            mapped_size = other.mapped_size;
            device_ordinal = other.device_ordinal;
            completion_event = std::exchange(other.completion_event, nullptr);
            other.mapped_size = 0;
            other.device_ordinal = -1;
        }
        return *this;
    }

    ~AllocationRecord() { reset(); }

    void reset() noexcept {
        if (completion_event != nullptr) {
            // A pointer may still be referenced by work on its freeing stream.
            // Never unmap its VMM backing before that work has completed.
            cuEventSynchronize(completion_event);
            cuEventDestroy(completion_event);
            completion_event = nullptr;
        }
        region.reset();
        mapped_size = 0;
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

void record_stream_completion(AllocationRecord* record, cudaStream_t stream) {
    if (record == nullptr || record->completion_event != nullptr) {
        return;
    }
    CUevent event = nullptr;
    if (
        cuEventCreate(&event, CU_EVENT_DISABLE_TIMING) == CUDA_SUCCESS
        && cuEventRecord(event, reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS
    ) {
        record->completion_event = event;
        return;
    }
    if (event != nullptr) {
        cuEventDestroy(event);
    }
    // If event setup failed, synchronize the freeing stream rather than
    // risk releasing a virtual mapping that a queued kernel can still read.
    cuStreamSynchronize(reinterpret_cast<CUstream>(stream));
}

void wait_for_stream_completion(AllocationRecord* record) {
    if (record == nullptr || record->completion_event == nullptr) {
        return;
    }
    cuEventSynchronize(record->completion_event);
    cuEventDestroy(record->completion_event);
    record->completion_event = nullptr;
}

bool take_cached_allocation(
    size_t minimum_mapped_size,
    int device_ordinal,
    AllocationRecord* out_record
) {
    std::scoped_lock lock(allocations_mutex);
    for (auto candidate = cached_allocations.lower_bound(minimum_mapped_size);
         candidate != cached_allocations.end(); ++candidate) {
        if (candidate->second.device_ordinal != device_ordinal) {
            continue;
        }
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
    cudaStream_t /*stream*/
) {
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
        ddo_vmm_region* raw_region = nullptr;
        if (ddo_vmm_region_create(device_ordinal, requested_size, &raw_region) != DDO_VMM_SUCCESS) {
            // A full cache must never make a new allocation fail.  Release
            // cached physical backing and retry once before reporting OOM.
            auto evicted = drain_cached_allocations();
            release_records(evicted);
            if (ddo_vmm_region_create(device_ordinal, requested_size, &raw_region) != DDO_VMM_SUCCESS) {
                return nullptr;
            }
        }
        record.region = std::unique_ptr<ddo_vmm_region, decltype(&ddo_vmm_region_destroy)>(
            raw_region, &ddo_vmm_region_destroy
        );
        record.mapped_size = ddo_vmm_region_mapped_size(raw_region);
        record.device_ordinal = device_ordinal;
    }
    wait_for_stream_completion(&record);
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
}

extern "C" DDO_VMM_API void ddo_vmm_torch_free(
    void* pointer,
    size_t /*size_bytes*/,
    int /*device_ordinal*/,
    cudaStream_t stream
) {
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

        record_stream_completion(&record, stream);

        const size_t cache_limit = cache_limit_bytes();
        if (cache_limit != 0 && record.mapped_size <= cache_limit) {
            while (
                !cached_allocations.empty()
                && (cached_mapped_bytes > cache_limit || record.mapped_size > cache_limit - cached_mapped_bytes)
            ) {
                auto candidate = cached_allocations.begin();
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
}

extern "C" DDO_VMM_API size_t ddo_vmm_torch_live_allocation_count(void) {
    std::scoped_lock lock(allocations_mutex);
    return allocations.size();
}

extern "C" DDO_VMM_API size_t ddo_vmm_torch_live_mapped_bytes(void) {
    std::scoped_lock lock(allocations_mutex);
    return live_mapped_bytes;
}

extern "C" DDO_VMM_API size_t ddo_vmm_torch_cached_allocation_count(void) {
    std::scoped_lock lock(allocations_mutex);
    return cached_allocations.size();
}

extern "C" DDO_VMM_API size_t ddo_vmm_torch_cached_mapped_bytes(void) {
    std::scoped_lock lock(allocations_mutex);
    return cached_mapped_bytes;
}
