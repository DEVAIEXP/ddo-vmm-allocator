// Optional PyTorch CUDAPluggableAllocator entry points.
//
// The API is intentionally small: PyTorch owns allocation scheduling and calls
// these functions; this DLL owns the CUDA VMM region lifetime.  It is not a
// replacement for a stream-aware caching allocator yet.

#include "ddo_vmm.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {

constexpr size_t kMinimumTorchAllocationBytes = 512;

struct AllocationRecord {
    std::unique_ptr<ddo_vmm_region, decltype(&ddo_vmm_region_destroy)> region{
        nullptr, &ddo_vmm_region_destroy
    };
    size_t mapped_size = 0;
};

std::mutex allocations_mutex;
std::unordered_map<void*, AllocationRecord> allocations;
size_t live_mapped_bytes = 0;

}  // namespace

extern "C" DDO_VMM_API void* ddo_vmm_torch_malloc(
    size_t size_bytes,
    int device_ordinal,
    cudaStream_t /*stream*/
) {
    // PyTorch may request a zero-byte tensor.  It still needs a unique,
    // non-null allocation record to preserve its allocator bookkeeping.
    const size_t requested_size = size_bytes == 0 ? kMinimumTorchAllocationBytes : size_bytes;
    ddo_vmm_region* raw_region = nullptr;
    if (ddo_vmm_region_create(device_ordinal, requested_size, &raw_region) != DDO_VMM_SUCCESS) {
        return nullptr;
    }

    const uintptr_t address = ddo_vmm_region_address(raw_region);
    const size_t mapped_size = ddo_vmm_region_mapped_size(raw_region);
    if (address == 0 || mapped_size == 0) {
        ddo_vmm_region_destroy(raw_region);
        return nullptr;
    }
    auto region = std::unique_ptr<ddo_vmm_region, decltype(&ddo_vmm_region_destroy)>(
        raw_region, &ddo_vmm_region_destroy
    );
    void* pointer = reinterpret_cast<void*>(address);

    std::scoped_lock lock(allocations_mutex);
    const auto [entry, inserted] = allocations.emplace(
        pointer, AllocationRecord{std::move(region), mapped_size}
    );
    if (!inserted) {
        return nullptr;
    }
    if (mapped_size > std::numeric_limits<size_t>::max() - live_mapped_bytes) {
        allocations.erase(entry);
        return nullptr;
    }
    live_mapped_bytes += mapped_size;
    return pointer;
}

extern "C" DDO_VMM_API void ddo_vmm_torch_free(
    void* pointer,
    size_t /*size_bytes*/,
    int /*device_ordinal*/,
    cudaStream_t /*stream*/
) {
    if (pointer == nullptr) {
        return;
    }
    AllocationRecord record;
    {
        std::scoped_lock lock(allocations_mutex);
        const auto found = allocations.find(pointer);
        if (found == allocations.end()) {
            return;
        }
        record = std::move(found->second);
        live_mapped_bytes -= record.mapped_size;
        allocations.erase(found);
    }
    // Release outside the lock because CUDA Driver calls may block.
    record.region.reset();
}

extern "C" DDO_VMM_API size_t ddo_vmm_torch_live_allocation_count(void) {
    std::scoped_lock lock(allocations_mutex);
    return allocations.size();
}

extern "C" DDO_VMM_API size_t ddo_vmm_torch_live_mapped_bytes(void) {
    std::scoped_lock lock(allocations_mutex);
    return live_mapped_bytes;
}

