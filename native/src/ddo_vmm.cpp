#include "ddo_vmm.h"

#include <cuda.h>

#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

thread_local std::string last_error;

void set_error(const char* operation, CUresult result) {
    const char* error_name = "unknown";
    const char* error_message = "unknown CUDA driver error";
    cuGetErrorName(result, &error_name);
    cuGetErrorString(result, &error_message);
    last_error = std::string(operation) + " failed: " + error_name + " (" + error_message + ")";
}

bool checked(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) {
        return true;
    }
    set_error(operation, result);
    return false;
}

size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0 || value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return 0;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

bool device_properties(int ordinal, ddo_vmm_backing_kind backing_kind, CUdevice* device, CUmemAllocationProp* properties) {
    if (!checked(cuDeviceGet(device, ordinal), "cuDeviceGet")) {
        return false;
    }
    *properties = {};
    properties->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    if (backing_kind == DDO_VMM_BACKING_HOST) {
        properties->location.type = CU_MEM_LOCATION_TYPE_HOST;
        properties->location.id = 0;
    } else {
        properties->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        properties->location.id = ordinal;
    }
    properties->requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    return true;
}

bool ensure_context(int device_ordinal, CUcontext* retained_primary_context) {
    CUcontext current = nullptr;
    if (!checked(cuCtxGetCurrent(&current), "cuCtxGetCurrent")) {
        return false;
    }
    if (current != nullptr) {
        return true;
    }
    if (!checked(
            cuDevicePrimaryCtxRetain(retained_primary_context, device_ordinal),
            "cuDevicePrimaryCtxRetain"
        )) {
        return false;
    }
    if (!checked(cuCtxSetCurrent(*retained_primary_context), "cuCtxSetCurrent")) {
        cuDevicePrimaryCtxRelease(device_ordinal);
        *retained_primary_context = nullptr;
        return false;
    }
    return true;
}

}  // namespace

struct ddo_vmm_region {
    int device_ordinal = -1;
    CUdeviceptr address = 0;
    size_t requested_size = 0;
    size_t mapped_size = 0;
    CUmemGenericAllocationHandle allocation = 0;
    CUcontext retained_primary_context = nullptr;
    ddo_vmm_backing_kind backing_kind = DDO_VMM_BACKING_DEVICE;
};

namespace {

struct PressureEntry {
    ddo_vmm_region* region = nullptr;
    bool pinned = false;
    uint64_t touch = 0;
};

std::mutex pressure_mutex;
std::vector<PressureEntry> pressure_entries;
uint64_t pressure_clock = 0;

PressureEntry* pressure_entry(ddo_vmm_region* region) {
    for (PressureEntry& entry : pressure_entries) {
        if (entry.region == region) {
            return &entry;
        }
    }
    return nullptr;
}

}  // namespace

namespace {

bool activate_region_context(const ddo_vmm_region* region, CUcontext* temporary_primary_context) {
    if (region->retained_primary_context != nullptr) {
        return checked(cuCtxSetCurrent(region->retained_primary_context), "cuCtxSetCurrent");
    }
    return ensure_context(region->device_ordinal, temporary_primary_context);
}

void release_temporary_primary_context(int device_ordinal, CUcontext temporary_primary_context) {
    if (temporary_primary_context != nullptr) {
        cuCtxSetCurrent(nullptr);
        cuDevicePrimaryCtxRelease(device_ordinal);
    }
}

}  // namespace

extern "C" ddo_vmm_status ddo_vmm_region_create(
    int device_ordinal,
    size_t size_bytes,
    ddo_vmm_region** out_region
) {
    return ddo_vmm_region_create_ex(device_ordinal, size_bytes, DDO_VMM_BACKING_DEVICE, out_region);
}

extern "C" ddo_vmm_status ddo_vmm_region_create_ex(
    int device_ordinal,
    size_t size_bytes,
    ddo_vmm_backing_kind backing_kind,
    ddo_vmm_region** out_region
) {
    if (out_region == nullptr || size_bytes == 0 || device_ordinal < 0) {
        last_error = "device ordinal, size, and output pointer must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    *out_region = nullptr;
    last_error.clear();

    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    auto region = std::make_unique<ddo_vmm_region>();
    region->device_ordinal = device_ordinal;
    region->backing_kind = backing_kind;
    region->requested_size = size_bytes;
    if (!ensure_context(device_ordinal, &region->retained_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }

    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(device_ordinal, backing_kind, &device, &properties)) {
        return DDO_VMM_CUDA_ERROR;
    }

    size_t granularity = 0;
    if (!checked(
            cuMemGetAllocationGranularity(
                &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM
            ),
            "cuMemGetAllocationGranularity"
        )) {
        return DDO_VMM_CUDA_ERROR;
    }
    const size_t mapped_size = align_up(size_bytes, granularity);
    if (mapped_size == 0) {
        last_error = "requested size overflows CUDA VMM granularity alignment";
        return DDO_VMM_INVALID_ARGUMENT;
    }

    region->mapped_size = mapped_size;

    if (!checked(
            cuMemAddressReserve(&region->address, mapped_size, granularity, 0, 0),
            "cuMemAddressReserve"
        )) {
        if (region->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_CUDA_ERROR;
    }
    if (!checked(cuMemCreate(&region->allocation, mapped_size, &properties, 0), "cuMemCreate")) {
        cuMemAddressFree(region->address, region->mapped_size);
        if (region->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_OUT_OF_MEMORY;
    }
    if (!checked(cuMemMap(region->address, mapped_size, 0, region->allocation, 0), "cuMemMap")) {
        cuMemRelease(region->allocation);
        cuMemAddressFree(region->address, region->mapped_size);
        if (region->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_CUDA_ERROR;
    }

    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_ordinal;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!checked(cuMemSetAccess(region->address, mapped_size, &access, 1), "cuMemSetAccess")) {
        cuMemUnmap(region->address, region->mapped_size);
        cuMemRelease(region->allocation);
        cuMemAddressFree(region->address, region->mapped_size);
        if (region->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_CUDA_ERROR;
    }

    *out_region = region.release();
    return DDO_VMM_SUCCESS;
}

extern "C" void ddo_vmm_region_destroy(ddo_vmm_region* region) {
    if (region == nullptr) {
        return;
    }
    ddo_vmm_pressure_unregister(region);
    if (cuInit(0) == CUDA_SUCCESS) {
        CUcontext temporary_primary_context = nullptr;
        if (activate_region_context(region, &temporary_primary_context)) {
            if (region->address != 0 && region->mapped_size != 0) {
                cuMemUnmap(region->address, region->mapped_size);
            }
            if (region->allocation != 0) {
                cuMemRelease(region->allocation);
            }
            if (region->address != 0 && region->mapped_size != 0) {
                cuMemAddressFree(region->address, region->mapped_size);
            }
        }
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        if (region->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(region->device_ordinal);
        }
    }
    delete region;
}

extern "C" ddo_vmm_status ddo_vmm_region_evict(ddo_vmm_region* region) {
    if (region == nullptr) {
        last_error = "VMM region must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    last_error.clear();
    if (region->allocation == 0) {
        return DDO_VMM_SUCCESS;
    }
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_region_context(region, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    if (!checked(cuMemUnmap(region->address, region->mapped_size), "cuMemUnmap")) {
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    if (!checked(cuMemRelease(region->allocation), "cuMemRelease")) {
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    region->allocation = 0;
    release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_region_materialize(ddo_vmm_region* region) {
    if (region == nullptr) {
        last_error = "VMM region must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    last_error.clear();
    if (region->allocation != 0) {
        return DDO_VMM_SUCCESS;
    }
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_region_context(region, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(region->device_ordinal, region->backing_kind, &device, &properties)) {
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    CUmemGenericAllocationHandle allocation = 0;
    while (cuMemCreate(&allocation, region->mapped_size, &properties, 0) != CUDA_SUCCESS) {
        allocation = 0;
        if (!ddo_vmm_pressure_evict_one(region->device_ordinal)) {
            set_error("cuMemCreate", CUDA_ERROR_OUT_OF_MEMORY);
            release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
            return DDO_VMM_OUT_OF_MEMORY;
        }
    }
    if (allocation == 0) {
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_OUT_OF_MEMORY;
    }
    if (!checked(cuMemMap(region->address, region->mapped_size, 0, allocation, 0), "cuMemMap")) {
        cuMemRelease(allocation);
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = region->device_ordinal;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!checked(cuMemSetAccess(region->address, region->mapped_size, &access, 1), "cuMemSetAccess")) {
        cuMemUnmap(region->address, region->mapped_size);
        cuMemRelease(allocation);
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    region->allocation = allocation;
    release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
    return DDO_VMM_SUCCESS;
}

extern "C" uintptr_t ddo_vmm_region_address(const ddo_vmm_region* region) {
    return region == nullptr ? 0 : static_cast<uintptr_t>(region->address);
}

extern "C" size_t ddo_vmm_region_requested_size(const ddo_vmm_region* region) {
    return region == nullptr ? 0 : region->requested_size;
}

extern "C" size_t ddo_vmm_region_mapped_size(const ddo_vmm_region* region) {
    return region == nullptr ? 0 : region->mapped_size;
}

extern "C" int ddo_vmm_region_is_materialized(const ddo_vmm_region* region) {
    return region != nullptr && region->allocation != 0 ? 1 : 0;
}

extern "C" size_t ddo_vmm_device_granularity(int device_ordinal) {
    last_error.clear();
    if (device_ordinal < 0 || !checked(cuInit(0), "cuInit")) {
        return 0;
    }
    CUcontext retained_primary_context = nullptr;
    if (!ensure_context(device_ordinal, &retained_primary_context)) {
        return 0;
    }
    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(device_ordinal, DDO_VMM_BACKING_DEVICE, &device, &properties)) {
        return 0;
    }
    size_t granularity = 0;
    const size_t result = checked(
               cuMemGetAllocationGranularity(
                   &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM
               ),
               "cuMemGetAllocationGranularity"
           )
        ? granularity
        : 0;
    if (retained_primary_context != nullptr) {
        cuCtxSetCurrent(nullptr);
        cuDevicePrimaryCtxRelease(device_ordinal);
    }
    return result;
}

extern "C" void ddo_vmm_pressure_register(ddo_vmm_region* region) {
    if (region == nullptr) {
        return;
    }
    std::scoped_lock lock(pressure_mutex);
    if (pressure_entry(region) == nullptr) {
        pressure_entries.push_back({region, false, ++pressure_clock});
    }
}

extern "C" void ddo_vmm_pressure_unregister(ddo_vmm_region* region) {
    std::scoped_lock lock(pressure_mutex);
    for (auto entry = pressure_entries.begin(); entry != pressure_entries.end(); ++entry) {
        if (entry->region == region) {
            pressure_entries.erase(entry);
            return;
        }
    }
}

extern "C" void ddo_vmm_pressure_set_pinned(ddo_vmm_region* region, int pinned) {
    std::scoped_lock lock(pressure_mutex);
    if (PressureEntry* entry = pressure_entry(region)) {
        entry->pinned = pinned != 0;
        entry->touch = ++pressure_clock;
    }
}

extern "C" void ddo_vmm_pressure_touch(ddo_vmm_region* region) {
    std::scoped_lock lock(pressure_mutex);
    if (PressureEntry* entry = pressure_entry(region)) {
        entry->touch = ++pressure_clock;
    }
}

extern "C" int ddo_vmm_pressure_evict_one(int device_ordinal) {
    std::scoped_lock lock(pressure_mutex);
    PressureEntry* selected = nullptr;
    for (PressureEntry& entry : pressure_entries) {
        if (
            entry.region == nullptr || entry.pinned || entry.region->device_ordinal != device_ordinal
            || entry.region->allocation == 0
        ) {
            continue;
        }
        if (selected == nullptr || entry.touch < selected->touch) {
            selected = &entry;
        }
    }
    if (selected == nullptr) {
        return 0;
    }
    if (ddo_vmm_region_evict(selected->region) != DDO_VMM_SUCCESS) {
        return 0;
    }
    selected->touch = ++pressure_clock;
    return 1;
}

extern "C" const char* ddo_vmm_last_error(void) {
    return last_error.c_str();
}
