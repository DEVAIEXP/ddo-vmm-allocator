#include "ddo_vmm.h"

#include <cuda.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#endif

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
    size_t granularity = 0;
    CUmemGenericAllocationHandle allocation = 0;
    struct Mapping {
        size_t offset = 0;
        size_t size = 0;
        CUmemGenericAllocationHandle allocation = 0;
    };
    std::vector<Mapping> range_mappings;
    CUcontext retained_primary_context = nullptr;
    ddo_vmm_backing_kind backing_kind = DDO_VMM_BACKING_DEVICE;
};

struct ddo_vmm_model_arena {
    int device_ordinal = -1;
    CUdeviceptr address = 0;
    size_t virtual_size = 0;
    size_t page_size = 0;
    size_t resident_budget = 0;
    size_t resident_pages = 0;
    size_t fault_ceiling_page = 0;
    const char* admission_limit_reason = "resident capacity";
    CUmemGenericAllocationHandle view_bootstrap_allocation = 0;
    CUcontext retained_primary_context = nullptr;
    ddo_vmm_backing_kind backing_kind = DDO_VMM_BACKING_DEVICE;

    struct Page {
        CUmemGenericAllocationHandle allocation = 0;
        uint32_t pin_count = 0;
        uint64_t generation = 0;
        uint64_t last_touch = 0;
        bool view_bootstrap_mapped = false;
    };
    std::vector<Page> pages;
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

struct DxgiBudgetGuardConfig {
    uint64_t local_reserve_bytes = 0;
    uint64_t shared_reserve_bytes = 0;
    uint64_t system_reserve_bytes = 0;
};

std::mutex dxgi_budget_guard_mutex;
std::vector<std::pair<int, DxgiBudgetGuardConfig>> dxgi_budget_guards;

PressureEntry* pressure_entry(ddo_vmm_region* region) {
    for (PressureEntry& entry : pressure_entries) {
        if (entry.region == region) {
            return &entry;
        }
    }
    return nullptr;
}

}  // namespace

#if defined(_WIN32)
bool query_windows_memory(int device_ordinal, ddo_vmm_windows_memory_info* out_info) {
    if (out_info == nullptr) {
        return false;
    }
    *out_info = {};
    CUdevice device = 0;
    char cuda_luid[8]{};
    unsigned int node_mask = 0;
    if (cuDeviceGet(&device, device_ordinal) != CUDA_SUCCESS
        || cuDeviceGetLuid(cuda_luid, &node_mask, device) != CUDA_SUCCESS) {
        return false;
    }
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
        return false;
    }
    IDXGIAdapter1* adapter = nullptr;
    for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
        DXGI_ADAPTER_DESC1 description{};
        if (SUCCEEDED(adapter->GetDesc1(&description)) && memcmp(&description.AdapterLuid, cuda_luid, sizeof(cuda_luid)) == 0) {
            break;
        }
        adapter->Release();
        adapter = nullptr;
    }
    factory->Release();
    if (adapter == nullptr) {
        return false;
    }
    IDXGIAdapter3* adapter3 = nullptr;
    const HRESULT query_result = adapter->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3));
    adapter->Release();
    if (FAILED(query_result)) {
        return false;
    }
    DXGI_QUERY_VIDEO_MEMORY_INFO local{};
    DXGI_QUERY_VIDEO_MEMORY_INFO nonlocal{};
    const bool queried = SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local))
        && SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal));
    adapter3->Release();
    if (!queried) {
        return false;
    }
    MEMORYSTATUSEX system_info{};
    system_info.dwLength = sizeof(system_info);
    if (!GlobalMemoryStatusEx(&system_info)) {
        return false;
    }
    out_info->local_budget_bytes = local.Budget;
    out_info->local_usage_bytes = local.CurrentUsage;
    out_info->nonlocal_budget_bytes = nonlocal.Budget;
    out_info->nonlocal_usage_bytes = nonlocal.CurrentUsage;
    out_info->available_physical_bytes = system_info.ullAvailPhys;
    out_info->available_commit_bytes = system_info.ullAvailPageFile;
    return true;
}
#endif

bool dxgi_budget_guard_allows_materialize(
    int device_ordinal,
    size_t required_bytes,
    ddo_vmm_backing_kind backing_kind,
    bool allow_wddm_fallback = false
) {
    DxgiBudgetGuardConfig config{};
    {
        std::scoped_lock lock(dxgi_budget_guard_mutex);
        for (const auto& [ordinal, candidate] : dxgi_budget_guards) {
            if (ordinal == device_ordinal) {
                config = candidate;
                break;
            }
        }
    }
    if (
        config.local_reserve_bytes == 0 && config.shared_reserve_bytes == 0
        && config.system_reserve_bytes == 0
    ) {
        return true;
    }
#if defined(_WIN32)
    ddo_vmm_windows_memory_info info{};
    if (!query_windows_memory(device_ordinal, &info)) {
        return true;
    }
    const auto retained_budget = [](uint64_t budget, uint64_t reserve) {
        return budget > reserve ? budget - reserve : 0;
    };
    const uint64_t required = static_cast<uint64_t>(required_bytes);
    const uint64_t local_limit = retained_budget(info.local_budget_bytes, config.local_reserve_bytes);
    const uint64_t shared_limit = retained_budget(info.nonlocal_budget_bytes, config.shared_reserve_bytes);
    // DEVICE allocations are intentionally handed to WDDM as device
    // allocations.  The driver may place their physical pages in the
    // non-local/shared segment when local VRAM is exhausted, so the guard must
    // not reject them merely because the local budget is smaller than the
    // model.  HOST backing, when explicitly requested, is checked against the
    // shared budget directly.
    // The model-address arena explicitly delegates physical placement to
    // WDDM. Rejecting it at the local budget would prevent non-local backing.
    // Other VMM regions retain the conservative local-budget guard.
    const bool local_safe = allow_wddm_fallback || info.local_budget_bytes == 0
        || (info.local_usage_bytes <= local_limit && required <= local_limit - info.local_usage_bytes);
    const bool shared_safe = backing_kind != DDO_VMM_BACKING_HOST || info.nonlocal_budget_bytes == 0
        || (info.nonlocal_usage_bytes <= shared_limit && required <= shared_limit - info.nonlocal_usage_bytes);
    const bool system_safe = config.system_reserve_bytes == 0
        || (info.available_physical_bytes >= config.system_reserve_bytes + required
            && info.available_commit_bytes >= config.system_reserve_bytes + required);
    return local_safe && shared_safe && system_safe;
#else
    (void)device_ordinal;
    return true;
#endif
}

bool make_dxgi_budget_room(
    int device_ordinal,
    size_t required_bytes,
    ddo_vmm_backing_kind backing_kind,
    bool allow_wddm_fallback = false
) {
    while (!dxgi_budget_guard_allows_materialize(
        device_ordinal, required_bytes, backing_kind, allow_wddm_fallback
    )) {
        if (!ddo_vmm_pressure_evict_one(device_ordinal)) {
            return false;
        }
    }
    return true;
}

bool dxgi_budget_guard_configured(int device_ordinal) {
    std::scoped_lock lock(dxgi_budget_guard_mutex);
    for (const auto& [ordinal, config] : dxgi_budget_guards) {
        if (ordinal == device_ordinal) {
            return config.local_reserve_bytes != 0 || config.shared_reserve_bytes != 0
                || config.system_reserve_bytes != 0;
        }
    }
    return false;
}

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

bool activate_model_arena_context(const ddo_vmm_model_arena* arena, CUcontext* temporary_primary_context) {
    if (arena->retained_primary_context != nullptr) {
        return checked(cuCtxSetCurrent(arena->retained_primary_context), "cuCtxSetCurrent");
    }
    return ensure_context(arena->device_ordinal, temporary_primary_context);
}

size_t model_arena_page_begin(const ddo_vmm_model_arena* arena, size_t offset) {
    return offset / arena->page_size;
}

size_t model_arena_page_end(const ddo_vmm_model_arena* arena, size_t offset, size_t size) {
    return align_up(offset + size, arena->page_size) / arena->page_size;
}

uint64_t mix_signature(uint64_t value, uint64_t page, uint64_t generation) {
    value ^= (page + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2));
    value ^= (generation + 0xbf58476d1ce4e5b9ULL + (value << 7) + (value >> 3));
    return value == 0 ? 0xfeedfacecafebeefULL : value;
}

bool release_model_arena_page(ddo_vmm_model_arena* arena, size_t page_index) {
    if (arena == nullptr || page_index >= arena->pages.size()) {
        return false;
    }
    auto& page = arena->pages[page_index];
    if (page.allocation == 0) {
        return true;
    }
    const CUdeviceptr page_address = arena->address + page_index * arena->page_size;
    if (!checked(cuMemUnmap(page_address, arena->page_size), "cuMemUnmap")) {
        return false;
    }
    if (!checked(cuMemRelease(page.allocation), "cuMemRelease")) {
        return false;
    }
    page.allocation = 0;
    page.pin_count = 0;
    if (arena->resident_pages > 0) {
        --arena->resident_pages;
    }
    return true;
}

bool model_arena_evict_one(
    ddo_vmm_model_arena* arena,
    size_t protected_begin = std::numeric_limits<size_t>::max(),
    size_t protected_end = std::numeric_limits<size_t>::max()
) {
    size_t selected = arena->pages.size();
    for (size_t index = arena->pages.size(); index > 0; --index) {
        const size_t page_index = index - 1;
        if (protected_begin != std::numeric_limits<size_t>::max()
            && protected_begin <= page_index && page_index < protected_end) {
            continue;
        }
        const auto& page = arena->pages[page_index];
        if (page.allocation == 0 || page.pin_count != 0) {
            continue;
        }
        if (selected == arena->pages.size() || page.last_touch < arena->pages[selected].last_touch) {
            selected = page_index;
        }
    }
    if (selected == arena->pages.size()) {
        return false;
    }
    // Eviction only releases the physical allocation.  The virtual span
    // remains fully faultable: a low-address eviction must never make later
    // model weights permanently inaccessible.  Pressure is handled by the
    // LRU resident set, not by shrinking the model's address range.
    return release_model_arena_page(arena, selected);
}

void model_arena_limit_admission(
    ddo_vmm_model_arena* arena,
    size_t page_index,
    const char* reason
) {
    if (arena != nullptr) {
        arena->fault_ceiling_page = std::min(arena->fault_ceiling_page, page_index);
        arena->admission_limit_reason = reason;
    }
}

bool model_arena_can_admit_page(const ddo_vmm_model_arena* arena) {
    if (arena == nullptr || arena->resident_budget == 0) {
        return true;
    }
    const size_t budget_pages = std::max<size_t>(1, arena->resident_budget / arena->page_size);
    return arena->resident_pages < budget_pages;
}

size_t region_resident_size(const ddo_vmm_region* region) {
    if (region == nullptr) {
        return 0;
    }
    if (region->allocation != 0) {
        return region->mapped_size;
    }
    size_t total = 0;
    for (const auto& mapping : region->range_mappings) {
        total += mapping.size;
    }
    return total;
}

bool release_range_mappings(ddo_vmm_region* region, const char* operation) {
    for (auto& mapping : region->range_mappings) {
        if (mapping.allocation == 0) {
            continue;
        }
        if (!checked(cuMemUnmap(region->address + mapping.offset, mapping.size), operation)) {
            return false;
        }
        if (!checked(cuMemRelease(mapping.allocation), "cuMemRelease")) {
            return false;
        }
        mapping.allocation = 0;
    }
    region->range_mappings.clear();
    return true;
}

bool region_range_already_mapped(const ddo_vmm_region* region, size_t offset, size_t size) {
    if (region->allocation != 0) {
        return true;
    }
    const size_t end = offset + size;
    size_t cursor = offset;
    while (cursor < end) {
        bool advanced = false;
        for (const auto& mapping : region->range_mappings) {
            if (mapping.offset <= cursor && cursor < mapping.offset + mapping.size) {
                cursor = mapping.offset + mapping.size;
                advanced = true;
                break;
            }
        }
        if (!advanced) {
            return false;
        }
    }
    return true;
}

std::vector<std::pair<size_t, size_t>> missing_region_ranges(
    const ddo_vmm_region* region,
    size_t offset,
    size_t size
) {
    std::vector<std::pair<size_t, size_t>> missing;
    const size_t end = offset + size;
    size_t cursor = offset;
    while (cursor < end) {
        bool advanced = false;
        for (const auto& mapping : region->range_mappings) {
            if (mapping.offset <= cursor && cursor < mapping.offset + mapping.size) {
                cursor = mapping.offset + mapping.size;
                advanced = true;
                break;
            }
        }
        if (advanced) {
            continue;
        }
        size_t next = end;
        for (const auto& mapping : region->range_mappings) {
            if (mapping.offset > cursor && mapping.offset < next) {
                next = mapping.offset;
            }
        }
        missing.push_back({cursor, next - cursor});
        cursor = next;
    }
    return missing;
}

ddo_vmm_status materialize_region_range(
    ddo_vmm_region* region,
    size_t aligned_offset,
    size_t aligned_size
) {
    if (region == nullptr || aligned_size == 0 || aligned_offset > region->mapped_size
        || aligned_size > region->mapped_size - aligned_offset) {
        last_error = "VMM region range must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    if (region_range_already_mapped(region, aligned_offset, aligned_size)) {
        return DDO_VMM_SUCCESS;
    }
    if (region->allocation != 0) {
        return DDO_VMM_SUCCESS;
    }
    const auto missing_ranges = missing_region_ranges(region, aligned_offset, aligned_size);

    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(region->device_ordinal, region->backing_kind, &device, &properties)) {
        return DDO_VMM_CUDA_ERROR;
    }
    const bool guard_configured = dxgi_budget_guard_configured(region->device_ordinal);
    std::vector<ddo_vmm_region::Mapping> added_mappings;
    for (const auto& [range_offset, range_size] : missing_ranges) {
        if (!make_dxgi_budget_room(region->device_ordinal, range_size, region->backing_kind)) {
            set_error("dxgi_budget_guard", CUDA_ERROR_OUT_OF_MEMORY);
            return DDO_VMM_OUT_OF_MEMORY;
        }

        CUmemGenericAllocationHandle allocation = 0;
        while (cuMemCreate(&allocation, range_size, &properties, 0) != CUDA_SUCCESS) {
            allocation = 0;
            if (guard_configured) {
                set_error("cuMemCreate", CUDA_ERROR_OUT_OF_MEMORY);
                return DDO_VMM_OUT_OF_MEMORY;
            }
            if (!ddo_vmm_pressure_evict_one(region->device_ordinal)) {
                set_error("cuMemCreate", CUDA_ERROR_OUT_OF_MEMORY);
                return DDO_VMM_OUT_OF_MEMORY;
            }
        }
        if (allocation == 0) {
            return DDO_VMM_OUT_OF_MEMORY;
        }
        const CUdeviceptr target = region->address + range_offset;
        if (!checked(cuMemMap(target, range_size, 0, allocation, 0), "cuMemMap")) {
            cuMemRelease(allocation);
            return DDO_VMM_CUDA_ERROR;
        }
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = region->device_ordinal;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (!checked(cuMemSetAccess(target, range_size, &access, 1), "cuMemSetAccess")) {
            cuMemUnmap(target, range_size);
            cuMemRelease(allocation);
            return DDO_VMM_CUDA_ERROR;
        }
        added_mappings.push_back({range_offset, range_size, allocation});
    }
    region->range_mappings.insert(region->range_mappings.end(), added_mappings.begin(), added_mappings.end());
    return DDO_VMM_SUCCESS;
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
    region->granularity = granularity;

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
            if (region->allocation != 0 && region->address != 0 && region->mapped_size != 0) {
                cuMemUnmap(region->address, region->mapped_size);
            }
            if (region->allocation != 0) {
                cuMemRelease(region->allocation);
            }
            release_range_mappings(region, "cuMemUnmap");
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
    if (region->allocation == 0 && region->range_mappings.empty()) {
        return DDO_VMM_SUCCESS;
    }
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_region_context(region, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    if (region->allocation != 0) {
        if (!checked(cuMemUnmap(region->address, region->mapped_size), "cuMemUnmap")) {
            release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
            return DDO_VMM_CUDA_ERROR;
        }
        if (!checked(cuMemRelease(region->allocation), "cuMemRelease")) {
            release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
            return DDO_VMM_CUDA_ERROR;
        }
        region->allocation = 0;
    }
    if (!release_range_mappings(region, "cuMemUnmap")) {
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
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
    if (!region->range_mappings.empty() && !release_range_mappings(region, "cuMemUnmap")) {
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(region->device_ordinal, region->backing_kind, &device, &properties)) {
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    CUmemGenericAllocationHandle allocation = 0;
    const bool guard_configured = dxgi_budget_guard_configured(region->device_ordinal);
    if (!make_dxgi_budget_room(region->device_ordinal, region->mapped_size, region->backing_kind)) {
        set_error("dxgi_budget_guard", CUDA_ERROR_OUT_OF_MEMORY);
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_OUT_OF_MEMORY;
    }
    while (cuMemCreate(&allocation, region->mapped_size, &properties, 0) != CUDA_SUCCESS) {
        allocation = 0;
        if (guard_configured) {
            set_error("cuMemCreate", CUDA_ERROR_OUT_OF_MEMORY);
            release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
            return DDO_VMM_OUT_OF_MEMORY;
        }
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

extern "C" ddo_vmm_status ddo_vmm_region_materialize_range(
    ddo_vmm_region* region,
    size_t offset_bytes,
    size_t size_bytes
) {
    if (region == nullptr || size_bytes == 0) {
        last_error = "VMM region range must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    last_error.clear();
    if (offset_bytes > region->mapped_size || size_bytes > region->mapped_size - offset_bytes) {
        last_error = "VMM region range is outside the reserved address span";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t granularity = region->granularity == 0 ? 1 : region->granularity;
    const size_t aligned_offset = (offset_bytes / granularity) * granularity;
    const size_t range_end = align_up(offset_bytes + size_bytes, granularity);
    if (range_end == 0 || range_end > region->mapped_size || range_end < aligned_offset) {
        last_error = "VMM range overflows CUDA granularity alignment";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t aligned_size = range_end - aligned_offset;
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_region_context(region, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    ddo_vmm_status status = materialize_region_range(region, aligned_offset, aligned_size);
    release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
    return status;
}

extern "C" size_t ddo_vmm_region_resident_size(const ddo_vmm_region* region) {
    return region_resident_size(region);
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
    return region != nullptr && region_resident_size(region) == region->mapped_size ? 1 : 0;
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

extern "C" ddo_vmm_status ddo_vmm_model_arena_create(
    int device_ordinal,
    size_t virtual_size_bytes,
    size_t resident_budget_bytes,
    size_t page_size_bytes,
    ddo_vmm_backing_kind backing_kind,
    ddo_vmm_model_arena** out_arena
) {
    if (out_arena == nullptr || virtual_size_bytes == 0 || device_ordinal < 0) {
        last_error = "device ordinal, virtual size, and output pointer must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    *out_arena = nullptr;
    last_error.clear();
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }

    auto arena = std::make_unique<ddo_vmm_model_arena>();
    arena->device_ordinal = device_ordinal;
    arena->resident_budget = resident_budget_bytes;
    arena->backing_kind = backing_kind;
    if (!ensure_context(device_ordinal, &arena->retained_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }

    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(device_ordinal, backing_kind, &device, &properties)) {
        if (arena->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_CUDA_ERROR;
    }

    size_t granularity = 0;
    if (!checked(
            cuMemGetAllocationGranularity(
                &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM
            ),
            "cuMemGetAllocationGranularity"
        )) {
        if (arena->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_CUDA_ERROR;
    }
    const size_t requested_page_size = page_size_bytes == 0 ? granularity : page_size_bytes;
    const size_t page_size = align_up(requested_page_size, granularity);
    if (page_size == 0) {
        last_error = "model arena page size overflows CUDA VMM granularity alignment";
        if (arena->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t virtual_size = align_up(virtual_size_bytes, page_size);
    if (virtual_size == 0) {
        last_error = "model arena virtual size overflows page alignment";
        if (arena->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t page_count = virtual_size / page_size;
    if (page_count == 0 || page_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        last_error = "model arena page count is invalid";
        if (arena->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_INVALID_ARGUMENT;
    }

    if (!checked(cuMemAddressReserve(&arena->address, virtual_size, page_size, 0, 0), "cuMemAddressReserve")) {
        if (arena->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_CUDA_ERROR;
    }
    arena->virtual_size = virtual_size;
    arena->page_size = page_size;
    arena->fault_ceiling_page = page_count;
    arena->pages.resize(page_count);

    *out_arena = arena.release();
    return DDO_VMM_SUCCESS;
}

extern "C" void ddo_vmm_model_arena_destroy(ddo_vmm_model_arena* arena) {
    if (arena == nullptr) {
        return;
    }
    if (cuInit(0) == CUDA_SUCCESS) {
        CUcontext temporary_primary_context = nullptr;
        if (activate_model_arena_context(arena, &temporary_primary_context)) {
            for (size_t page_index = 0; page_index < arena->pages.size(); ++page_index) {
                auto& page = arena->pages[page_index];
                if (page.view_bootstrap_mapped) {
                    cuMemUnmap(arena->address + page_index * arena->page_size, arena->page_size);
                    page.view_bootstrap_mapped = false;
                }
                release_model_arena_page(arena, page_index);
            }
            if (arena->view_bootstrap_allocation != 0) {
                cuMemRelease(arena->view_bootstrap_allocation);
                arena->view_bootstrap_allocation = 0;
            }
            if (arena->address != 0 && arena->virtual_size != 0) {
                cuMemAddressFree(arena->address, arena->virtual_size);
            }
        }
        release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
        if (arena->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(arena->device_ordinal);
        }
    }
    delete arena;
}

extern "C" ddo_vmm_status ddo_vmm_model_arena_prepare_view_range(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes
) {
    if (arena == nullptr || size_bytes == 0 || offset_bytes > arena->virtual_size
        || size_bytes > arena->virtual_size - offset_bytes) {
        last_error = "model arena view range is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t begin_page = model_arena_page_begin(arena, offset_bytes);
    const size_t end_page = model_arena_page_end(arena, offset_bytes, size_bytes);
    if (begin_page >= end_page || end_page > arena->pages.size()) {
        last_error = "model arena view page range is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_model_arena_context(arena, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(arena->device_ordinal, arena->backing_kind, &device, &properties)) {
        release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    if (arena->view_bootstrap_allocation == 0
        && !checked(cuMemCreate(&arena->view_bootstrap_allocation, arena->page_size, &properties, 0), "cuMemCreate")) {
        release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    std::vector<size_t> mapped_pages;
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = arena->device_ordinal;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    for (size_t page_index = begin_page; page_index < end_page; ++page_index) {
        auto& page = arena->pages[page_index];
        if (page.allocation != 0 || page.view_bootstrap_mapped) {
            continue;
        }
        const CUdeviceptr page_address = arena->address + page_index * arena->page_size;
        if (cuMemMap(page_address, arena->page_size, 0, arena->view_bootstrap_allocation, 0) != CUDA_SUCCESS
            || cuMemSetAccess(page_address, arena->page_size, &access, 1) != CUDA_SUCCESS) {
            for (const size_t mapped_index : mapped_pages) {
                cuMemUnmap(arena->address + mapped_index * arena->page_size, arena->page_size);
                arena->pages[mapped_index].view_bootstrap_mapped = false;
            }
            last_error = "unable to map transient model arena view backing";
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            return DDO_VMM_CUDA_ERROR;
        }
        page.view_bootstrap_mapped = true;
        mapped_pages.push_back(page_index);
    }
    release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_model_arena_release_view_range(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes
) {
    if (arena == nullptr || size_bytes == 0 || offset_bytes > arena->virtual_size
        || size_bytes > arena->virtual_size - offset_bytes) {
        last_error = "model arena view range is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t begin_page = model_arena_page_begin(arena, offset_bytes);
    const size_t end_page = model_arena_page_end(arena, offset_bytes, size_bytes);
    if (begin_page >= end_page || end_page > arena->pages.size()) {
        last_error = "model arena view page range is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_model_arena_context(arena, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    for (size_t page_index = begin_page; page_index < end_page; ++page_index) {
        auto& page = arena->pages[page_index];
        if (!page.view_bootstrap_mapped) {
            continue;
        }
        if (!checked(cuMemUnmap(arena->address + page_index * arena->page_size, arena->page_size), "cuMemUnmap")) {
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            return DDO_VMM_CUDA_ERROR;
        }
        page.view_bootstrap_mapped = false;
    }
    release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_model_arena_fault_range(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes,
    uint64_t* out_signature
) {
    if (arena == nullptr || size_bytes == 0 || out_signature == nullptr) {
        last_error = "model arena range and signature output must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    last_error.clear();
    if (offset_bytes > arena->virtual_size || size_bytes > arena->virtual_size - offset_bytes) {
        last_error = "model arena range is outside the reserved address span";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t begin_page = model_arena_page_begin(arena, offset_bytes);
    const size_t end_page = model_arena_page_end(arena, offset_bytes, size_bytes);
    if (begin_page >= end_page || end_page > arena->pages.size()) {
        last_error = "model arena page range is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    if (begin_page >= arena->fault_ceiling_page) {
        last_error = std::string("model arena admission boundary reached by ")
            + arena->admission_limit_reason;
        return DDO_VMM_OUT_OF_MEMORY;
    }
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_model_arena_context(arena, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }

    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(arena->device_ordinal, arena->backing_kind, &device, &properties)) {
        release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    for (size_t page_index = begin_page; page_index < end_page; ++page_index) {
        auto& page = arena->pages[page_index];
        if (page.allocation != 0) {
            continue;
        }
        if (page.view_bootstrap_mapped) {
            if (!checked(cuMemUnmap(arena->address + page_index * arena->page_size, arena->page_size), "cuMemUnmap")) {
                release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
                return DDO_VMM_CUDA_ERROR;
            }
            page.view_bootstrap_mapped = false;
        }
        // A sequential model pass with a resident set smaller than the model
        // must retain the already admitted prefix. Replacing it page by page
        // would guarantee a full re-upload on the next pass. The caller uses
        // a short-lived ordinary CUDA tensor for ranges beyond this boundary.
        if (page_index >= arena->fault_ceiling_page || !model_arena_can_admit_page(arena)) {
            model_arena_limit_admission(arena, page_index, "resident capacity");
            last_error = "model arena admission boundary reached by resident capacity";
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            return DDO_VMM_OUT_OF_MEMORY;
        }
        if (!make_dxgi_budget_room(
            arena->device_ordinal, arena->page_size, arena->backing_kind, true
        )) {
            model_arena_limit_admission(arena, page_index, "memory guard");
            last_error = "model arena admission boundary reached by memory guard";
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            return DDO_VMM_OUT_OF_MEMORY;
        }
        CUmemGenericAllocationHandle allocation = 0;
        if (cuMemCreate(&allocation, arena->page_size, &properties, 0) != CUDA_SUCCESS) {
            model_arena_limit_admission(arena, page_index, "CUDA allocation pressure");
            last_error = "model arena admission boundary reached by CUDA allocation pressure";
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            return DDO_VMM_OUT_OF_MEMORY;
        }
        const CUdeviceptr page_address = arena->address + page_index * arena->page_size;
        if (!checked(cuMemMap(page_address, arena->page_size, 0, allocation, 0), "cuMemMap")) {
            cuMemRelease(allocation);
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            return DDO_VMM_CUDA_ERROR;
        }
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = arena->device_ordinal;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUresult access_result = CUDA_ERROR_NOT_READY;
        for (int attempt = 0; attempt < 4 && access_result == CUDA_ERROR_NOT_READY; ++attempt) {
            access_result = cuMemSetAccess(page_address, arena->page_size, &access, 1);
            if (access_result == CUDA_ERROR_NOT_READY) {
                cuCtxSynchronize();
            }
        }
        if (!checked(access_result, "cuMemSetAccess")) {
            cuMemUnmap(page_address, arena->page_size);
            cuMemRelease(allocation);
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            return DDO_VMM_CUDA_ERROR;
        }
        page.allocation = allocation;
        page.generation += 1;
        arena->resident_pages += 1;
    }

    uint64_t signature = 0x1234abcd5678ef00ULL;
    const uint64_t touch = ++pressure_clock;
    for (size_t page_index = begin_page; page_index < end_page; ++page_index) {
        auto& page = arena->pages[page_index];
        page.pin_count += 1;
        page.last_touch = touch;
        signature = mix_signature(signature, static_cast<uint64_t>(page_index), page.generation);
    }
    *out_signature = signature;
    release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_model_arena_copy_from_host(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes,
    const void* host_ptr
) {
    if (arena == nullptr || host_ptr == nullptr || size_bytes == 0 ||
        offset_bytes > arena->virtual_size || size_bytes > arena->virtual_size - offset_bytes) {
        return DDO_VMM_INVALID_ARGUMENT;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_model_arena_context(arena, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUresult result = cuMemcpyHtoD(arena->address + offset_bytes, host_ptr, size_bytes);
    release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
    if (result != CUDA_SUCCESS) {
        set_error("cuMemcpyHtoD", result);
        return DDO_VMM_CUDA_ERROR;
    }
    return DDO_VMM_SUCCESS;
}
extern "C" ddo_vmm_status ddo_vmm_model_arena_copy_from_host_async(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes,
    const void* host_ptr,
    void* cuda_stream
) {
    if (arena == nullptr || host_ptr == nullptr || size_bytes == 0 ||
        offset_bytes > arena->virtual_size || size_bytes > arena->virtual_size - offset_bytes) {
        return DDO_VMM_INVALID_ARGUMENT;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_model_arena_context(arena, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUresult result = cuMemcpyHtoDAsync(
        arena->address + offset_bytes,
        host_ptr,
        size_bytes,
        reinterpret_cast<CUstream>(cuda_stream)
    );
    release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
    if (result != CUDA_SUCCESS) {
        set_error("cuMemcpyHtoDAsync", result);
        return DDO_VMM_CUDA_ERROR;
    }
    return DDO_VMM_SUCCESS;
}
extern "C" ddo_vmm_status ddo_vmm_model_arena_unpin_range(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes
) {
    if (arena == nullptr || size_bytes == 0) {
        last_error = "model arena range must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    last_error.clear();
    if (offset_bytes > arena->virtual_size || size_bytes > arena->virtual_size - offset_bytes) {
        last_error = "model arena range is outside the reserved address span";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t begin_page = model_arena_page_begin(arena, offset_bytes);
    const size_t end_page = model_arena_page_end(arena, offset_bytes, size_bytes);
    if (begin_page >= end_page || end_page > arena->pages.size()) {
        last_error = "model arena page range is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    for (size_t page_index = begin_page; page_index < end_page; ++page_index) {
        auto& page = arena->pages[page_index];
        if (page.pin_count > 0) {
            page.pin_count -= 1;
        }
    }
    return DDO_VMM_SUCCESS;
}

extern "C" uintptr_t ddo_vmm_model_arena_address(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : static_cast<uintptr_t>(arena->address);
}

extern "C" size_t ddo_vmm_model_arena_virtual_size(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : arena->virtual_size;
}

extern "C" size_t ddo_vmm_model_arena_page_size(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : arena->page_size;
}

extern "C" size_t ddo_vmm_model_arena_resident_size(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : arena->resident_pages * arena->page_size;
}

extern "C" size_t ddo_vmm_model_arena_fault_ceiling(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : arena->fault_ceiling_page * arena->page_size;
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
            || region_resident_size(entry.region) == 0
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

extern "C" ddo_vmm_status ddo_vmm_windows_memory_query(
    int device_ordinal, ddo_vmm_windows_memory_info* out_info
) {
    if (device_ordinal < 0 || out_info == nullptr) {
        return DDO_VMM_INVALID_ARGUMENT;
    }
#if defined(_WIN32)
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    if (query_windows_memory(device_ordinal, out_info)) {
        return DDO_VMM_SUCCESS;
    }
    last_error = "unable to query DXGI memory budget for CUDA device";
    return DDO_VMM_CUDA_ERROR;
#else
    (void)out_info;
    last_error = "Windows DXGI memory budgets are unavailable on this platform";
    return DDO_VMM_INVALID_ARGUMENT;
#endif
}

extern "C" ddo_vmm_status ddo_vmm_dxgi_budget_guard_set(
    int device_ordinal,
    uint64_t local_reserve_bytes,
    uint64_t shared_reserve_bytes,
    uint64_t system_reserve_bytes
) {
    if (device_ordinal < 0) {
        return DDO_VMM_INVALID_ARGUMENT;
    }
    std::scoped_lock lock(dxgi_budget_guard_mutex);
    for (auto& [ordinal, config] : dxgi_budget_guards) {
        if (ordinal == device_ordinal) {
            config = {local_reserve_bytes, shared_reserve_bytes, system_reserve_bytes};
            return DDO_VMM_SUCCESS;
        }
    }
    dxgi_budget_guards.push_back({
        device_ordinal,
        {local_reserve_bytes, shared_reserve_bytes, system_reserve_bytes},
    });
    return DDO_VMM_SUCCESS;
}

extern "C" const char* ddo_vmm_last_error(void) {
    return last_error.c_str();
}
