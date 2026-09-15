#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(DDO_VMM_BUILDING_LIBRARY)
#define DDO_VMM_API __declspec(dllexport)
#else
#define DDO_VMM_API __declspec(dllimport)
#endif
#else
#define DDO_VMM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ddo_vmm_region ddo_vmm_region;
typedef struct ddo_vmm_model_arena ddo_vmm_model_arena;

typedef enum ddo_vmm_status {
    DDO_VMM_SUCCESS = 0,
    DDO_VMM_INVALID_ARGUMENT = 1,
    DDO_VMM_OUT_OF_MEMORY = 2,
    DDO_VMM_CUDA_ERROR = 3,
} ddo_vmm_status;

typedef enum ddo_vmm_backing_kind {
    DDO_VMM_BACKING_DEVICE = 0,
    DDO_VMM_BACKING_HOST = 1,
} ddo_vmm_backing_kind;

typedef struct ddo_vmm_windows_memory_info {
    uint64_t local_budget_bytes;
    uint64_t local_usage_bytes;
    uint64_t nonlocal_budget_bytes;
    uint64_t nonlocal_usage_bytes;
    uint64_t available_physical_bytes;
    uint64_t available_commit_bytes;
} ddo_vmm_windows_memory_info;

/**
 * Reserve a stable CUDA virtual address, create device-local physical backing,
 * map it, and grant read/write access to `device_ordinal`.
 *
 * The returned virtual address is valid until ddo_vmm_region_destroy().
 * `size_bytes` is rounded up to CUDA's minimum VMM allocation granularity.
 */
DDO_VMM_API ddo_vmm_status ddo_vmm_region_create(
    int device_ordinal,
    size_t size_bytes,
    ddo_vmm_region** out_region
);

DDO_VMM_API ddo_vmm_status ddo_vmm_region_create_ex(
    int device_ordinal,
    size_t size_bytes,
    ddo_vmm_backing_kind backing_kind,
    ddo_vmm_region** out_region
);

/** Unmap, release physical backing, and return the virtual address reservation. */
DDO_VMM_API void ddo_vmm_region_destroy(ddo_vmm_region* region);

/** Release physical backing while preserving the region's reserved virtual address. */
DDO_VMM_API ddo_vmm_status ddo_vmm_region_evict(ddo_vmm_region* region);

/** Recreate device-local backing and map it at the region's original virtual address. */
DDO_VMM_API ddo_vmm_status ddo_vmm_region_materialize(ddo_vmm_region* region);

/**
 * Recreate backing for a byte range inside the reserved virtual address.
 *
 * `offset_bytes` and `size_bytes` are rounded outward to CUDA's VMM
 * allocation granularity.  The returned resident-byte accounting therefore
 * may be larger than the requested payload range.
 */
DDO_VMM_API ddo_vmm_status ddo_vmm_region_materialize_range(
    ddo_vmm_region* region,
    size_t offset_bytes,
    size_t size_bytes
);

/** Return resident physical bytes currently mapped into this virtual region. */
DDO_VMM_API size_t ddo_vmm_region_resident_size(const ddo_vmm_region* region);

DDO_VMM_API uintptr_t ddo_vmm_region_address(const ddo_vmm_region* region);
DDO_VMM_API size_t ddo_vmm_region_requested_size(const ddo_vmm_region* region);
DDO_VMM_API size_t ddo_vmm_region_mapped_size(const ddo_vmm_region* region);
DDO_VMM_API int ddo_vmm_region_is_materialized(const ddo_vmm_region* region);
DDO_VMM_API size_t ddo_vmm_device_granularity(int device_ordinal);

/**
 * Reserve one stable virtual address span for a complete model-weight address
 * space. Physical pages are not committed until ddo_vmm_model_arena_fault_range().
 *
 * resident_budget_bytes == 0 lets the CUDA driver/WDDM decide residency until
 * allocation pressure is reported by cuMemCreate. page_size_bytes == 0 uses
 * the CUDA minimum VMM allocation granularity for the selected backing kind.
 */
DDO_VMM_API ddo_vmm_status ddo_vmm_model_arena_create(
    int device_ordinal,
    size_t virtual_size_bytes,
    size_t resident_budget_bytes,
    size_t page_size_bytes,
    ddo_vmm_backing_kind backing_kind,
    ddo_vmm_model_arena** out_arena
);

DDO_VMM_API void ddo_vmm_model_arena_destroy(ddo_vmm_model_arena* arena);

/**
 * Materialize and pin the pages covering a byte range. out_signature receives
 * a stable content-generation signature for the current resident pages. If the
 * same range faults again without any page being recycled, the signature stays
 * unchanged and callers can skip copying immutable source bytes.
 */
DDO_VMM_API ddo_vmm_status ddo_vmm_model_arena_fault_range(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes,
    uint64_t* out_signature
);
DDO_VMM_API ddo_vmm_status ddo_vmm_model_arena_copy_from_host(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes,
    const void* host_ptr
);
DDO_VMM_API ddo_vmm_status ddo_vmm_model_arena_copy_from_host_async(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes,
    const void* host_ptr,
    void* cuda_stream
);
/** Release one pin acquired by fault_range for each page covering the range. */
DDO_VMM_API ddo_vmm_status ddo_vmm_model_arena_unpin_range(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes
);

DDO_VMM_API uintptr_t ddo_vmm_model_arena_address(const ddo_vmm_model_arena* arena);
DDO_VMM_API size_t ddo_vmm_model_arena_virtual_size(const ddo_vmm_model_arena* arena);
DDO_VMM_API size_t ddo_vmm_model_arena_page_size(const ddo_vmm_model_arena* arena);
DDO_VMM_API size_t ddo_vmm_model_arena_resident_size(const ddo_vmm_model_arena* arena);
DDO_VMM_API size_t ddo_vmm_model_arena_fault_ceiling(const ddo_vmm_model_arena* arena);

/** Register model-weight regions that may be evicted under allocator pressure. */
DDO_VMM_API void ddo_vmm_pressure_register(ddo_vmm_region* region);
DDO_VMM_API void ddo_vmm_pressure_unregister(ddo_vmm_region* region);
DDO_VMM_API void ddo_vmm_pressure_set_pinned(ddo_vmm_region* region, int pinned);
DDO_VMM_API void ddo_vmm_pressure_touch(ddo_vmm_region* region);
/** Evict one least-recently-used unpinned region for a device; returns 1 on success. */
DDO_VMM_API int ddo_vmm_pressure_evict_one(int device_ordinal);
DDO_VMM_API ddo_vmm_status ddo_vmm_windows_memory_query(
    int device_ordinal, ddo_vmm_windows_memory_info* out_info
);
DDO_VMM_API ddo_vmm_status ddo_vmm_dxgi_budget_guard_set(
    int device_ordinal,
    uint64_t local_headroom_bytes,
    uint64_t nonlocal_headroom_bytes,
    uint64_t system_headroom_bytes
);
DDO_VMM_API const char* ddo_vmm_last_error(void);

#ifdef __cplusplus
}
#endif
