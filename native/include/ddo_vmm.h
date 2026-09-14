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

typedef enum ddo_vmm_status {
    DDO_VMM_SUCCESS = 0,
    DDO_VMM_INVALID_ARGUMENT = 1,
    DDO_VMM_OUT_OF_MEMORY = 2,
    DDO_VMM_CUDA_ERROR = 3,
} ddo_vmm_status;

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

/** Unmap, release physical backing, and return the virtual address reservation. */
DDO_VMM_API void ddo_vmm_region_destroy(ddo_vmm_region* region);

DDO_VMM_API uintptr_t ddo_vmm_region_address(const ddo_vmm_region* region);
DDO_VMM_API size_t ddo_vmm_region_requested_size(const ddo_vmm_region* region);
DDO_VMM_API size_t ddo_vmm_region_mapped_size(const ddo_vmm_region* region);
DDO_VMM_API size_t ddo_vmm_device_granularity(int device_ordinal);
DDO_VMM_API const char* ddo_vmm_last_error(void);

#ifdef __cplusplus
}
#endif

