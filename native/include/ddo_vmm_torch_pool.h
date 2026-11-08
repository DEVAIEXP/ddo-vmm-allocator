#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include "ddo_vmm_completed_backing.h"
// Internal bridge, compiled only into the isolated pool candidate.
void* ddo_vmm_pool_malloc(size_t bytes, int device, cudaStream_t stream) noexcept;
void ddo_vmm_pool_free(void* pointer, cudaStream_t stream) noexcept;
int ddo_vmm_pool_stats(uint64_t* values, size_t count) noexcept;
int ddo_vmm_pool_trim_idle() noexcept;
// Internal arena-to-pool recovery. Never waits for a pool lock or CUDA event.
ddo_vmm_detail::ArenaPoolRelease ddo_vmm_pool_release_for_arena(
    int device, uintptr_t context, size_t required) noexcept;
