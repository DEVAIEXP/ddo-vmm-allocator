#pragma once
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include "ddo_vmm_access_recovery.h"
#ifdef DDO_VMM_TIMING_COUNTERS
#include "ddo_vmm_timing_counters.h"
#define DDO_SEGMENT_TIMED(op, bytes, expr) \
    ddo_vmm_detail::timed_call(ddo_vmm_detail::TimingOp::op, [&]() noexcept { return (expr); }, (bytes))
#else
#define DDO_SEGMENT_TIMED(op, bytes, expr) (expr)
#endif

namespace ddo_dead_pages {
// Compilation-only adapter. The owner must already be in its captured context.
// No initialization or unconditional synchronization occurs here. NOT_READY
// permits one barrier and one continuation. OOM is returned verbatim and is
// handled by the pool after rollback; this adapter never recovers capacity.
class CudaSegmentDriver {
public:
    struct AccessSequence {
        bool first_observed = false, barrier_observed = false, continuation_observed = false;
        int first = 0, barrier = 0, continuation = 0;
    };
private:
    CUcontext context_;
    int device_;
    AccessSequence last_access_;
    int materialization_result(const char* operation, CUresult result,
            uintptr_t address, size_t bytes, CUmemGenericAllocationHandle handle) const noexcept {
        if (result != CUDA_SUCCESS) {
            // Preserve the original call result before rollback or observation.
            // This path performs no CUDA query, allocation, synchronization or retry.
            std::fprintf(stderr, "DDO segmented Driver refusal: operation=%s status=%d device=%d address=%llu bytes=%llu handle=%llu\n",
                operation, static_cast<int>(result), device_,
                static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes),
                static_cast<unsigned long long>(handle));
        }
        return static_cast<int>(result);
    }
    CUmemAllocationProp properties() const noexcept {
        CUmemAllocationProp p{};
        p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        p.location.id = device_;
        return p;
    }
    int identity_error() const noexcept {
        CUcontext current = nullptr;
        const auto status = cuCtxGetCurrent(&current);
        if (status != CUDA_SUCCESS) return static_cast<int>(status);
        if (!context_ || current != context_) return CUDA_ERROR_INVALID_CONTEXT;
        CUdevice device = -1;
        const auto result = cuCtxGetDevice(&device);
        if (result != CUDA_SUCCESS) return static_cast<int>(result);
        return device == device_ ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE;
    }
public:
    using Handle = CUmemGenericAllocationHandle;
    CudaSegmentDriver(CUcontext context, int device) noexcept : context_(context), device_(device) {}
    bool capacity_refusal(int status) const noexcept { return status == CUDA_ERROR_OUT_OF_MEMORY; }
    const AccessSequence& last_access() const noexcept { return last_access_; }
    bool identity(int device) const noexcept { return device == device_ && !identity_error(); }
    int granularity(size_t& result) noexcept {
        if (const int error = identity_error()) return error;
        auto p = properties();
        return static_cast<int>(cuMemGetAllocationGranularity(&result, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    }
    int reserve(size_t bytes, size_t alignment, uintptr_t& output) noexcept {
        output = 0;
        if (const int error = identity_error()) return error;
        CUdeviceptr address = 0;
        const auto status = cuMemAddressReserve(&address, bytes, alignment, 0, 0);
        output = static_cast<uintptr_t>(address);
        return static_cast<int>(status);
    }
    int free_va(uintptr_t address, size_t bytes) noexcept {
        if (const int error = identity_error()) return error;
        return static_cast<int>(cuMemAddressFree(address, bytes));
    }
    int create(size_t bytes, Handle& handle) noexcept {
        handle = 0;
        if (const int error = identity_error()) return error;
        auto p = properties();
        const auto result = DDO_SEGMENT_TIMED(segment_create, bytes, cuMemCreate(&handle, bytes, &p, 0));
        return materialization_result("cuMemCreate", result, 0, bytes, handle);
    }
    int map(uintptr_t address, size_t bytes, Handle handle) noexcept {
        if (const int error = identity_error()) return error;
        const auto result = DDO_SEGMENT_TIMED(segment_map, bytes, cuMemMap(address, bytes, 0, handle, 0));
        return materialization_result("cuMemMap", result, address, bytes, handle);
    }
    int access(uintptr_t address, size_t bytes) noexcept {
        if (const int error = identity_error()) return error;
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device_;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        const auto result = DDO_SEGMENT_TIMED(segment_access, bytes, cuMemSetAccess(address, bytes, &access, 1));
        last_access_ = {};
        last_access_.first_observed = true; last_access_.first = static_cast<int>(result);
        materialization_result("cuMemSetAccess", result, address, bytes, 0);
        if (result == CUDA_ERROR_NOT_READY) {
            // Reuse the region-creation helper, preserving the already observed
            // first result. Never chain the OOM path after this continuation.
            bool first = true;
            const auto continued = ddo_vmm_detail::recover_access_once(
                CUDA_SUCCESS, CUDA_ERROR_NOT_READY,
                [&]() noexcept -> CUresult {
                    if (first) { first = false; return result; }
                    if (const int error = identity_error()) {
                        std::fprintf(stderr, "DDO segmented NOT_READY continuation refused: identity_status=%d\n", error);
                        return static_cast<CUresult>(error);
                    }
                    const auto status = DDO_SEGMENT_TIMED(segment_access, bytes, cuMemSetAccess(address, bytes, &access, 1));
                    last_access_.continuation_observed = true;
                    last_access_.continuation = static_cast<int>(status);
                    std::fprintf(stderr, "DDO segmented access continuation: original_status=%d status=%d address=%llu bytes=%llu\n",
                        static_cast<int>(result), static_cast<int>(status),
                        static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes));
                    return status;
                },
                [&]() noexcept -> CUresult {
                    if (const int error = identity_error()) {
                        std::fprintf(stderr, "DDO segmented NOT_READY barrier: called=0 identity_status=%d status=unavailable\n", error);
                        return static_cast<CUresult>(error);
                    }
                    const auto status = cuCtxSynchronize();
                    last_access_.barrier_observed = true;
                    last_access_.barrier = static_cast<int>(status);
                    std::fprintf(stderr, "DDO segmented NOT_READY barrier: called=1 identity_status=0 status=%d\n",
                        static_cast<int>(status));
                    return status;
                });
            return static_cast<int>(continued);
        }
        return static_cast<int>(result);
    }
    int unmap(uintptr_t address, size_t bytes) noexcept {
        if (const int error = identity_error()) return error;
        return static_cast<int>(DDO_SEGMENT_TIMED(segment_unmap, bytes, cuMemUnmap(address, bytes)));
    }
    int release(Handle handle) noexcept {
        if (const int error = identity_error()) return error;
        return static_cast<int>(DDO_SEGMENT_TIMED(segment_release, 0, cuMemRelease(handle)));
    }
};
}
