#include "ddo_vmm_mapping_trace.h"
#if defined(DDO_VMM_MAPPING_TRACE) && defined(_WIN32)
#include "ddo_vmm_mapping_trace_core.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cuda.h>

namespace {
ddo_vmm_detail::MappingTrace trace;
thread_local unsigned refused_access_calls = 0;
thread_local CUresult injected_access_result = CUDA_ERROR_NOT_READY;
thread_local bool access_refusal_armed_once = false;
uint64_t ticks() noexcept {
    LARGE_INTEGER value{};
    return QueryPerformanceCounter(&value) ? static_cast<uint64_t>(value.QuadPart) : 0;
}
uint64_t filetime(FILETIME value) noexcept {
    return (static_cast<uint64_t>(value.dwHighDateTime) << 32) | value.dwLowDateTime;
}
ddo_vmm_mapping_record_v1 record(uint64_t action) noexcept {
    ddo_vmm_mapping_record_v1 row{};
    row.action = action;
    row.thread = GetCurrentThreadId();
    row.device = UINT64_MAX;
    return row;
}
}

extern "C" int ddo_vmm_mapping_trace_start_v1(size_t capacity, size_t byte_limit) {
    ddo_vmm_detail::MappingTrace::Header identity{};
    FILETIME creation{}, exit{}, kernel{}, user{}, utc{};
    LARGE_INTEGER frequency{};
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)
        || !QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0) return 1;
    identity[6] = GetCurrentProcessId(); identity[7] = filetime(creation);
    identity[8] = static_cast<uint64_t>(frequency.QuadPart); identity[9] = 1;
    identity[10] = ticks();
    GetSystemTimePreciseAsFileTime(&utc);
    identity[11] = filetime(utc); identity[12] = ticks();
    if (!identity[10] || identity[12] < identity[10] || !identity[11]) return 1;
    return trace.start(capacity, byte_limit, identity);
}
extern "C" int ddo_vmm_mapping_trace_stop_v1() { return trace.stop(); }
extern "C" __declspec(dllexport) int ddo_test_arm_access_refusal(unsigned calls) {
    if (access_refusal_armed_once || calls < 1 || calls > 2) return 1;
    access_refusal_armed_once = true;
    refused_access_calls = calls;
    return 0;
}
extern "C" __declspec(dllexport) int ddo_test_arm_access_oom() {
    if (ddo_test_arm_access_refusal(1)) return 1;
    injected_access_result = CUDA_ERROR_OUT_OF_MEMORY;
    return 0;
}
extern "C" int ddo_vmm_mapping_trace_snapshot_v1(uint64_t* header, size_t count,
    ddo_vmm_mapping_record_v1* rows, size_t capacity) {
    return trace.snapshot(header, count, rows, capacity);
}

// Wrappers preserve exact driver arguments/results and never query CUDA context.
CUresult ddo_trace_mem_map(CUdeviceptr address, size_t bytes, size_t offset,
    CUmemGenericAllocationHandle allocation, unsigned long long flags) {
    auto row = record(1); row.address = address; row.bytes = bytes;
    row.allocation_offset = offset; row.allocation = allocation; row.flags = flags;
    return trace.measure(row, [&] { return cuMemMap(address, bytes, offset, allocation, flags); }, ticks);
}
CUresult ddo_trace_mem_unmap(CUdeviceptr address, size_t bytes) {
    auto row = record(2); row.address = address; row.bytes = bytes;
    return trace.measure(row, [&] { return cuMemUnmap(address, bytes); }, ticks);
}
CUresult ddo_trace_mem_release(CUmemGenericAllocationHandle allocation) {
    auto row = record(3); row.allocation = allocation;
    return trace.measure(row, [&] { return cuMemRelease(allocation); }, ticks);
}
CUresult ddo_trace_mem_set_access(CUdeviceptr address, size_t bytes,
    const CUmemAccessDesc* descriptors, size_t count) {
    auto row = record(4); row.address = address; row.bytes = bytes;
    return trace.measure(row, [&] {
        if (refused_access_calls) {
            --refused_access_calls;
            return injected_access_result;
        }
        return cuMemSetAccess(address, bytes, descriptors, count);
    }, ticks);
}
#else
extern "C" int ddo_vmm_mapping_trace_start_v1(size_t, size_t) { return 2; }
extern "C" int ddo_vmm_mapping_trace_stop_v1() { return 2; }
extern "C" int ddo_vmm_mapping_trace_snapshot_v1(uint64_t*, size_t,
    ddo_vmm_mapping_record_v1*, size_t) { return 2; }
#endif
