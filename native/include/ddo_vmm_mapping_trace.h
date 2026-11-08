#pragma once
#include "ddo_vmm.h"

// Internal diagnostic ABI, not a residency-control or physical-placement API.
// Rows contain exactly 16 uint64 values. Result is the original driver status.
// Actions: 1 map, 2 unmap, 3 release, 4 set access.
struct ddo_vmm_mapping_record_v1 {
    uint64_t sequence, action, address, bytes, allocation, allocation_offset;
    uint64_t flags, result, begin_ticks, end_ticks, thread;
    uint64_t owner, generation, role, device, context;
};
// Header: schema, state(0 unused/1 recording/2 stopped), capacity, count,
// dropped, in_flight, pid, process_creation, frequency, clock_kind(1 QPC),
// anchor_before, anchor_utc_filetime, anchor_after, reserved, clock_failures,
// next_sequence. Rows append in return order; sequence denotes admission order.
// A zero owner/generation/role/context or UINT64_MAX device means unavailable.
// Start/stop/snapshot perform no CUDA calls. Stop refuses in-flight driver calls.
// Start is one-shot; capacity is explicit and must be <= the caller's byte limit.
extern "C" {
DDO_VMM_API int ddo_vmm_mapping_trace_start_v1(size_t capacity, size_t byte_limit);
DDO_VMM_API int ddo_vmm_mapping_trace_stop_v1();
DDO_VMM_API int ddo_vmm_mapping_trace_snapshot_v1(uint64_t* header, size_t header_count,
    ddo_vmm_mapping_record_v1* rows, size_t row_capacity);
}
static_assert(sizeof(ddo_vmm_mapping_record_v1) == 128);
