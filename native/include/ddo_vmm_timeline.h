#pragma once
#include "ddo_vmm.h"

// Observation only. Types are stable C ABI; timestamps are signed CUDA ms.
typedef struct ddo_vmm_timeline ddo_vmm_timeline;
typedef struct ddo_vmm_timeline_sample {
    uint64_t sequence, unit, stream, bytes;
    uint32_t kind, state; // kinds 1..5; state 0 pending, 1 complete, 2 failed
    float start_ms, end_ms;
} ddo_vmm_timeline_sample;
typedef struct ddo_vmm_timeline_info {
    uint64_t capacity, count, dropped, failures, pending;
    uint32_t sealed, version;
} ddo_vmm_timeline_info;

extern "C" {
DDO_VMM_API int ddo_vmm_timeline_open_v1(size_t capacity, void* stream, ddo_vmm_timeline** out);
DDO_VMM_API uint64_t ddo_vmm_timeline_mark_begin_v1(uint32_t kind, uint64_t unit, uint64_t bytes, void* stream);
DDO_VMM_API void ddo_vmm_timeline_mark_end_v1(uint64_t token, void* stream);
DDO_VMM_API int ddo_vmm_timeline_seal_v1(ddo_vmm_timeline* timeline);
DDO_VMM_API int ddo_vmm_timeline_snapshot_v1(ddo_vmm_timeline* timeline,
    ddo_vmm_timeline_info* info, ddo_vmm_timeline_sample* samples, size_t capacity);
DDO_VMM_API int ddo_vmm_timeline_close_v1(ddo_vmm_timeline** timeline);
}

namespace ddo_vmm_detail {
static_assert(sizeof(ddo_vmm_timeline_sample) == 48, "timeline sample ABI v1");
static_assert(sizeof(ddo_vmm_timeline_info) == 48, "timeline info ABI v1");
inline uint64_t timeline_token(uint32_t generation, size_t index) {
    return (static_cast<uint64_t>(generation) << 32) | (index + 1);
}
inline size_t timeline_index(uint64_t token, uint32_t generation, size_t count) {
    const uint32_t index = static_cast<uint32_t>(token);
    return token >> 32 == generation && index && index <= count ? index - 1 : SIZE_MAX;
}
inline size_t admit_timeline_sample(size_t capacity, size_t& count, uint64_t& dropped) {
    if (count >= capacity) { ++dropped; return SIZE_MAX; }
    return count++;
}
class TimelineScope {
    uint64_t token_;
    void* stream_;
public:
    TimelineScope(uint32_t kind, uint64_t unit, uint64_t bytes, void* stream)
        : token_(ddo_vmm_timeline_mark_begin_v1(kind, unit, bytes, stream)), stream_(stream) {}
    ~TimelineScope() { ddo_vmm_timeline_mark_end_v1(token_, stream_); }
    TimelineScope(const TimelineScope&) = delete;
    TimelineScope& operator=(const TimelineScope&) = delete;
};
}
