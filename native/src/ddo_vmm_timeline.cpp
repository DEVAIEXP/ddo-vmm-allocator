#include "ddo_vmm_timeline.h"
#include <cuda.h>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

struct TimingSlot {
    CUevent start = nullptr, end = nullptr;
    ddo_vmm_timeline_sample sample{};
    bool started = false, ended = false, failed = false;
};
struct ddo_vmm_timeline {
    CUcontext context = nullptr;
    CUevent epoch = nullptr;
    uint32_t generation = 0;
    size_t count = 0;
    uint64_t dropped = 0, failures = 0;
    bool sealed = false;
    std::vector<TimingSlot> slots;
};
namespace {
std::mutex mutex;
std::atomic<size_t> active_count{0};
uint32_t generation = 0;
std::map<CUcontext, ddo_vmm_timeline*> active;
// Pending owners stay here after a nonblocking close. No destructor CUDA call:
// if callers abandon a failed/pending diagnostic, driver teardown owns events.
std::map<ddo_vmm_timeline*, std::unique_ptr<ddo_vmm_timeline>> owners;
bool current(ddo_vmm_timeline* t) {
    CUcontext ctx = nullptr;
    return t && owners.count(t) && cuCtxGetCurrent(&ctx) == CUDA_SUCCESS && ctx == t->context;
}
void destroy_events(ddo_vmm_timeline& t) {
    for (auto& s : t.slots) {
        if (s.start) cuEventDestroy(s.start);
        if (s.end) cuEventDestroy(s.end);
    }
    if (t.epoch) cuEventDestroy(t.epoch);
}
void seal(ddo_vmm_timeline& t) {
    if (!t.sealed) {
        active.erase(t.context);
        active_count.fetch_sub(1, std::memory_order_relaxed);
        t.sealed = true;
    }
}
}

extern "C" int ddo_vmm_timeline_open_v1(size_t capacity, void* stream, ddo_vmm_timeline** out) {
    if (!out || !capacity || capacity > 8192) return DDO_VMM_INVALID_ARGUMENT;
    *out = nullptr;
    std::lock_guard lock(mutex);
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx) return DDO_VMM_CUDA_ERROR;
    if (active.count(ctx) || generation == UINT32_MAX) return DDO_VMM_INVALID_ARGUMENT;
    for (const auto& [raw, owner] : owners) {
        (void)raw;
        if (owner->context == ctx) return DDO_VMM_INVALID_ARGUMENT;
    }
    try {
        auto t = std::make_unique<ddo_vmm_timeline>();
        t->context = ctx;
        t->generation = ++generation;
        t->slots.resize(capacity);
        // Preallocate all events outside the sampled phase. None is re-recorded.
        for (auto& s : t->slots) {
            if (cuEventCreate(&s.start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
                cuEventCreate(&s.end, CU_EVENT_DEFAULT) != CUDA_SUCCESS) {
                destroy_events(*t);
                return DDO_VMM_CUDA_ERROR;
            }
        }
        if (cuEventCreate(&t->epoch, CU_EVENT_DEFAULT) != CUDA_SUCCESS) {
            destroy_events(*t); return DDO_VMM_CUDA_ERROR;
        }
        auto* raw = t.get();
        // Allocate the registry node before moving the event owner, so an
        // allocation exception can still clean up unrecorded events.
        try { owners.try_emplace(raw); }
        catch (...) { destroy_events(*t); throw; }
        owners[raw] = std::move(t);
        try { active.emplace(ctx, raw); }
        catch (...) { destroy_events(*raw); owners.erase(raw); throw; }
        if (cuEventRecord(raw->epoch, static_cast<CUstream>(stream)) != CUDA_SUCCESS) {
            active.erase(ctx); destroy_events(*raw); owners.erase(raw);
            return DDO_VMM_CUDA_ERROR;
        }
        active_count.fetch_add(1, std::memory_order_relaxed);
        *out = raw;
        return DDO_VMM_SUCCESS;
    } catch (...) { return DDO_VMM_OUT_OF_MEMORY; }
}

extern "C" uint64_t ddo_vmm_timeline_mark_begin_v1(uint32_t kind, uint64_t unit,
    uint64_t bytes, void* stream) {
    if (!active_count.load(std::memory_order_relaxed)) return 0; // No CUDA when off.
    std::lock_guard lock(mutex);
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS) {
        for (auto& [key, t] : active) { (void)key; ++t->failures; }
        return 0;
    }
    auto found = active.find(ctx);
    if (found == active.end()) return 0;
    auto& t = *found->second;
    if (kind < 1 || kind > 5) { ++t.failures; return 0; }
    const auto index = ddo_vmm_detail::admit_timeline_sample(t.slots.size(), t.count, t.dropped);
    if (index == SIZE_MAX) return 0;
    auto& s = t.slots[index];
    s.sample = {index + 1, unit, reinterpret_cast<uintptr_t>(stream), bytes, kind, 0, 0, 0};
    s.started = cuEventRecord(s.start, static_cast<CUstream>(stream)) == CUDA_SUCCESS;
    if (!s.started) { s.failed = true; ++t.failures; }
    return ddo_vmm_detail::timeline_token(t.generation, index);
}

extern "C" void ddo_vmm_timeline_mark_end_v1(uint64_t token, void* stream) {
    if (!token) return;
    std::lock_guard lock(mutex);
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS) {
        for (auto& [raw, owner] : owners) { (void)raw; ++owner->failures; }
        return;
    }
    // Sealed sessions still permit their own outstanding scope to end.
    for (auto& [raw, owner] : owners) {
        auto& t = *owner;
        if (t.generation != token >> 32 || t.context != ctx) continue;
        const size_t index = ddo_vmm_detail::timeline_index(token, t.generation, t.count);
        if (index == SIZE_MAX) { ++t.failures; return; }
        auto& s = t.slots[index];
        if (s.ended || s.failed || s.sample.stream != reinterpret_cast<uintptr_t>(stream)) {
            ++t.failures; return;
        }
        s.ended = cuEventRecord(s.end, static_cast<CUstream>(stream)) == CUDA_SUCCESS;
        if (!s.ended) { s.failed = true; ++t.failures; }
        return;
    }
}

extern "C" int ddo_vmm_timeline_seal_v1(ddo_vmm_timeline* t) {
    std::lock_guard lock(mutex);
    if (!current(t)) return DDO_VMM_INVALID_ARGUMENT;
    seal(*t); return DDO_VMM_SUCCESS;
}

extern "C" int ddo_vmm_timeline_snapshot_v1(ddo_vmm_timeline* t,
    ddo_vmm_timeline_info* info, ddo_vmm_timeline_sample* samples, size_t capacity) {
    std::lock_guard lock(mutex);
    if (!current(t) || !info || !samples || capacity < t->count) return DDO_VMM_INVALID_ARGUMENT;
    *info = {t->slots.size(), t->count, t->dropped, t->failures, 0, t->sealed ? 1U : 0U, 1};
    const auto epoch_status = cuEventQuery(t->epoch);
    for (size_t i = 0; i < t->count; ++i) {
        auto& s = t->slots[i];
        auto row = s.sample;
        if (s.failed || (epoch_status != CUDA_SUCCESS && epoch_status != CUDA_ERROR_NOT_READY)) {
            row.state = 2;
        } else if (!s.started || !s.ended || epoch_status == CUDA_ERROR_NOT_READY) {
            ++info->pending;
        } else {
            auto a = cuEventQuery(s.start), b = cuEventQuery(s.end);
            if (a == CUDA_ERROR_NOT_READY || b == CUDA_ERROR_NOT_READY) ++info->pending;
            else if (a != CUDA_SUCCESS || b != CUDA_SUCCESS ||
                     cuEventElapsedTime(&row.start_ms, t->epoch, s.start) != CUDA_SUCCESS ||
                     cuEventElapsedTime(&row.end_ms, t->epoch, s.end) != CUDA_SUCCESS) row.state = 2;
            else row.state = 1;
        }
        samples[i] = row;
    }
    return DDO_VMM_SUCCESS;
}

extern "C" int ddo_vmm_timeline_close_v1(ddo_vmm_timeline** handle) {
    if (!handle || !*handle) return DDO_VMM_INVALID_ARGUMENT;
    std::lock_guard lock(mutex);
    auto* t = *handle;
    if (!current(t)) return DDO_VMM_INVALID_ARGUMENT;
    seal(*t);
    if (cuEventQuery(t->epoch) != CUDA_SUCCESS) return DDO_VMM_CUDA_ERROR;
    for (size_t i = 0; i < t->count; ++i) {
        auto& s = t->slots[i];
        if (s.failed || !s.started || !s.ended || cuEventQuery(s.start) != CUDA_SUCCESS ||
            cuEventQuery(s.end) != CUDA_SUCCESS) return DDO_VMM_CUDA_ERROR;
    }
    destroy_events(*t);
    owners.erase(t); *handle = nullptr;
    return DDO_VMM_SUCCESS;
}
