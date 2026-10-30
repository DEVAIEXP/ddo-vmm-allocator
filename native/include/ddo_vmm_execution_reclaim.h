#pragma once
#include <cstddef>
#include <cstdint>

namespace execution_reclaim {
enum class Result { success, busy, failed };
enum class Pressure { unavailable, retain, reclaim };
inline Pressure assess_current(uint64_t budget, uint64_t usage, uint64_t margin) {
    if (!budget) return Pressure::unavailable;
    if (usage > budget) return Pressure::reclaim;
    const auto headroom = budget - usage;
    return margin <= headroom ? Pressure::retain : Pressure::reclaim;
}

inline Pressure assess(uint64_t required, uint64_t budget, uint64_t usage, uint64_t margin) {
    if (!required || !budget) return Pressure::unavailable;
    const auto headroom = usage < budget ? budget - usage : 0;
    return required <= headroom && margin <= headroom - required
        ? Pressure::retain : Pressure::reclaim;
}

// Query must prove CPU-visible completion. A stream wait is insufficient.
// Failure retains ownership and permanently refuses further reuse/close.
template<class Query, class Release>
Result run(bool leased, bool& poisoned, bool& materialized,
           std::size_t pages, Query query, Release release) {
    if (poisoned) return Result::failed;
    if (leased) return Result::busy;
    if (!materialized) return Result::success;
    const auto ready = query();
    if (ready == Result::busy) return ready;
    if (ready != Result::success) {
        poisoned = true;
        return Result::failed;
    }
    for (std::size_t index = 0; index < pages; ++index) {
        if (!release(index)) {
            poisoned = true;
            return Result::failed;
        }
    }
    materialized = false;
    return Result::success;
}
}
