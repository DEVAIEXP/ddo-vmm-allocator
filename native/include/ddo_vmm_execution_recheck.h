#pragma once
#include <cstdint>

namespace execution_reclaim {
// Protected by the execution-pool mutex. The clock is injected in CPU tests.
class ProbeCadence {
    bool observed_ = false;
    uint64_t previous_ms_ = 0;
public:
    static constexpr uint64_t interval_ms = 250;
    bool claim(uint64_t now_ms) noexcept {
        if (observed_ && (now_ms < previous_ms_ || now_ms - previous_ms_ < interval_ms))
            return false;
        observed(now_ms);
        return true;
    }
    void observed(uint64_t now_ms) noexcept {
        previous_ms_ = now_ms;
        observed_ = true;
    }
};
}

namespace ddo_vmm_detail {
// Internal allocator entry: no new allocation is requested by reuse itself.
int recheck_execution_backing(int device);
}
