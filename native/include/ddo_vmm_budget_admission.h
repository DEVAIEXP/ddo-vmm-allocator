#pragma once
#include <cstdint>

namespace ddo_vmm_detail {
enum class BudgetAdmission { allowed, insufficient, unavailable };

inline bool budget_room(uint64_t budget, uint64_t usage, uint64_t required,
                        uint64_t reserve) {
    return usage <= budget && reserve <= budget - usage
        && required <= budget - usage - reserve;
}

template <typename Observe, typename Evict>
bool make_budget_room(Observe observe, Evict evict) {
    for (;;) {
        const auto decision = observe();
        if (decision == BudgetAdmission::allowed) return true;
        // An unavailable observation is not evidence that eviction can help.
        if (decision == BudgetAdmission::unavailable || !evict()) return false;
    }
}
}
