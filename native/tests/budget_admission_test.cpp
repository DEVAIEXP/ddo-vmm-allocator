#include "ddo_vmm_budget_admission.h"
#include <cassert>
#include <limits>

int main() {
    using namespace ddo_vmm_detail;
    const auto maximum = std::numeric_limits<uint64_t>::max();
    assert(budget_room(1024, 100, 900, 24));
    assert(!budget_room(1024, 100, 900, 25));
    assert(!budget_room(100, 101, 0, 0));
    assert(!budget_room(100, 0, 0, 101));
    assert(budget_room(maximum, 0, maximum, 0));
    assert(!budget_room(maximum, 0, maximum, 1));
    assert(!budget_room(maximum, maximum, 1, maximum));
    assert(budget_room(0, 0, 0, 0));
    int observations = 0, evictions = 0;
    assert(!make_budget_room([&] { ++observations; return BudgetAdmission::unavailable; },
                             [&] { ++evictions; return true; }));
    assert(observations == 1 && evictions == 0);
    observations = evictions = 0;
    assert(make_budget_room([&] { return ++observations == 1
        ? BudgetAdmission::insufficient : BudgetAdmission::allowed; },
        [&] { ++evictions; return true; }));
    assert(observations == 2 && evictions == 1);
    observations = evictions = 0;
    assert(!make_budget_room([&] { return ++observations == 1
        ? BudgetAdmission::insufficient : BudgetAdmission::unavailable; },
        [&] { ++evictions; return true; }));
    assert(observations == 2 && evictions == 1);
    assert(!make_budget_room([] { return BudgetAdmission::insufficient; }, [] { return false; }));
    assert(make_budget_room([] { return BudgetAdmission::allowed; }, [] { assert(false); return true; }));
}
