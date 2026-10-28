#include "ddo_vmm_allocation_attempt.h"
#include <cassert>
#include <initializer_list>

enum Result { success, oom, invalid_context, driver_lost };

int main() {
    using ddo_vmm_detail::allocation_attempt;
    for (Result initial : {success, invalid_context, driver_lost}) {
        int creates = 0, evictions = 0;
        const auto result = allocation_attempt(
            [&] { ++creates; return initial; },
            [&] { ++evictions; return true; }, oom, true);
        assert(result == initial && creates == 1 && evictions == 0);
    }
    {
        int evictions = 0;
        const auto result = allocation_attempt([] { return oom; },
            [&] { ++evictions; return true; }, oom, false);
        assert(result == oom && evictions == 0);
    }
    for (Result final : {success, oom, invalid_context, driver_lost}) {
        int creates = 0, evictions = 0;
        const auto result = allocation_attempt(
            [&] { return ++creates == 1 ? oom : final; },
            [&] { return ++evictions == 1; }, oom, true);
        assert(result == final && creates == 2);
        assert(evictions == (final == oom ? 2 : 1));
    }
    {
        int creates = 0, evictions = 0;
        const auto result = allocation_attempt([&] { ++creates; return oom; },
            [&] { ++evictions; return false; }, oom, true);
        assert(result == oom && creates == 1 && evictions == 1);
    }
}
