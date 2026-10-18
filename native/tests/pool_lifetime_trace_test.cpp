#include "ddo_vmm_pool_lifetime_trace.h"
#include "ddo_vmm_pool_dependency_trace.h"
#include <array>
#include <cstdlib>
#include <iostream>

void require(bool condition) { if (!condition) std::abort(); }
int main() {
    ddo_vmm_detail::PoolLifetimeTrace<3> trace;
    std::array<uint64_t, 4> report{};
    std::array<uint64_t, 39> rows{};
    auto add = [&](uint64_t pointer) { trace.record(1, pointer, 4096, 8192, 0,
        512, 500, 7, pointer, 0, 0, 0); };
    add(1);
    require(trace.snapshot(report.data(), 4, rows.data(), 3) == 0);
    require(report[1] == 0 && report[2] == 0 && report[3] == 0);
    require(trace.start() && !trace.start());
    for (uint64_t index = 1; index <= 5; ++index) add(index);
    require(trace.snapshot(report.data(), 4, rows.data(), 2) != 0);
    require(trace.snapshot(report.data(), 4, nullptr, 3) != 0);
    require(trace.snapshot(report.data(), 4, rows.data(), 3) == 0);
    require(report[0] == 1 && report[1] == 1 && report[2] == 5 && report[3] == 3);
    require(trace.total_sequence() == 5);
    for (size_t index = 0; index < 3; ++index) {
        require(rows[index * 13] == index + 3);
        require(rows[index * 13 + 2] == index + 3);
        require(rows[index * 13 + 8] == 7);
    }
    ddo_vmm_detail::PoolDependencyTrace<3> dependency;
    require(dependency.start() && !dependency.start());
    dependency.record(2, 4096, 4096, 9, 7, 42, 43, 3, 123, 456, 0, 5);
    require(dependency.snapshot(report.data(), 4, rows.data(), 3) == 0);
    require(report[2] == 1 && report[3] == 1);
    const std::array<uint64_t, 13> expected{1, 2, 4096, 4096, 9, 7, 42, 43, 3, 123, 456, 0, 5};
    for (size_t index = 0; index < 13; ++index) require(rows[index] == expected[index]);
    ddo_vmm_detail::PoolLifetimeTrace<3> trigger;
    require(trigger.start());
    trigger.trigger(2, 461373440, 0, 123, 8);
    require(trigger.snapshot(report.data(), 4, rows.data(), 3) == 0);
    const std::array<uint64_t, 13> expected_trigger{1, 6, 0, 0, 0, 2, 0, 461373440, 0, 0, 123, 0, 8};
    for (size_t index = 0; index < 13; ++index) require(rows[index] == expected_trigger[index]);
    std::cout << "Pool lifetime trace CPU contracts: PASS\n";
}
