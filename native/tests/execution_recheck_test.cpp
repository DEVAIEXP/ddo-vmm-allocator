#include "ddo_vmm_execution_recheck.h"
#include "ddo_vmm_execution_reclaim.h"
#include <cstdio>
#include <stdexcept>
#include <limits>

static void require(bool valid) {
    if (!valid) throw std::runtime_error("execution reuse pressure contract failed");
}

int main() {
    using namespace execution_reclaim;
    ProbeCadence gate;
    require(gate.claim(0));
    for (uint64_t tick = 0; tick < 250; ++tick) require(!gate.claim(tick));
    require(gate.claim(250));
    require(!gate.claim(249)); // Clock regression cannot generate extra probes.
    require(!gate.claim(499));
    require(gate.claim(500));
    gate.observed(700); // An existing growth query postpones the reuse check.
    require(!gate.claim(949));
    require(gate.claim(950));
    ProbeCadence other_pool;
    require(other_pool.claim(950)); // Independent execution pools/devices.
    const auto maximum = std::numeric_limits<uint64_t>::max();
    other_pool.observed(maximum - 100);
    require(!other_pool.claim(maximum));
    require(!other_pool.claim(0)); // Unsigned rollover is not elapsed time.
    other_pool.observed(maximum - 250);
    require(other_pool.claim(maximum));
    require(!other_pool.claim(maximum));

    unsigned polls = 0;
    ProbeCadence flood;
    for (uint64_t ms = 0; ms < 1000; ++ms)
        for (unsigned allocation = 0; allocation < 100; ++allocation)
            if (flood.claim(ms)) ++polls;
    require(polls == 4); // 100000 allocation attempts, at most four probes.

    require(assess_current(1024, 768, 256) == Pressure::retain);
    require(assess_current(1024, 769, 256) == Pressure::reclaim);
    require(assess_current(0, 0, 256) == Pressure::unavailable);
    require(assess_current(1024, 1025, 0) == Pressure::reclaim);
    require(assess_current(maximum, 0, maximum) == Pressure::retain);
    require(assess_current(maximum, maximum, maximum) == Pressure::reclaim);
    require(assess(0, 1024, 0, 256) == Pressure::unavailable); // Growth ABI unchanged.

    // Reuse does not reserve another allocation's bytes. Adequate current
    // headroom must retain backing even if a hypothetical growth would not fit.
    require(assess_current(1024, 768, 256) == Pressure::retain);
    require(assess(1, 1024, 768, 256) == Pressure::reclaim);
    std::puts("{\"status\":\"pass\",\"allocation_attempts\":100000,\"bounded_probes\":4,\"cuda_loaded\":false}");
}
