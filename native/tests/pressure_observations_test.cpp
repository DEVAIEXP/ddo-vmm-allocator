#include "ddo_vmm_pressure_observations.h"
#include <cstdio>

int main() {
    ddo_vmm_detail::PressureObservations observations;
    auto& v = observations.values;
    int failures = 0;
    const auto check = [&](bool valid) { if (!valid) ++failures; };
    check(v[0] == 1 && v.size() == 23);
    observations.sample(true, 10, 20, 12, 14);
    observations.transition(false, true, true, 10, 20, 12, 100, 80);
    check(v[3] == 1 && v[5] == 1 && v[9] == 1 && v[12] == 100 && v[13] == 80);
    observations.transition(true, true, true, 9, 20, 12, 100, 80);
    check(v[3] == 1);
    observations.reclaimed(64);
    observations.blocked();
    check(v[20] == 64 && v[21] == 1);
    observations.sample(true, 15, 21, 12, 14);
    observations.transition(true, false, true, 15, 21, 12, 36, 16);
    check(v[4] == 1 && v[14] == 15 && v[15] == 21);
    observations.sample(true, 20, 9, 12, 14);
    observations.transition(false, true, true, 20, 9, 12, 36, 16);
    check(v[6] == 1 && v[9] == 2 && v[20] == 0);
    observations.transition(true, false, true, 20, 20, 12, 36, 16);
    observations.transition(false, true, true, 9, 9, 12, 36, 16);
    check(v[7] == 1 && v[9] == 3);
    observations.transition(true, false, true, 20, 20, 12, 36, 16);
    observations.sample(false, 0, 0, 12, 14);
    observations.transition(false, true, false, 0, 0, 12, 36, 16);
    observations.failed();
    check(v[8] == 1 && v[9] == 4 && v[2] == 1 && v[22] == 1);
    check(v[10] == 0 && v[11] == 0 && v[16] == 10 && v[17] == 9);
    for (int i = 0; i < 10000; ++i) observations.sample(true, 20, 20, 12, 14);
    check(v[1] == 10003 && v.size() == 23 && v[18] == 12 && v[19] == 14);
    std::printf("Pressure observations CPU: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
