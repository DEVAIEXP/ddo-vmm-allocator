#include "ddo_vmm_allocation_diagnostics.h"
#include <array>
#include <cstdlib>

using ddo_vmm_detail::AllocationDiagnostics;
void require(bool condition) { if (!condition) std::abort(); }
int main() {
    AllocationDiagnostics counters;
    int called = 0;
    auto call = [&] { ++called; return 0; };
    require(counters.measure(AllocationDiagnostics::create, call) == 0 && called == 1);
    std::array<uint64_t, 26> snapshot;
    snapshot.fill(99);
    require(counters.snapshot(snapshot.data(), 25) == 1 && snapshot[0] == 99);
    require(counters.snapshot(nullptr, 26) == 1);
    require(counters.snapshot(snapshot.data(), 26) == 0 && snapshot[2] == 0);
    require(counters.start() && !counters.start());
    require(counters.measure(AllocationDiagnostics::create, call) == 0 && called == 2);
    require(counters.measure(AllocationDiagnostics::map, [] { return 7; }) == 7);
    counters.pool_reuse();
    counters.pool_backing(1024);
    counters.pool_backing(512);
    require(counters.snapshot(snapshot.data(), 26) == 0);
    require(snapshot[0] == 1 && snapshot[1] == 1 && snapshot[2] == 1 && snapshot[4] == 0);
    require(snapshot[5] == 1 && snapshot[7] == 1);
    require(snapshot[23] == 1 && snapshot[24] == 2 && snapshot[25] == 1024);
    counters.enabled.store(false);
    counters.pool_reuse();
    counters.pool_backing(4096);
    require(counters.measure(AllocationDiagnostics::create, call) == 0 && called == 3);
    require(counters.snapshot(snapshot.data(), 26) == 0 && snapshot[1] == 0 && snapshot[2] == 1);
    require(snapshot[23] == 1 && snapshot[25] == 1024);
    std::array<uint64_t, 36> extended;
    extended.fill(99);
    require(counters.snapshot_v2(extended.data(), 35) == 1 && extended[0] == 99);
    require(counters.snapshot_v2(nullptr, 36) == 1);
    counters.access_result(600, 600, 2); // Disabled: no observations.
    { AllocationDiagnostics::FaultScope scope(counters); }
    require(counters.start());
    counters.access_result(0, 600, 2);
    counters.access_result(600, 600, 2);
    counters.access_result(2, 600, 2);
    counters.access_result(99, 600, 2);
    int sync_calls = 0;
    require(counters.measure(AllocationDiagnostics::access_retry_sync,
        [&] { ++sync_calls; return 7; }) == 7 && sync_calls == 1);
    { AllocationDiagnostics::FaultScope scope(counters); }
    require(counters.snapshot_v2(extended.data(), 36) == 0);
    require(extended[0] == 2 && extended[26] == 1 && extended[28] == 1);
    require(extended[29] == 1 && extended[31] == 0);
    for (size_t i = 32; i < 36; ++i) require(extended[i] == 1);
    require(counters.snapshot(snapshot.data(), 26) == 0 && snapshot[0] == 1);
    return 0;
}
