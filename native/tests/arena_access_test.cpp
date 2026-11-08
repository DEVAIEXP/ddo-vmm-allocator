#include "ddo_vmm_access_recovery.h"
#include "ddo_vmm_arena_access.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
constexpr int success = 0, oom = 2, pending = 600, illegal = 700, unknown = 999;
int checks = 0;
void require(bool condition) {
    ++checks;
    if (!condition) { std::cerr << "FAIL check " << checks << '\n'; std::exit(1); }
}
void rollback_case(int access, bool barrier_failed, std::array<int, 3> replies,
                   int expected_steps, bool clean, bool mapped, bool live) {
    std::vector<int> order;
    const auto result = ddo_vmm_detail::rollback_arena_access(
        access, success, oom, barrier_failed,
        [&] { order.push_back(1); return replies[0]; },
        [&] { order.push_back(2); return replies[1]; },
        [&] { order.push_back(3); return replies[2]; });
    require(static_cast<int>(order.size()) == expected_steps);
    for (int i = 0; i < expected_steps; ++i) require(order[i] == i + 1);
    require((result.disposition == ddo_vmm_detail::ArenaAccessDisposition::clean_oom) == clean);
    require(result.mapped == mapped);
    require(result.allocation_live == live);
    require(result.context_released == clean);
}
void sequence_case(std::array<int, 2> access, int barrier, int expected_access,
                   int expected_barriers, bool expected_clean) {
    int accesses = 0, barriers = 0, cleanups = 0;
    bool barrier_failed = false;
    const auto status = ddo_vmm_detail::recover_access_once(success, pending,
        [&] { return access.at(accesses++); },
        [&] { ++barriers; barrier_failed = barrier != success; return barrier; });
    // Mirror the production failure-only entry. Success publishes normally.
    if (status != success) {
        const auto result = ddo_vmm_detail::rollback_arena_access(
            status, success, oom, barrier_failed,
            [&] { ++cleanups; return success; },
            [&] { ++cleanups; return success; },
            [&] { ++cleanups; return success; });
        require((result.disposition == ddo_vmm_detail::ArenaAccessDisposition::clean_oom) == expected_clean);
    }
    require(accesses == expected_access);
    require(barriers == expected_barriers);
    require(cleanups == (expected_clean ? 3 : 0));
}
}
int main() {
    rollback_case(oom, false, {success, success, success}, 3, true, false, false);
    rollback_case(oom, false, {unknown, success, success}, 1, false, true, true);
    rollback_case(oom, false, {success, unknown, success}, 2, false, false, true);
    rollback_case(oom, false, {success, success, unknown}, 3, false, false, false);
    rollback_case(oom, true, {success, success, success}, 0, false, true, true);
    for (int error : {illegal, unknown, pending})
        rollback_case(error, false, {success, success, success}, 0, false, true, true);
    // A synchronization OOM is a failed completion, never allocation pressure.
    sequence_case({pending, success}, oom, 1, 1, false);
    sequence_case({pending, success}, illegal, 1, 1, false);
    sequence_case({pending, success}, success, 2, 1, false);
    sequence_case({pending, oom}, success, 2, 1, true);
    sequence_case({pending, pending}, success, 2, 1, false);
    sequence_case({pending, illegal}, success, 2, 1, false);
    sequence_case({oom, success}, success, 1, 0, true);
    sequence_case({illegal, success}, success, 1, 0, false);
    sequence_case({success, illegal}, success, 1, 0, false);
    std::cout << "PASS " << checks << " CPU assertions; no CUDA calls\n";
}
