#include "ddo_vmm_access_recovery.h"
#include <array>

namespace {
constexpr int success = 0, pending = 600, illegal_address = 700, oom = 2;

bool check(std::array<int, 2> replies, int barrier_result, int expected,
           int expected_accesses, int expected_barriers) {
    int accesses = 0, barriers = 0;
    const auto result = ddo_vmm_detail::recover_access_once(success, pending,
        [&] { return replies.at(accesses++); },
        [&] { ++barriers; return barrier_result; });
    return result == expected && accesses == expected_accesses && barriers == expected_barriers;
}
}

int main() {
    return !(check({success, pending}, success, success, 1, 0)
        && check({oom, success}, success, oom, 1, 0)
        && check({pending, success}, success, success, 2, 1)
        && check({pending, pending}, success, pending, 2, 1)
        && check({pending, oom}, success, oom, 2, 1)
        && check({pending, success}, illegal_address, illegal_address, 1, 1));
}
