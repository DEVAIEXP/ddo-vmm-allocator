#include "host_platform.h"
#include <cassert>
#include <limits>

int main() {
    using namespace ddo_host_platform;
    const auto maximum = std::numeric_limits<uint64_t>::max();
    assert(permits({1024, 1024}, 1024, 0));
    assert(!permits({1024, 1023}, 1024, 0));
    assert(!permits({1023, 1024}, 1024, 0));
    assert(permits({1024, 1024}, 1000, 24));
    assert(!permits({1024, 1024}, 1000, 25));
    assert(permits({maximum, maximum}, maximum, 0));
    assert(!permits({maximum, maximum}, maximum, 1));
    assert(!permits({maximum, maximum}, 1, maximum));
    assert(!permits({0, 0}, 1, 0));
    int observations = 0;
    auto observe = [&](Headroom& memory) {
        ++observations;
        memory = {1024, 1024};
        return true;
    };
    assert(admission(1024, 0, observe) == Admission::allowed);
    assert(admission(1024, 1, observe) == Admission::insufficient);
    assert(observations == 2);
    assert(admission(1, 0, [maximum](Headroom& memory) {
        memory = {maximum, maximum};
        return false;
    }) == Admission::unavailable);
#if defined(_WIN32)
    SetLastError(ERROR_COMMITMENT_LIMIT);
    const auto original = allocation_failure_message("host ring storage allocation");
    SetLastError(ERROR_ACCESS_DENIED);
    assert(original == "host ring storage allocation failed: Win32 error 1455");
#elif defined(__linux__)
    errno = ENOMEM;
    const auto original = allocation_failure_message("host ring storage allocation");
    errno = EACCES;
    assert(original == "host ring storage allocation failed: errno " + std::to_string(ENOMEM));
#endif
}
