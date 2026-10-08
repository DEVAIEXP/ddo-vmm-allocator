#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace ddo_vmm_detail {
// Fixed-size, externally serialized observations. Never makes policy decisions,
// allocates storage, samples the OS, waits for events or changes ownership.
struct PressureObservations {
    static constexpr size_t count = 23;
    std::array<uint64_t, count> values{1};

    void sample(bool valid, uint64_t ram, uint64_t commit, uint64_t low, uint64_t high) {
        values[18] = low; values[19] = high;
        if (!valid) { ++values[2]; return; }
        if (!values[1]) { values[16] = ram; values[17] = commit; }
        else { values[16] = std::min(values[16], ram); values[17] = std::min(values[17], commit); }
        ++values[1];
    }
    void transition(bool before, bool after, bool valid, uint64_t ram,
                    uint64_t commit, uint64_t low, uint64_t committed, uint64_t registered) {
        if (before == after) return;
        if (after) {
            ++values[3];
            const uint64_t reason = !valid ? 4 : (ram < low ? 1 : 0) + (commit < low ? 2 : 0);
            values[9] = reason;
            if (reason >= 1 && reason <= 4) ++values[4 + reason];
            values[10] = valid ? ram : 0; values[11] = valid ? commit : 0;
            values[12] = committed; values[13] = registered;
            values[20] = 0;
        } else {
            ++values[4]; values[14] = ram; values[15] = commit;
        }
    }
    void reclaimed(uint64_t bytes) { values[20] += bytes; }
    void blocked() { ++values[21]; }
    void failed() { ++values[22]; }
};
} // namespace ddo_vmm_detail
