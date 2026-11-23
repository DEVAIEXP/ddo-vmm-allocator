// Checked-allocation policy for the optional Torch bridge. No CUDA, Torch or
// residency policy: it only decides what happens when the native allocator
// returns null, so PyTorch never receives null storage.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <utility>

namespace ddo_vmm_bridge {

// Layout of ddo_exp_segmented_last_refusal (native/src/ddo_vmm_torch_pool.cpp).
inline constexpr std::size_t kReportFields = 31;
inline constexpr std::size_t kRecorded = 0, kRequest = 1, kDeficit = 2, kT1Released = 15, kT2Bytes = 24,
                                 kTermination = 30;

// Native termination codes (ddo_vmm_torch_pool.cpp, term_*).
inline const char* termination_name(std::uint64_t value) {
    switch (value) {
        case 1: return "satisfied";
        case 2: return "retry_refused";
        case 3: return "no_release";
        case 4: return "partial_release";
        case 5: return "rollback_retained";
        case 6: return "completion_error";
        case 7: return "driver_error";
        case 8: return "quota_consumed";
        default: return "unknown";
    }
}

inline const char* termination_meaning(std::uint64_t value) {
    switch (value) {
        case 2: return "the retry after recovery was still refused";
        case 3: return "no unused GPU backing could be released";
        case 4: return "released GPU backing was not enough";
        case 8: return "the recovery quota was already consumed";
        default: return "allocation was not satisfied";
    }
}

// Capacity refusals only. Completion/driver errors and rollback retention are
// not out-of-memory conditions and must not be reported as such.
inline bool is_capacity_refusal(std::uint64_t termination) {
    return termination == 2 || termination == 3 || termination == 4 || termination == 8;
}

struct RefusalClass {
    bool capacity = false;
    std::string message;
};

// report_status: 0 = report read; otherwise unavailable (no export, lock busy).
inline RefusalClass classify_refusal(std::size_t bytes, int device, int report_status, const std::uint64_t* report) {
    RefusalClass result;
    const bool matching = report_status == 0 && report && report[kRecorded] == 1 && report[kRequest] == bytes;
    if (matching && is_capacity_refusal(report[kTermination])) {
        const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
        char amount[32];
        std::snprintf(amount, sizeof(amount), "%.1f", mib);
        result.capacity = true;
        result.message = std::string("DDO VMM allocator could not obtain ") + amount + " MiB of GPU memory on device "
            + std::to_string(device) + " after recovery (termination=" + termination_name(report[kTermination])
            + ": " + termination_meaning(report[kTermination]) + "; requested_bytes=" + std::to_string(bytes)
            + "). Other applications may be using GPU memory, or this workload's activations do not fit the "
              "device. No tensor storage was published.";
        return result;
    }
    std::string detail;
    if (report_status != 0 || !report) {
        detail = "native refusal report unavailable (status " + std::to_string(report_status) + ")";
    } else if (report[kRecorded] != 1 || report[kRequest] != bytes) {
        detail = "no capacity transaction recorded for this request";
    } else {
        detail = std::string("termination=") + termination_name(report[kTermination]);
    }
    result.message = "DDO native allocation failed for " + std::to_string(bytes) + " bytes on device "
        + std::to_string(device) + " (" + detail + "); this is not a capacity refusal. "
          "No tensor storage was published; see the native diagnostics above.";
    return result;
}

// Calls the native allocator; a null result is never returned to the caller.
// `reject` must throw in the caller's exception domain.
template <class Allocate, class Reject, class... Args>
void* checked_allocation(Allocate&& allocate, Reject&& reject, Args&&... args) {
    void* pointer = std::forward<Allocate>(allocate)(std::forward<Args>(args)...);
    if (pointer) return pointer;
    std::forward<Reject>(reject)();
    std::terminate();  // A rejection that returns must not publish null storage.
}

}  // namespace ddo_vmm_bridge
