#pragma once
#include <atomic>

namespace ddo_vmm_detail {
// Presentation only; allocation, receipts and recovery do not depend on this flag.
inline std::atomic<bool> recovery_warnings_enabled{false};
inline bool show_recovery_warnings() noexcept {
    return recovery_warnings_enabled.load(std::memory_order_relaxed);
}
inline bool show_capacity_report(bool recovered) noexcept {
    return !recovered || show_recovery_warnings();
}
}
