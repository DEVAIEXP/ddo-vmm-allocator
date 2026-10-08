#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ddo_vmm_detail {
// Process-lifetime CPU counters. No CUDA calls, policy, locks or reset.
class AllocationDiagnostics {
public:
    static constexpr size_t operation_count = 9;
    static constexpr size_t snapshot_count = 26;
    static constexpr size_t snapshot_v2_count = 36;
    enum Operation { create, map, access, unmap, release, reserve, address_free,
                     access_retry_sync, fault_range };
    std::atomic<bool> enabled{false};

    bool start() noexcept {
        bool expected = false;
        return enabled.compare_exchange_strong(expected, true);
    }
    template<class Call> auto measure(Operation op, Call&& call) {
        if (!enabled.load(std::memory_order_relaxed)) return call();
        const auto begin = std::chrono::steady_clock::now();
        const auto result = call();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count();
        counters[op][0].fetch_add(1, std::memory_order_relaxed);
        counters[op][1].fetch_add(static_cast<uint64_t>(ns), std::memory_order_relaxed);
        if (result != 0) counters[op][2].fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    // RAII measures the enclosing fault call, including locks and nested calls.
    // Its failure field is unavailable: final fault failures have existing metrics.
    class FaultScope {
    public:
        explicit FaultScope(AllocationDiagnostics& owner) : owner_(owner),
            active_(owner.enabled.load(std::memory_order_relaxed)) {
            if (active_) begin_ = std::chrono::steady_clock::now();
        }
        FaultScope(const FaultScope&) = delete;
        FaultScope& operator=(const FaultScope&) = delete;
        ~FaultScope() {
            if (!active_) return;
            owner_.counters[fault_range][0].fetch_add(1, std::memory_order_relaxed);
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - begin_).count();
            owner_.counters[fault_range][1].fetch_add(static_cast<uint64_t>(ns), std::memory_order_relaxed);
        }
    private:
        AllocationDiagnostics& owner_;
        bool active_;
        std::chrono::steady_clock::time_point begin_;
    };
    void access_result(int result, int not_ready, int out_of_memory) noexcept {
        if (!enabled.load(std::memory_order_relaxed)) return;
        const size_t category = result == 0 ? 0 : result == not_ready ? 1
            : result == out_of_memory ? 2 : 3;
        access_results[category].fetch_add(1, std::memory_order_relaxed);
    }
    void pool_reuse() noexcept {
        if (enabled.load(std::memory_order_relaxed)) reuse.fetch_add(1, std::memory_order_relaxed);
    }
    void pool_backing(uint64_t mapped) noexcept {
        if (!enabled.load(std::memory_order_relaxed)) return;
        backing.fetch_add(1, std::memory_order_relaxed);
        auto previous = peak.load(std::memory_order_relaxed);
        while (previous < mapped && !peak.compare_exchange_weak(previous, mapped)) {}
    }
    int snapshot(uint64_t* values, size_t count) const noexcept {
        if (!values || count != snapshot_count) return 1;
        std::array<uint64_t, snapshot_count> result{};
        result[0] = 1;
        result[1] = enabled.load(std::memory_order_relaxed);
        for (size_t op = 0; op < 7; ++op)
            for (size_t field = 0; field < 3; ++field)
                result[2 + op * 3 + field] = counters[op][field].load(std::memory_order_relaxed);
        result[23] = reuse.load(std::memory_order_relaxed);
        result[24] = backing.load(std::memory_order_relaxed);
        result[25] = peak.load(std::memory_order_relaxed);
        for (size_t i = 0; i < count; ++i) values[i] = result[i];
        return 0;
    }
    int snapshot_v2(uint64_t* values, size_t count) const noexcept {
        if (!values || count != snapshot_v2_count) return 1;
        std::array<uint64_t, snapshot_v2_count> result{};
        snapshot(result.data(), snapshot_count);
        result[0] = 2;
        for (size_t op = 7; op < operation_count; ++op)
            for (size_t field = 0; field < 3; ++field)
                result[26 + (op - 7) * 3 + field] = counters[op][field].load(std::memory_order_relaxed);
        for (size_t i = 0; i < 4; ++i)
            result[32 + i] = access_results[i].load(std::memory_order_relaxed);
        for (size_t i = 0; i < count; ++i) values[i] = result[i];
        return 0;
    }
private:
    std::array<std::array<std::atomic<uint64_t>, 3>, operation_count> counters{};
    std::atomic<uint64_t> reuse{0}, backing{0}, peak{0};
    std::array<std::atomic<uint64_t>, 4> access_results{};
};

#ifdef DDO_VMM_ALLOCATION_DIAGNOSTICS
extern AllocationDiagnostics allocation_diagnostics;
#endif
}
