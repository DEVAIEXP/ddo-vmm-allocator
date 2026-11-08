#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>

// Observation-only timing counters (experimental candidate feature
// DDO_VMM_TIMING_COUNTERS). Process lifetime, relaxed atomics, no CUDA calls,
// no locks, no reset, no policy. Host wall time only: values do not include
// GPU completion. A field never observed stays UINT64_MAX in the export.
namespace ddo_vmm_detail {
enum class TimingOp : size_t {
    source_read,         // Positioned weight-source file read (one per chunk).
    segment_create,      // Segmented activation backing Driver calls.
    segment_map,
    segment_access,
    segment_unmap,
    segment_release,
    pool_malloc,         // Whole ddo_vmm_pool_malloc request.
    pool_free,           // Whole ddo_vmm_pool_free request.
    transaction,         // Whole capacity transaction including its retry.
    transaction_tier1,
    transaction_tier2,
    transaction_tier3,
    transaction_retry,
    count
};
constexpr size_t timing_op_count = static_cast<size_t>(TimingOp::count);
// Fields per operation: calls, wall nanoseconds, bytes (0 when not applicable).
constexpr size_t timing_fields = 3;
// Windows memory observations at two moments: the first capacity refusal and
// the last transaction that did not end satisfied. Order:
// local budget, local usage, non-local budget, non-local usage,
// available physical, available commit.
constexpr size_t memory_fields = 6;
constexpr size_t timing_export_count = 2 + timing_op_count * timing_fields + 2 * (1 + memory_fields);

struct TimingCounters {
    std::array<std::array<std::atomic<uint64_t>, timing_fields>, timing_op_count> ops{};
    // Guards the memory observation blocks only. Taken briefly when a refusal
    // is recorded and on export; never held while any other lock is taken.
    std::mutex memory_mutex;
    std::array<uint64_t, memory_fields> first_refusal{};
    std::array<uint64_t, memory_fields> last_unsatisfied{};
    bool first_refusal_observed = false, last_unsatisfied_observed = false;
    uint64_t first_refusal_status = UINT64_MAX, last_unsatisfied_status = UINT64_MAX;

    void add(TimingOp op, uint64_t nanoseconds, uint64_t bytes = 0) noexcept {
        auto& row = ops[static_cast<size_t>(op)];
        row[0].fetch_add(1, std::memory_order_relaxed);
        row[1].fetch_add(nanoseconds, std::memory_order_relaxed);
        if (bytes) row[2].fetch_add(bytes, std::memory_order_relaxed);
    }
};
inline TimingCounters timing_counters;

inline uint64_t timing_now_ns() noexcept {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

// RAII scope; measures the enclosing statement block on the calling thread.
class TimingScope {
    TimingOp op_;
    uint64_t start_, bytes_;
public:
    explicit TimingScope(TimingOp op, uint64_t bytes = 0) noexcept
        : op_(op), start_(timing_now_ns()), bytes_(bytes) {}
    ~TimingScope() { timing_counters.add(op_, timing_now_ns() - start_, bytes_); }
    TimingScope(const TimingScope&) = delete;
    TimingScope& operator=(const TimingScope&) = delete;
};

template<class Call> auto timed_call(TimingOp op, Call&& call, uint64_t bytes = 0) {
    TimingScope scope(op, bytes);
    return call();
}

// Record one memory observation. `unsatisfied` selects the last-unsatisfied
// block (overwritten each time); otherwise only the first refusal is kept.
// Callers must not hold the pool lock (keeps lock order trivial).
inline void timing_record_memory(bool unsatisfied, uint64_t status,
                                 const std::array<uint64_t, memory_fields>& values) {
    std::lock_guard<std::mutex> lock(timing_counters.memory_mutex);
    if (unsatisfied) {
        timing_counters.last_unsatisfied = values;
        timing_counters.last_unsatisfied_status = status;
        timing_counters.last_unsatisfied_observed = true;
    } else if (!timing_counters.first_refusal_observed) {
        timing_counters.first_refusal = values;
        timing_counters.first_refusal_status = status;
        timing_counters.first_refusal_observed = true;
    }
}
inline bool timing_first_refusal_recorded() {
    std::lock_guard<std::mutex> lock(timing_counters.memory_mutex);
    return timing_counters.first_refusal_observed;
}

// Throttled memory sampler (observation only). At most one sample per period,
// taken by whichever allocation thread wins the claim, with no pool lock held.
// Tracks peaks and over-budget samples so demotion pressure inside a step is
// visible, not only at refusals. Export order (13 fields):
// 0 supported, 1-6 current query (local budget, local usage, non-local budget,
// non-local usage, available physical, available commit; UINT64_MAX when the
// query fails), 7 samples, 8 samples with local usage > local budget,
// 9 max local usage, 10 max non-local usage, 11 max local bytes over budget,
// 12 sample period in milliseconds.
constexpr uint64_t memory_sample_period_ns = 250ull * 1000 * 1000;
constexpr size_t memory_export_count = 13;
struct MemorySampler {
    std::atomic<uint64_t> next_sample_ns{0};
    std::atomic<uint64_t> samples{0}, over_budget_samples{0};
    std::mutex peak_mutex;
    uint64_t max_local_usage = 0, max_nonlocal_usage = 0, max_local_over_budget = 0;
    bool any_sample = false;
    // Latest successful observation, for budget-aware retention decisions.
    uint64_t last_budget = 0, last_usage = 0, last_ns = 0;

    // Copy the latest observation if it is at most `max_age_ns` old. May be
    // called with the pool lock held (budget retention); peak_mutex is a leaf
    // lock: nothing else is ever acquired while it is held. No OS query here.
    bool latest(uint64_t now, uint64_t max_age_ns, uint64_t& budget, uint64_t& usage) {
        std::lock_guard<std::mutex> lock(peak_mutex);
        if (!any_sample || now < last_ns || now - last_ns > max_age_ns) return false;
        budget = last_budget; usage = last_usage; return true;
    }

    // Returns true if the caller won this period and should take a sample.
    bool claim(uint64_t now) noexcept {
        uint64_t next = next_sample_ns.load(std::memory_order_relaxed);
        if (now < next) return false;
        return next_sample_ns.compare_exchange_strong(next, now + memory_sample_period_ns,
                                                      std::memory_order_relaxed);
    }
    void record(const std::array<uint64_t, memory_fields>& v, uint64_t now = 0) {
        samples.fetch_add(1, std::memory_order_relaxed);
        const uint64_t budget = v[0], usage = v[1];
        if (usage > budget) over_budget_samples.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(peak_mutex);
        any_sample = true;
        last_budget = budget; last_usage = usage; last_ns = now;
        if (usage > max_local_usage) max_local_usage = usage;
        if (v[3] != UINT64_MAX && v[3] > max_nonlocal_usage) max_nonlocal_usage = v[3];
        if (usage > budget && usage - budget > max_local_over_budget) max_local_over_budget = usage - budget;
    }
};
inline MemorySampler memory_sampler;

// Device-usage composition (observation only), taken with each successful
// throttled memory sample, never under the pool lock. Row order (12):
// 0 local budget, 1 local usage, 2 non-local usage, 3 model-arena resident,
// 4 execution-slot resident, 5 pressure-region resident, 6 pool physical,
// 7 pool live, 8 pool pending, 9 model arenas, 10 slot arenas, 11 flags
// (bit 0 arena registry busy, bit 1 an arena busy, bit 2 pool busy). Totals
// that could not be fully observed are UINT64_MAX. Export (31): schema,
// composition samples, complete samples, last row, row at the highest local
// usage among complete samples, then maxima of model-arena, slot, pool
// physical and pool live bytes over complete samples.
constexpr size_t composition_fields = 12;
constexpr size_t composition_export_count = 3 + 2 * composition_fields + 4;
struct CompositionSampler {
    std::mutex mutex;
    uint64_t samples = 0, complete = 0;
    std::array<uint64_t, composition_fields> last{}, at_peak{};
    bool any = false, any_complete = false;
    uint64_t max_model = 0, max_slot = 0, max_pool_physical = 0, max_pool_live = 0;

    void record(const std::array<uint64_t, composition_fields>& row) {
        std::lock_guard<std::mutex> lock(mutex);
        ++samples; any = true; last = row;
        if (row[11] != 0) return;
        ++complete;
        if (!any_complete || row[1] > at_peak[1]) at_peak = row;
        any_complete = true;
        if (row[3] > max_model) max_model = row[3];
        if (row[4] > max_slot) max_slot = row[4];
        if (row[6] > max_pool_physical) max_pool_physical = row[6];
        if (row[7] > max_pool_live) max_pool_live = row[7];
    }
    void snapshot(uint64_t* out) {
        constexpr uint64_t unavailable = UINT64_MAX;
        std::lock_guard<std::mutex> lock(mutex);
        size_t i = 0;
        out[i++] = 1; out[i++] = samples; out[i++] = complete;
        for (size_t k = 0; k < composition_fields; ++k) out[i++] = any ? last[k] : unavailable;
        for (size_t k = 0; k < composition_fields; ++k) out[i++] = any_complete ? at_peak[k] : unavailable;
        for (uint64_t v : {max_model, max_slot, max_pool_physical, max_pool_live}) out[i++] = any_complete ? v : unavailable;
    }
};
inline CompositionSampler composition_sampler;

// `current` holds the caller's fresh query, or nullptr when unavailable.
inline void memory_snapshot(uint64_t* values, bool supported,
                            const std::array<uint64_t, memory_fields>* current) {
    constexpr uint64_t unavailable = UINT64_MAX;
    values[0] = supported ? 1 : 0;
    for (size_t i = 0; i < memory_fields; ++i) values[1 + i] = current ? (*current)[i] : unavailable;
    values[7] = memory_sampler.samples.load(std::memory_order_relaxed);
    values[8] = memory_sampler.over_budget_samples.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(memory_sampler.peak_mutex);
    values[9] = memory_sampler.any_sample ? memory_sampler.max_local_usage : unavailable;
    values[10] = memory_sampler.any_sample ? memory_sampler.max_nonlocal_usage : unavailable;
    values[11] = memory_sampler.any_sample ? memory_sampler.max_local_over_budget : unavailable;
    values[12] = memory_sample_period_ns / 1000000;
}

// Fill `values` in export order. `windows` reports whether memory
// observations are supported on this platform; when not, they stay unavailable.
inline void timing_snapshot(uint64_t* values, bool windows) {
    constexpr uint64_t unavailable = UINT64_MAX;
    size_t index = 0;
    values[index++] = 1;                    // Schema.
    values[index++] = windows ? 1 : 0;      // Memory observations supported.
    for (size_t op = 0; op < timing_op_count; ++op)
        for (size_t field = 0; field < timing_fields; ++field)
            values[index++] = timing_counters.ops[op][field].load(std::memory_order_relaxed);
    auto block = [&](bool observed, uint64_t status, const std::array<uint64_t, memory_fields>& data) {
        values[index++] = observed ? status : unavailable;
        for (size_t i = 0; i < memory_fields; ++i) values[index++] = observed ? data[i] : unavailable;
    };
    std::lock_guard<std::mutex> lock(timing_counters.memory_mutex);
    block(timing_counters.first_refusal_observed, timing_counters.first_refusal_status, timing_counters.first_refusal);
    block(timing_counters.last_unsatisfied_observed, timing_counters.last_unsatisfied_status, timing_counters.last_unsatisfied);
}
}
