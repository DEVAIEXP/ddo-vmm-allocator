// Timing counters: actual pool with CPU Driver doubles. Counts must match the
// doubles exactly; nanoseconds are only required to be observed; unobserved
// memory fields stay UINT64_MAX. No GPU library is loaded.
#include "auto_cpu.cpp"

std::array<uint64_t, ddo_vmm_detail::timing_export_count> timing() {
    std::array<uint64_t, ddo_vmm_detail::timing_export_count> out{};
    check(ddo_exp_timing_counters_v1(out.data(), out.size()) == 0, "timing export readable");
    return out;
}
uint64_t calls(const std::array<uint64_t, ddo_vmm_detail::timing_export_count>& t, ddo_vmm_detail::TimingOp op) {
    return t[2 + static_cast<size_t>(op) * 3];
}
uint64_t bytes(const std::array<uint64_t, ddo_vmm_detail::timing_export_count>& t, ddo_vmm_detail::TimingOp op) {
    return t[2 + static_cast<size_t>(op) * 3 + 2];
}
constexpr size_t memory_first = 2 + ddo_vmm_detail::timing_op_count * 3;

int main() {
    try {
        using ddo_vmm_detail::TimingOp;
        check(ddo_vmm_detail::timing_export_count == 55, "export size");
        uint64_t bad[54]{};
        check(ddo_exp_timing_counters_v1(bad, 54) == 1 && ddo_exp_timing_counters_v1(nullptr, 55) == 1, "argument checks");
        auto cold = timing();
        check(cold[0] == 1 && cold[1] == 1, "schema and Windows memory support flag");
        for (size_t i = 2; i < memory_first; ++i) check(cold[i] == 0, "cold counters are zero (observed none)");
        for (size_t i = memory_first; i < cold.size(); ++i) check(cold[i] == UINT64_MAX, "unobserved memory stays unavailable");

        // Plain allocation: Driver calls counted exactly as the doubles see them.
        reset(); enable();
        auto p = allocate(4 * fixture_page);
        retire(p);
        auto a = timing();
        check(calls(a, TimingOp::segment_create) == native_creates && native_creates == 4, "creates counted");
        check(calls(a, TimingOp::segment_map) == 4 && calls(a, TimingOp::segment_access) == native_accesses, "maps/access counted");
        check(bytes(a, TimingOp::segment_create) == 4 * fixture_page, "create bytes");
        check(calls(a, TimingOp::pool_malloc) == 1 && bytes(a, TimingOp::pool_malloc) == 4 * fixture_page, "pool malloc counted");
        check(calls(a, TimingOp::pool_free) == 1 && calls(a, TimingOp::transaction) == 0, "free counted; no transaction");
        closed();
        auto b = timing();
        check(calls(b, TimingOp::segment_unmap) == native_unmaps && calls(b, TimingOp::segment_release) == native_releases
              && native_releases == 4, "close teardown counted");

        // NOT_READY continuation is counted as a second access call.
        reset(); enable();
        const auto before_nr = calls(timing(), TimingOp::segment_access);
        native_fault = "access"; native_error = CUDA_ERROR_NOT_READY; native_fault_remaining = 1;
        auto q = allocate(fixture_page);
        check(q && calls(timing(), TimingOp::segment_access) == before_nr + 2 && native_accesses == 2,
              "access and continuation both counted");
        retire(q); closed();

        // Satisfied transaction: transaction, tiers, retry and first-refusal memory.
        reset(); enable(); Split split; retire(split.middle); events_complete = true;
        const auto c0 = timing();
        auto creates = std::make_shared<size_t>(0);
        native_observer = [creates](const char* op) {
            if (std::string(op) == "create" && ++*creates == 1) { native_fault = "create"; native_error = CUDA_ERROR_OUT_OF_MEMORY; native_fault_remaining = 1; }
        };
        const auto queries0 = fixture_memory_queries;
        const auto samples0 = ddo_vmm_detail::memory_sampler.samples.load();
        auto t = allocate(3 * fixture_page);
        check(t != nullptr, "transaction satisfied");
        const auto c1 = timing();
        check(calls(c1, TimingOp::transaction) == calls(c0, TimingOp::transaction) + 1, "one transaction");
        check(calls(c1, TimingOp::transaction_tier1) == calls(c0, TimingOp::transaction_tier1) + 1
              && calls(c1, TimingOp::transaction_tier2) == calls(c0, TimingOp::transaction_tier2) + 1
              && calls(c1, TimingOp::transaction_tier3) == calls(c0, TimingOp::transaction_tier3) + 1
              && calls(c1, TimingOp::transaction_retry) == calls(c0, TimingOp::transaction_retry) + 1, "tiers and retry");
        check(calls(c1, TimingOp::pool_malloc) == calls(c0, TimingOp::pool_malloc) + 1, "retry is not a second pool request");
        const auto sampled = ddo_vmm_detail::memory_sampler.samples.load() - samples0;
        check(fixture_memory_queries == queries0 + 1 + sampled,
              "one first-refusal query plus throttled samples, all outside the pool lock");
        check(c1[memory_first] == 2 && c1[memory_first + 1] == 8000 && c1[memory_first + 6] == 50000, "first refusal block");
        check(c1[memory_first + 7] == UINT64_MAX, "no unsatisfied transaction yet");
        native_observer = {}; retire(t); split.finish(); closed();

        // Unsatisfied transaction records the last-unsatisfied block; first is kept.
        reset(); enable();
        native_fault = "create"; native_error = CUDA_ERROR_OUT_OF_MEMORY; native_fault_remaining = SIZE_MAX;
        check(!allocate(2 * fixture_page), "no-release refusal");
        const auto c2 = timing();
        check(c2[memory_first + 7] == term_no_release && c2[memory_first + 8] == 8000, "last-unsatisfied block");
        check(c2[memory_first] == 2, "first-refusal block unchanged");
        native_fault.clear(); closed();

        // A failed memory query leaves the block unavailable rather than zero.
        reset(); enable(); fixture_memory_query_ok = false;
        native_fault = "create"; native_error = CUDA_ERROR_OUT_OF_MEMORY; native_fault_remaining = SIZE_MAX;
        check(!allocate(2 * fixture_page), "second no-release refusal");
        const auto c3 = timing();
        check(c3[memory_first + 7] == term_no_release, "previous observation retained after failed query");
        fixture_memory_query_ok = true; native_fault.clear(); closed();

        // Throttled sampler: one claim per 250 ms period, exact over-budget counts and peaks.
        {
            ddo_vmm_detail::MemorySampler s;
            const uint64_t t0 = 10'000'000'000ull, period = ddo_vmm_detail::memory_sample_period_ns;
            check(s.claim(t0) && !s.claim(t0) && !s.claim(t0 + period - 1) && s.claim(t0 + period),
                  "sampler claims at most once per period");
            s.record({8000, 7000, 32000, 1000, 1, 2});   // Within budget.
            s.record({8000, 9500, 32000, 4000, 1, 2});   // 1500 over budget.
            s.record({8000, 8200, 32000, 2000, 1, 2});   // 200 over budget.
            check(s.samples == 3 && s.over_budget_samples == 2, "over-budget samples counted");
            check(s.max_local_usage == 9500 && s.max_nonlocal_usage == 4000 && s.max_local_over_budget == 1500,
                  "peaks recorded");
        }
        std::array<uint64_t, ddo_vmm_detail::memory_export_count> mem{};
        check(ddo_exp_memory_observation_v1(0, mem.data(), 12) == 1 && ddo_exp_memory_observation_v1(-1, mem.data(), 13) == 1,
              "memory export argument checks");
        check(ddo_exp_memory_observation_v1(0, mem.data(), mem.size()) == 0, "memory export readable");
        check(mem[0] == 1 && mem[1] == 8000 && mem[2] == 7000 && mem[6] == 50000 && mem[12] == 250,
              "current query and period in export order");
        check(mem[7] >= 1 && mem[8] == 0 && mem[9] == 7000, "sampler state exported (stub never over budget)");
        fixture_memory_query_ok = false;
        check(ddo_exp_memory_observation_v1(0, mem.data(), mem.size()) == 0 && mem[1] == UINT64_MAX && mem[6] == UINT64_MAX,
              "failed current query is unavailable, never zero");
        fixture_memory_query_ok = true;

        // Source-read counter is shared with ddo_vmm.cpp (inline global).
        ddo_vmm_detail::timing_counters.add(TimingOp::source_read, 1000, 2 * fixture_page);
        check(calls(timing(), TimingOp::source_read) == 1 && bytes(timing(), TimingOp::source_read) == 2 * fixture_page,
              "source read counter");
        reset();
        check(!GetModuleHandleW(L"nvcuda.dll") && !GetModuleHandleW(L"cudart64_13.dll"), "no GPU library loaded");
        std::cout << "PASS timing counter CPU contracts; checks=" << checks << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; reset(); return 1;
    }
}
