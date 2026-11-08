// Budget-aware idle activation retention: actual pool with CPU Driver doubles
// and a configurable memory-observation double. Fixed policy stays the default;
// the budget policy keeps, reduces (dead units first, no waits) or falls back
// to the fixed limit when no fresh observation exists. No GPU library is loaded.
#include "auto_cpu.cpp"

using ddo_vmm_detail::memory_sampler;
constexpr uint64_t no_cap = 1ull << 30;

std::array<uint64_t, 7> retention() {
    std::array<uint64_t, 7> out{};
    check(ddo_exp_retention_stats_v1(out.data(), out.size()) == 0, "retention stats readable");
    return out;
}
// Fresh scenario: archived pool state, cold policy, next allocation samples.
void scenario(int policy, uint64_t headroom, uint64_t cap, uint64_t budget, uint64_t usage) {
    reset();
    check(ddo_exp_segmented_configure_retention(policy, headroom, cap) == 0, "cold retention configuration");
    enable();
    fixture_local_budget = budget; fixture_local_usage = usage; fixture_memory_query_ok = true;
    memory_sampler.next_sample_ns = 0;
}
void forget_observation() {
    std::lock_guard<std::mutex> lock(memory_sampler.peak_mutex);
    memory_sampler.any_sample = false;
}
void age_observation(uint64_t nanoseconds) {
    std::lock_guard<std::mutex> lock(memory_sampler.peak_mutex);
    memory_sampler.last_ns = ddo_vmm_detail::timing_now_ns() - nanoseconds;
}
// Allocate four units, optionally complete their consumer, then free them.
void four_units_freed(bool complete) {
    auto p = allocate(4 * fixture_page);
    check(p && physical() == 4 * fixture_page, "four units materialized");
    events_complete = complete;
    retire(p);
}

int main() {
    try {
        // Arguments and cold-only configuration.
        reset();
        uint64_t bad[6]{};
        check(ddo_exp_retention_stats_v1(bad, 6) == 1 && ddo_exp_retention_stats_v1(nullptr, 7) == 1, "stats argument checks");
        auto s0 = retention();
        check(s0[0] == 0 && s0[1] == 512ull << 20 && s0[2] == 4096ull << 20, "fixed policy and defaults");
        check(s0[3] == 0 && s0[4] == 0 && s0[5] == 0 && s0[6] == 0, "cold decision counters");
        check(ddo_exp_segmented_configure_retention(2, 0, 0) == 1 && ddo_exp_segmented_configure_retention(-1, 0, 0) == 1,
              "unknown policy refused");
        check(ddo_exp_segmented_configure_retention(1, (64ull << 30) + 1, 0) == 1
              && ddo_exp_segmented_configure_retention(1, 0, (64ull << 30) + 1) == 1, "bounds refused");
        check(retention()[0] == 0, "refused configuration changes nothing");
        enable();
        auto live = allocate(fixture_page);
        check(live && ddo_exp_segmented_configure_retention(1, 0, no_cap) == 1, "refused after first allocation");
        check(retention()[0] == 0, "policy unchanged after refused hot configuration");
        retire(live); closed();

        // Fixed policy (default) is unchanged: idle above the 0 MiB test limit closes; no decisions counted.
        scenario(0, 512ull << 20, 4096ull << 20, 8000, 7000);
        four_units_freed(true);
        check(backings.empty() && physical() == 0 && native_va.empty(), "fixed policy closes idle owner");
        auto s1 = retention();
        check(s1[3] == 0 && s1[4] == 0 && s1[5] == 0, "fixed policy records no budget decisions");
        closed();

        // Kept: within budget the idle backing stays and is reused without new Driver creation.
        scenario(1, 0, no_cap, 8000, 7000);
        const auto k0 = retention();
        four_units_freed(true);
        const auto queries_after_free = fixture_memory_queries;
        check(physical() == 4 * fixture_page && backings.size() == 1, "within budget: idle backing retained");
        const auto k1 = retention();
        check(k1[0] == 1 && k1[3] == k0[3] + 1 && k1[4] == k0[4] && k1[5] == k0[5], "kept decision counted");
        const auto creates = native_creates;
        auto again = allocate(4 * fixture_page);
        check(again && native_creates == creates && physical() == 4 * fixture_page, "retained units reused; no re-creation");
        check(fixture_memory_queries == queries_after_free, "free path never queries the OS (allocation sampler only)");
        retire(again); closed();

        // Reduced: over budget by 1000 bytes -> exactly one completed dead unit discarded; owner stays.
        // idle 4096, others 9000-4096=4904, allowed 8000-4904=3096, need 1000 -> 1 unit.
        scenario(1, 0, no_cap, 8000, 9000);
        const auto r0 = retention();
        const auto releases = native_releases;
        four_units_freed(true);
        check(physical() == 3 * fixture_page && backings.size() == 1 && native_releases == releases + 1,
              "discard one dead unit before any close");
        const auto r1 = retention();
        check(r1[4] == r0[4] + 1 && r1[6] == r0[6] + fixture_page && r1[3] == r0[3], "reduced decision and bytes");
        check(event_sync_calls == 0 && ctx_sync_calls == 0, "reduction never waits");
        closed();

        // Allowed zero: every dead unit discarded, then the owner without backing is closed.
        scenario(1, 0, no_cap, 8000, 20000);
        four_units_freed(true);
        check(physical() == 0 && backings.empty() && native_va.empty(), "all units discarded then owner closed");
        closed();

        // Headroom: target 8000-3000=5000, allowed 5000-2904=2096, need 2000 -> 2 units.
        scenario(1, 3000, no_cap, 8000, 7000);
        four_units_freed(true);
        check(physical() == 2 * fixture_page && backings.size() == 1, "headroom lowers the target");
        closed();

        // Cap: plenty of budget but a 2048-byte cap -> 2 units discarded.
        scenario(1, 0, 2 * fixture_page, 64000, 7000);
        four_units_freed(true);
        check(physical() == 2 * fixture_page, "cap bounds retention even with budget available");
        closed();

        // Pending consumer: over budget, nothing completed -> nothing released, no wait, owner kept.
        scenario(1, 0, no_cap, 8000, 20000);
        const auto p0 = retention();
        four_units_freed(false);
        check(physical() == 4 * fixture_page && event_sync_calls == 0 && ctx_sync_calls == 0,
              "pending units retained without waiting");
        const auto p1 = retention();
        check(p1[4] == p0[4] + 1 && p1[6] == p0[6], "reduction attempted, zero bytes released");
        closed();

        // Fallback: no observation -> fixed limit (closes), counted.
        scenario(1, 0, no_cap, 8000, 7000);
        auto f = allocate(4 * fixture_page);
        forget_observation();
        const auto f0 = retention();
        events_complete = true; retire(f);
        check(backings.empty() && physical() == 0, "no observation: fixed limit applies, never grows retention");
        check(retention()[5] == f0[5] + 1 && retention()[3] == f0[3], "fallback counted, not kept");
        closed();

        // Fallback: stale observation (older than 2 s).
        scenario(1, 0, no_cap, 8000, 7000);
        auto g = allocate(4 * fixture_page);
        age_observation(3ull * 1000 * 1000 * 1000);
        const auto g0 = retention();
        events_complete = true; retire(g);
        check(backings.empty() && retention()[5] == g0[5] + 1, "stale observation falls back to fixed");
        closed();

        // Fallback: failed query never records an observation.
        scenario(1, 0, no_cap, 8000, 7000);
        forget_observation(); fixture_memory_query_ok = false;
        auto h = allocate(4 * fixture_page);
        const auto h0 = retention();
        events_complete = true; retire(h);
        check(backings.empty() && retention()[5] == h0[5] + 1, "failed query falls back to fixed");
        fixture_memory_query_ok = true; closed();

        // Budget changes are followed: retained under a generous budget, reduced when it tightens.
        scenario(1, 0, no_cap, 8000, 7000);
        four_units_freed(true);
        check(physical() == 4 * fixture_page, "retained under generous budget");
        fixture_local_usage = 9000; memory_sampler.next_sample_ns = 0;
        auto one_unit = allocate(fixture_page);   // Samples the tighter budget, reuses one retained unit.
        check(one_unit && physical() == 4 * fixture_page, "reuse before reduction");
        retire(one_unit);
        check(physical() == 3 * fixture_page, "tighter budget reduces retained backing on next free");
        closed();

        reset();
        fixture_local_budget = 8000; fixture_local_usage = 7000;
        check(!GetModuleHandleW(L"nvcuda.dll") && !GetModuleHandleW(L"cudart64_13.dll"), "no GPU library loaded");
        std::cout << "PASS budget retention CPU contracts; checks=" << checks << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; reset(); return 1;
    }
}
