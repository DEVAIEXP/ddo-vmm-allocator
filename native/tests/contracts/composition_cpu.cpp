// Device-usage composition: actual pool with CPU Driver doubles, a memory
// observation double and an arena-composition double. Observation only: rows
// are taken with throttled memory samples, never under the pool lock, and
// totals that could not be observed are UINT64_MAX. No GPU library is loaded.
#include "auto_cpu.cpp"

using ddo_vmm_detail::composition_export_count;
using ddo_vmm_detail::memory_sampler;
constexpr uint64_t U = UINT64_MAX, pu = fixture_page;
using Row = std::array<uint64_t, ddo_vmm_detail::composition_fields>;

std::array<uint64_t, composition_export_count> composition() {
    std::array<uint64_t, composition_export_count> out{};
    check(ddo_exp_device_composition_v1(out.data(), out.size()) == 0, "composition export readable");
    return out;
}
Row last_row(const std::array<uint64_t, composition_export_count>& c) {
    Row r{}; std::copy(c.begin() + 3, c.begin() + 15, r.begin()); return r;
}
Row peak_row(const std::array<uint64_t, composition_export_count>& c) {
    Row r{}; std::copy(c.begin() + 15, c.begin() + 27, r.begin()); return r;
}
void expect_row(const char* label, const Row& got, const Row& exact) {
    if (got != exact) {
        std::cerr << label << ":"; for (auto v : got) std::cerr << ' ' << v; std::cerr << '\n';
        check(false, label);
    }
}
void* sampled_allocate(size_t bytes) { memory_sampler.next_sample_ns = 0; return allocate(bytes); }

int main() {
    try {
        check(composition_export_count == 31, "export size");
        uint64_t bad[30]{};
        check(ddo_exp_device_composition_v1(bad, 30) == 1 && ddo_exp_device_composition_v1(nullptr, 31) == 1,
              "argument checks");
        const auto cold = composition();
        check(cold[0] == 1 && cold[1] == 0 && cold[2] == 0, "cold schema and counts");
        for (size_t i = 3; i < cold.size(); ++i) check(cold[i] == U, "cold rows and maxima unavailable, never zero");

        // First sample happens before the allocation: pool still empty.
        reset(); enable();
        auto p = sampled_allocate(2 * pu); check(p != nullptr, "first allocation");
        auto c1 = composition();
        check(c1[1] == 1 && c1[2] == 1 && composition_calls == 1, "one complete sample");
        const Row first{8000, 7000, 1000, 3000, 1500, 0, 0, 0, 0, 1, 2, 0};
        expect_row("first last", last_row(c1), first);
        expect_row("first peak", peak_row(c1), first);
        check(c1[27] == 3000 && c1[28] == 1500 && c1[29] == 0 && c1[30] == 0, "first maxima");

        // Higher usage with a live lease: pool physical/live observed; peak follows usage.
        fixture_local_usage = 7500;
        auto q = sampled_allocate(pu); check(q != nullptr, "second allocation");
        auto c2 = composition();
        const Row second{8000, 7500, 1000, 3000, 1500, 0, 2 * pu, 2 * pu, 0, 1, 2, 0};
        expect_row("second last", last_row(c2), second);
        expect_row("second peak", peak_row(c2), second);
        check(c2[29] == 2 * pu && c2[30] == 2 * pu, "pool maxima");

        // Throttled: no new sample inside the period.
        auto r = allocate(pu); check(r != nullptr && composition()[1] == 2, "throttled sampler adds no row");

        // Pending retired leases are counted as pending, not live; lower usage keeps the peak.
        events_complete = false;
        retire(p); retire(q); retire(r);
        fixture_local_usage = 6000;
        auto s = sampled_allocate(pu); check(s != nullptr, "third sample allocation");
        auto c3 = composition();
        const Row third{8000, 6000, 1000, 3000, 1500, 0, 4 * pu, 0, 4 * pu, 1, 2, 0};
        expect_row("pending last", last_row(c3), third);
        expect_row("peak kept", peak_row(c3), second);

        // Busy arena: arena totals unavailable, row incomplete, peak and maxima unchanged.
        fixture_composition.busy_arenas = 1; fixture_local_usage = 9000;
        auto t = sampled_allocate(pu); check(t != nullptr, "busy-arena sample allocation");
        auto c4 = composition();
        check(c4[1] == 4 && c4[2] == 3, "incomplete sample counted separately");
        const auto busy = last_row(c4);
        check(busy[1] == 9000 && busy[3] == U && busy[4] == U && busy[5] == U && busy[9] == U && busy[10] == U
              && busy[6] != U && busy[11] == 2, "busy arena marks arena totals unavailable");
        expect_row("peak ignores incomplete", peak_row(c4), second);
        check(c4[27] == 3000 && c4[28] == 1500, "maxima ignore incomplete rows");

        // Registry busy.
        fixture_composition.busy_arenas = 0; fixture_composition.registry_observed = false;
        auto v = sampled_allocate(pu); check(v != nullptr, "registry-busy sample allocation");
        check(last_row(composition())[11] == 1 && last_row(composition())[3] == U, "registry busy flag");
        fixture_composition.registry_observed = true;

        check(!arena_lock_violation, "arena observation never under the pool lock");
        // Pool busy: another thread holds the pool lock; the sample never waits.
        // (The double's own lock probe sees that thread's lock; not a violation.)
        {
            std::atomic<bool> held{false}, done{false};
            std::thread holder([&] { std::scoped_lock lock(pool_mutex); held = true; while (!done) std::this_thread::yield(); });
            while (!held) std::this_thread::yield();
            sample_composition(0, {8000, 7000, 32000, 1000, 40000, 50000});
            done = true; holder.join();
            const auto pool_busy = last_row(composition());
            check(pool_busy[11] == 4 && pool_busy[6] == U && pool_busy[7] == U && pool_busy[8] == U
                  && pool_busy[3] == 3000, "pool busy marks pool totals unavailable");
            arena_lock_violation = false;
        }

        // Failed memory query: no sample, so no composition row.
        const auto before_failed = composition()[1];
        fixture_memory_query_ok = false;
        auto w = sampled_allocate(pu); check(w != nullptr, "allocation with failed query");
        check(composition()[1] == before_failed, "failed query adds no composition row");
        fixture_memory_query_ok = true;

        check(!arena_lock_violation, "arena observation never under the pool lock (after)");
        for (auto pointer : {s, t, v, w}) retire(pointer);
        closed();
        reset();
        fixture_local_budget = 8000; fixture_local_usage = 7000;
        check(!GetModuleHandleW(L"nvcuda.dll") && !GetModuleHandleW(L"cudart64_13.dll"), "no GPU library loaded");
        std::cout << "PASS device composition CPU contracts; checks=" << checks << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; reset(); return 1;
    }
}
