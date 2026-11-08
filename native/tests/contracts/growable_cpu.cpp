// Growable shared segmented owner: actual pool with CPU Driver doubles.
// One shared owner per (device, context) is tried before request-sized
// owners; the free path trims its completed dead units above the fixed idle
// limit with the bounded reclaim (no waits); explicit trim closes it.
// Disabled by default. No GPU library is loaded.
#include "auto_cpu.cpp"

constexpr uint64_t pu = fixture_page;
std::array<uint64_t, 8> growable() {
    std::array<uint64_t, 8> out{};
    check(ddo_exp_growable_stats_v1(out.data(), out.size()) == 0, "growable stats readable");
    return out;
}
void limit_mib(const char* value) { check(_putenv_s("DDO_VMM_TORCH_CACHE_MIB", value) == 0, "idle limit environment"); }
void configure(uint64_t reserve) {
    reset();
    check(ddo_exp_segmented_configure_growable(reserve) == 0, "cold growable configuration");
    enable();
}
std::uintptr_t at(void* p) { return reinterpret_cast<std::uintptr_t>(p); }

int main() {
    try {
        // Arguments and cold-only configuration.
        reset();
        uint64_t bad[7]{};
        check(ddo_exp_growable_stats_v1(bad, 7) == 1 && ddo_exp_growable_stats_v1(nullptr, 8) == 1, "stats argument checks");
        check(growable()[0] == 0 && growable()[1] == 0 && growable()[2] == 0, "disabled by default");
        check(ddo_exp_segmented_configure_growable((64ull << 30) + 1) == 1, "reserve bound refused");
        enable();
        auto live = allocate(pu);
        check(live && ddo_exp_segmented_configure_growable(16 * pu) == 1, "refused after first allocation");
        check(growable()[0] == 0, "reserve unchanged after refused configuration");
        retire(live); closed();

        // Disabled: request-sized owners exactly as before.
        limit_mib("1");  // 1 MiB = 1024 fixture units: nothing trimmed by size.
        configure(0);
        auto d1 = allocate(2 * pu), d2 = allocate(3 * pu);
        check(backings.size() == 2 && native_va.size() == 2 && growable()[1] == 0 && growable()[3] == 0,
              "disabled: one owner per request, no shared owner, no fallback count");
        retire(d1); retire(d2); closed();

        // Enabled: first request creates the shared owner; later requests share it.
        configure(16 * pu);
        auto a = allocate(2 * pu); check(a != nullptr, "first request");
        check(backings.size() == 1 && backings[0]->shared && native_va.size() == 1
              && native_va.begin()->second == 16 * pu && physical() == 2 * pu, "shared owner: 16-unit VA, 2 units physical");
        auto b = allocate(3 * pu);
        check(b && backings.size() == 1 && at(b) == at(a) + 2 * pu && physical() == 5 * pu, "second request packs into the shared owner");
        auto g = growable();
        check(g[0] == 16 * pu && g[1] == 1 && g[2] == 1 && g[3] == 0 && g[6] == 5 * pu && g[7] == 5 * pu, "stats after packing");

        // Freed spans coalesce and are reused for a different size without Driver creation.
        events_complete = true;
        retire(a); retire(b);
        check(backings.size() == 1 && physical() == 5 * pu, "owner-level trim keeps the shared owner; under the idle limit");
        const auto creates = native_creates;
        auto c = allocate(5 * pu);
        check(c && at(c) == at(a) && native_creates == creates && backings.size() == 1,
              "coalesced 2+3 units serve a 5-unit request with no new unit or owner");

        // Same-stream reuse of a pending span: no wait, no creation.
        events_complete = false;
        retire(c);
        const auto syncs = event_sync_calls + ctx_sync_calls;
        auto e = allocate(4 * pu);
        check(e && at(e) == at(a) && native_creates == creates && event_sync_calls + ctx_sync_calls == syncs,
              "pending span reused on the same stream without waiting or creating");

        // A request that does not fit the shared owner contiguously falls back to a request-sized owner.
        auto f = allocate(12 * pu);
        check(f && backings.size() == 1 && physical() == 16 * pu, "tail fit inside the shared owner (4 + 12 units)");
        auto h = allocate(2 * pu);
        check(h && backings.size() == 2 && !backings[1]->shared && growable()[3] == 1, "no contiguous fit: request-sized fallback counted");
        auto big = allocate(20 * pu);
        check(big && backings.size() == 3 && growable()[3] == 2 && growable()[1] == 1, "request above the reserve: fallback, no second shared owner");
        events_complete = true;
        retire(e); retire(f); retire(h); retire(big);
        check(backings.size() == 3, "under the idle limit all owners are retained as before; shared owner kept");
        closed();
        check(backings.empty() && native_va.empty(), "explicit trim closes the shared owner");

        // Unit-level trim: idle physical above a 0 limit is released as completed dead units, bounded per free.
        limit_mib("0");
        configure(16 * pu);
        auto keep = allocate(pu), drop = allocate(6 * pu);
        check(backings.size() == 1 && physical() == 7 * pu, "shared owner with 7 units");
        events_complete = true;
        const auto releases = native_releases;
        retire(drop);
        g = growable();
        check(physical() == pu && native_releases == releases + 6 && backings.size() == 1
              && g[4] >= 1 && g[5] == 6, "dead completed units discarded down to the live unit; owner kept");
        // Pending units are never released by the trim and never waited on.
        auto pend = allocate(3 * pu); check(pend != nullptr && physical() == 4 * pu, "rematerialized 3 units");
        events_complete = false;
        const auto syncs2 = event_sync_calls + ctx_sync_calls;
        retire(pend);
        check(physical() == 4 * pu && event_sync_calls + ctx_sync_calls == syncs2, "pending units kept; no wait");
        // Once completed, the next free observes completion and trims.
        events_complete = true;
        retire(keep);
        check(physical() <= pu, "completed units trimmed on the next free (bounded pass)");
        closed();

        // Bounded work: one pass releases at most 16 units.
        limit_mib("0");
        configure(64 * pu);
        auto wide = allocate(40 * pu); auto pin = allocate(pu);
        events_complete = true;
        const auto before_wide = native_releases;
        retire(wide);
        check(native_releases - before_wide == 16 && physical() == 41 * pu - 16 * pu, "one pass releases at most 16 units");
        retire(pin);
        check(native_releases - before_wide == 32, "next free continues the bounded reclaim");
        closed();

        limit_mib("0");
        reset();
        check(!GetModuleHandleW(L"nvcuda.dll") && !GetModuleHandleW(L"cudart64_13.dll"), "no GPU library loaded");
        std::cout << "PASS growable shared owner CPU contracts; checks=" << checks << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; reset(); return 1;
    }
}
