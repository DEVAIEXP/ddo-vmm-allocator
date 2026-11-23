// Segmented capacity transaction: actual pool, CPU Driver doubles only.
#include "auto_cpu.cpp"

// Export field indexes (ddo_exp_segmented_last_refusal order).
enum Field : size_t {
    f_available, f_request, f_deficit, f_owner_new, f_operation, f_first, f_barrier, f_continuation,
    f_rb_fresh, f_rb_released, f_rb_retained, f_t1_pool_busy, f_t1_owners, f_t1_owners_busy,
    f_t1_scanned, f_t1_released, f_t1_pending, f_t1_live, f_t1_quarantined, f_t1_discarded,
    f_t1_excluded, f_t2_evictable, f_t2_busy, f_t2_evictions, f_t2_bytes, f_t2_status, f_t3_status,
    f_t3_fences, f_t3_released, f_retry, f_termination,
};
constexpr uint64_t none = UINT64_MAX;
constexpr int oom = static_cast<int>(CUDA_ERROR_OUT_OF_MEMORY);

// Fail the nth call (1-based) of one Driver operation once with status.
void fail_nth(const char* operation, size_t nth, int status) {
    auto count = std::make_shared<size_t>(0);
    std::string name = operation;
    native_observer = [=](const char* current) {
        if (name == current && ++*count == nth) {
            native_fault = name; native_error = status; native_fault_remaining = 1;
        }
    };
}
size_t unit_count(const std::shared_ptr<Backing>& owner, SegmentedUnitKind kind) {
    size_t total = 0;
    for (size_t i = 0; i < owner->segmented->page_count(); ++i) total += owner->segmented->unit_kind(i) == kind;
    return total;
}
void reap_now(const std::shared_ptr<Backing>& owner) {
    std::scoped_lock lock(pool_mutex); check(!reap_segmented(*owner), "fixture reap");
}
void no_waits() { check(!event_sync_calls && !ctx_sync_calls, "no synchronization in Tiers 1-2"); }
void clean_rollback_owner(const std::shared_ptr<Backing>& owner) {
    check(!owner->retiring && !owner->segmented->error() && !owner->segmented_error,
          "owner healthy after clean rollback");
    const auto spans = owner->ledger.snapshot();
    check(!spans.reserved_count && !spans.leased && !spans.quarantined, "lease cancelled, nothing published");
}

int main(int argc, char** argv) {
    try {
        if (argc > 1) {
            check(argc == 2 && std::string(argv[1]) == "--show-recovery-warns", "known CPU presentation option");
            recovery_warnings_enabled.store(true, std::memory_order_relaxed);
        }
        // C2: OOM on create/map/access of unit k rolls back this call's fresh units.
        struct Case { const char* op; size_t nth; uint64_t operation; size_t unmaps, releases; };
        for (const Case c : {Case{"create", 3, 1, 2, 2}, Case{"map", 2, 2, 1, 2}, Case{"access", 1, 3, 1, 1}}) {
            reset(); enable(); fail_nth(c.op, c.nth, oom);
            check(!allocate(4 * fixture_page), "capacity refusal returns null");
            check(native_physical.empty() && native_mapped.empty() && native_va.size() == 1,
                  "fresh handles/mappings released; VA owner retained");
            check(native_unmaps == c.unmaps && native_releases == c.releases, "exact rollback Driver calls");
            check(backings.size() == 1 && !backings[0]->segmented->physical_charge(), "pre-claim charge restored");
            clean_rollback_owner(backings[0]);
            const auto r = refusal_record();
            check(r[f_available] == 1 && r[f_request] == 4 * fixture_page && r[f_deficit] == 4 * fixture_page
                  && r[f_owner_new] == 1 && r[f_operation] == c.operation && r[f_first] == 2, "refusal identity");
            check(r[f_barrier] == none && r[f_continuation] == none, "unobserved access statuses stay unavailable");
            check(r[f_rb_fresh] == c.nth && r[f_rb_released] == c.nth && r[f_rb_retained] == 0, "rollback counts");
            check(r[f_t1_pool_busy] == 0 && r[f_t1_owners] == 1 && r[f_t1_excluded] == 4 && r[f_t1_released] == 0,
                  "requester range excluded");
            check(r[f_t2_evictable] == 0 && r[f_t2_evictions] == 0 && r[f_t3_status] == none && r[f_t3_fences] == none
                  && r[f_retry] == none && r[f_termination] == term_no_release, "no release means no retry");
            check(native_accesses <= 4 && native_creates == (c.op == std::string("create") ? 3u : c.nth), "one attempt");
            no_waits(); native_observer = {}; closed();
        }
        std::cout << "PASS clean rollback at create, map and access\n";

        // C2 failure: unmap failure during rollback retains ownership; no capacity work.
        reset(); enable(); fail_nth("access", 2, oom);
        native_access_failure = [] { native_fault = "unmap"; native_error = CUDA_ERROR_INVALID_CONTEXT; native_fault_remaining = SIZE_MAX; };
        check(!allocate(4 * fixture_page), "retained rollback refuses");
        check(backings[0]->retiring && backings[0]->segmented_error == oom && backings[0]->segmented->error() == oom,
              "original OOM retained, owner excluded");
        check(native_physical.size() == 2 && !native_releases, "fresh units stay retained after failed unmap");
        check(!arena_count_calls && backings[0]->ledger.snapshot().quarantined, "no transaction; lease quarantined");
        auto retained = refusal_record();
        check(retained[f_termination] == term_rollback_retained && retained[f_rb_fresh] == 2
              && retained[f_rb_released] == 0 && retained[f_rb_retained] == 2 && retained[f_t1_owners] == none,
              "retained rollback report");
        std::cout << "PASS failed rollback retains ownership\n";

        // C1: non-capacity status keeps today's retention and writes no record.
        reset(); enable(); fail_nth("create", 2, CUDA_ERROR_ILLEGAL_ADDRESS);
        check(!allocate(4 * fixture_page), "non-OOM refusal");
        check(backings[0]->retiring && backings[0]->segmented->error() == CUDA_ERROR_ILLEGAL_ADDRESS
              && native_physical.size() == 1 && !native_unmaps && !native_releases, "non-OOM retention unchanged");
        auto absent = refusal_record();
        check(absent[f_available] == 0 && absent[f_termination] == none, "no capacity record, never zero-filled");
        std::cout << "PASS non-capacity status unchanged\n";

        // NOT_READY -> barrier -> OOM continuation enters the transaction once;
        // completed dead units of another owner satisfy one retry.
        reset(); enable(); Split split; retire(split.middle); events_complete = true;
        native_fault = "access"; native_error = CUDA_ERROR_NOT_READY; native_fault_remaining = 1;
        native_access_failure = [] { native_error = oom; native_fault_remaining = 1; };
        auto continued = allocate(3 * fixture_page);
        check(continued != nullptr, "retry after NOT_READY->OOM succeeds");
        auto seq = refusal_record();
        check(seq[f_operation] == 3 && seq[f_first] == 600 && seq[f_barrier] == 0 && seq[f_continuation] == 2,
              "original status sequence preserved");
        check(ctx_sync_calls == 1 && !event_sync_calls, "only the single NOT_READY barrier");
        check(seq[f_t1_released] == 2 && seq[f_t1_live] == 2 && seq[f_t1_excluded] == 3
              && seq[f_retry] == 1 && seq[f_termination] == term_satisfied, "Tier 1 then one retry");
        retire(continued); split.finish(); closed();
        std::cout << "PASS NOT_READY then OOM enters transaction once\n";

        // W1: release no more units than the deficit.
        reset(); enable();
        { auto whole = allocate(6 * fixture_page); retire(whole); }
        std::vector<void*> units;
        for (int i = 0; i < 6; ++i) units.push_back(allocate(fixture_page));
        for (int i : {1, 3, 5}) retire(units[i]);
        events_complete = true;
        auto fragmented = backings[0];
        fail_nth("create", 1, oom);
        auto w1 = allocate(2 * fixture_page);
        check(w1 != nullptr, "fragmented dead units satisfy the retry");
        auto r1 = refusal_record();
        check(r1[f_t1_released] == 2 && unit_count(fragmented, SegmentedUnitKind::resident) == 4,
              "W1: exactly ceil(D/unit) units released; third dead unit kept");
        check(r1[f_termination] == term_satisfied && r1[f_t2_evictable] == none, "Tier 2 not entered");
        no_waits();
        native_observer = {}; retire(w1); for (int i : {0, 2, 4}) retire(units[i]); closed();
        std::cout << "PASS W1 deficit bound\n";

        // A2: requester's resident units inside the cancelled lease are excluded;
        // its dead units outside the lease are eligible. Rollback restores charge.
        reset(); enable();
        { auto whole = allocate(8 * fixture_page); retire(whole); }
        events_complete = true;
        auto requester = backings[0];
        check(reclaim_segmented(requester, 64, 2).released == 2, "fixture: units 0-1 discarded");
        bool charge_checked = false;
        auto creates = std::make_shared<size_t>(0);
        native_observer = [&, creates](const char* op) {
            if (std::string(op) == "create" && ++*creates == 1) {
                native_fault = "create"; native_error = oom; native_fault_remaining = 1;
            }
            if (std::string(op) == "unmap" && !charge_checked) {
                charge_checked = true;
                check(requester->segmented->physical_charge() == 6 * fixture_page, "rollback restored pre-claim charge");
            }
        };
        auto excluded_ptr = allocate(4 * fixture_page);
        check(excluded_ptr == reinterpret_cast<void*>(requester->address) && charge_checked, "same VA on retry");
        auto rx = refusal_record();
        check(rx[f_owner_new] == 0 && rx[f_deficit] == 2 * fixture_page && rx[f_t1_excluded] == 4
              && rx[f_t1_released] == 2, "exclusion then eligible outside units");
        check(requester->segmented->unit_kind(4) == SegmentedUnitKind::discarded
              && requester->segmented->unit_kind(5) == SegmentedUnitKind::discarded
              && requester->segmented->unit_kind(6) == SegmentedUnitKind::resident, "only outside units released");
        native_observer = {}; retire(excluded_ptr); closed();
        std::cout << "PASS requester exclusion and charge restoration\n";

        // Categories: pending (with capture-active Tier 3 skip), live-shared, busy owner.
        reset(); enable();
        { auto p = allocate(2 * fixture_page); retire(p); }
        fixture_capture_status = CU_STREAM_CAPTURE_STATUS_ACTIVE; fail_nth("create", 1, oom);
        check(!allocate(3 * fixture_page), "pending only, capture active");
        auto rp = refusal_record();
        check(rp[f_t1_pending] == 2 && rp[f_t1_released] == 0 && rp[f_t3_status] == 2 && rp[f_t3_fences] == none
              && rp[f_termination] == term_no_release && !event_sync_calls, "pending reported, never waited under capture");
        native_observer = {}; closed();

        reset(); enable();
        { auto a = allocate(512), b = allocate(512); retire(a); events_complete = true;
          fail_nth("create", 1, oom);
          check(!allocate(2 * fixture_page), "live-shared unit");
          auto rl = refusal_record();
          check(rl[f_t1_live] == 1 && rl[f_t1_released] == 0 && rl[f_termination] == term_no_release, "live-shared counted");
          native_observer = {}; retire(b); }
        closed();

        reset(); enable();
        { auto d = allocate(fixture_page); retire(d); auto c = allocate(2 * fixture_page); retire(c); }
        events_complete = true;
        auto busy_owner = backings[1];
        auto busy_creates = std::make_shared<size_t>(0);
        native_observer = [busy_creates](const char* op) {
            if (std::string(op) == "create" && ++*busy_creates == 1) {
                native_fault = "create"; native_error = oom; native_fault_remaining = 1;
            }
        };
        native_callback = {};
        bool busied = false;
        auto previous = native_observer;
        native_observer = [&, previous](const char* op) {
            previous(op);
            if (std::string(op) == "unmap" && !busied) { busied = true; busy_owner->retiring = true; }
        };
        auto busy_ptr = allocate(4 * fixture_page);
        auto rb = refusal_record();
        check(busy_ptr && rb[f_t1_owners] == 3 && rb[f_t1_owners_busy] == 1 && rb[f_t1_released] == 1
              && rb[f_termination] == term_satisfied, "busy owner counted and skipped, never waited");
        check(native_physical.size() == 2 + 4, "busy owner's units untouched");
        busy_owner->retiring = false; native_observer = {}; retire(busy_ptr); closed();
        std::cout << "PASS pending, live-shared and busy counted separately\n";

        // Completion error in Tier 1 retains ownership.
        reset(); enable();
        { auto q = allocate(fixture_page); retire(q); }
        auto poisoned = backings[0];
        native_observer = [](const char* op) {
            if (std::string(op) == "reserve") query_failure = true; // After fit, before the refusal.
            if (std::string(op) == "create") { native_fault = "create"; native_error = oom; native_fault_remaining = 1; }
        };
        check(!allocate(2 * fixture_page), "completion error refuses");
        auto rc = refusal_record();
        check(rc[f_termination] == term_completion_error && poisoned->retiring
              && poisoned->segmented_error == static_cast<int>(CUDA_ERROR_ILLEGAL_ADDRESS)
              && poisoned->segmented->physical_charge() == fixture_page, "failed completion retains ownership");
        std::cout << "PASS completion error retention\n";

        // Tier 2: arena pages cover the deficit; then W4 bound with one retry only.
        reset(); enable(); arena_pages = 5; fail_nth("create", 1, oom);
        auto t2 = allocate(3 * fixture_page);
        auto r2 = refusal_record();
        check(t2 && r2[f_t2_evictable] == 5 && r2[f_t2_evictions] == 3 && r2[f_t2_bytes] == 3 * fixture_page
              && r2[f_t2_status] == 0 && arena_pages == 2 && r2[f_termination] == term_satisfied, "Tier 2 bytes cover deficit");
        check(!arena_lock_violation, "Tier 2 runs without the pool lock"); no_waits();
        native_observer = {}; retire(t2); closed();

        reset(); enable(); arena_pages = 10; arena_count_override = 2;
        native_fault = "create"; native_error = oom; native_fault_remaining = SIZE_MAX;
        check(!allocate(4 * fixture_page), "persistent refusal");
        auto r4 = refusal_record();
        check(r4[f_t2_evictions] == 2 && arena_pages == 8, "W4: evictions bounded by observed count");
        check(native_creates == 2 && r4[f_retry] == 2 && r4[f_termination] == term_partial_release,
              "exactly one retry, then partial-release refusal");
        clean_rollback_owner(backings[0]); native_fault.clear(); closed();

        reset(); enable(); arena_pages = 3; arena_fail = true; fail_nth("create", 1, oom);
        check(!allocate(2 * fixture_page), "arena failure");
        auto rf = refusal_record();
        check(rf[f_t2_status] == 2 && rf[f_termination] == term_driver_error && rf[f_retry] == none
              && native_creates == 1, "A3: arena release failure stops Tier 2 without retry");
        native_observer = {}; closed();
        std::cout << "PASS Tier 2 arena pages, W4 and one retry\n";

        // Tier 3: bounded fence wait on selected pending units only (W5).
        reset(); enable();
        { auto whole = allocate(6 * fixture_page); retire(whole); }
        events_complete = true; reap_now(backings[0]); events_complete = false;
        std::vector<void*> parts;
        for (int i = 0; i < 6; ++i) parts.push_back(allocate(fixture_page));
        for (int i : {1, 3, 5}) retire(parts[i]); // Three distinct pending fences.
        fixture_event_sync_status = CUDA_SUCCESS; fail_nth("create", 1, oom);
        auto t3 = allocate(2 * fixture_page);
        auto r3 = refusal_record();
        check(t3 && r3[f_t1_pending] == 3 && r3[f_t3_status] == 1 && r3[f_t3_fences] == 2 && r3[f_t3_released] == 2,
              "W5: waits only the fences of the units needed");
        check(event_sync_calls == 2 && !ctx_sync_calls && !sync_lock_violation, "fence waits only, outside pool lock");
        check(r3[f_termination] == term_satisfied, "retry after bounded wait");
        native_observer = {}; retire(t3); for (int i : {0, 2, 4}) retire(parts[i]); closed();

        reset(); enable();
        { auto p = allocate(2 * fixture_page); retire(p); }
        auto waited = backings[0];
        fixture_event_sync_status = CUDA_ERROR_ILLEGAL_ADDRESS; fail_nth("create", 1, oom);
        check(!allocate(3 * fixture_page), "failed wait refuses");
        auto rw = refusal_record();
        check(rw[f_t3_status] == 3 && rw[f_termination] == term_completion_error && rw[f_retry] == none
              && waited->ledger.snapshot().quarantined == 2 * fixture_page
              && waited->segmented->physical_charge() == 2 * fixture_page, "failed completion quarantines and retains");
        std::cout << "PASS Tier 3 bounded completion wait\n";

        // Shared request quota: admission recovery already used the allowance.
        reset(); enable(); { Split quota; retire(quota.middle); events_complete = true;
          auto calls = std::make_shared<int>(0);
          auto_pressure = [calls](size_t) { return ++*calls == 1 ? DDO_VMM_OUT_OF_MEMORY : DDO_VMM_SUCCESS; };
          fail_nth("create", 1, oom);
          check(!allocate(3 * fixture_page), "quota refusal");
          auto rq = refusal_record();
          check(rq[f_termination] == term_quota_consumed && rq[f_t1_pool_busy] == none && rq[f_rb_released] == 1,
                "rollback still clean; no second recovery");
          auto_pressure = {}; native_observer = {}; quota.finish(); }
        closed();
        std::cout << "PASS shared request quota\n";

        // Tier 1 Driver failure retains the owner and stops without retry.
        reset(); enable();
        { auto d = allocate(fixture_page); retire(d); }
        events_complete = true;
        auto failing = backings[0];
        native_observer = [](const char* op) {
            if (std::string(op) == "create") { native_fault = "create"; native_error = oom; native_fault_remaining = 1; }
            if (std::string(op) == "unmap") { native_fault = "unmap"; native_error = CUDA_ERROR_INVALID_CONTEXT; native_fault_remaining = SIZE_MAX; }
        };
        const auto creates_before = native_creates;
        check(!allocate(2 * fixture_page), "discard failure");
        auto rd = refusal_record();
        check(rd[f_termination] == term_driver_error && rd[f_retry] == none && native_creates == creates_before + 1
              && failing->retiring && failing->segmented_error == static_cast<int>(CUDA_ERROR_INVALID_CONTEXT),
              "failed discard retained, no retry");
        std::cout << "PASS Tier 1 Driver failure retention\n";

        reset();
        check(!GetModuleHandleW(L"nvcuda.dll") && !GetModuleHandleW(L"cudart64_13.dll"), "no GPU library loaded");
        std::cout << "PASS capacity transaction CPU contracts; checks=" << checks << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; reset(); return 1;
    }
}
