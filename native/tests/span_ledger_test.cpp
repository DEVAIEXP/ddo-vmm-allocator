#include "ddo_vmm_span_ledger.h"
#include <cstdio>

int main() {
    using namespace ddo_vmm_detail;
    size_t failures = 0, checks = 0;
    auto require = [&](bool condition) { ++checks; failures += !condition; };
    auto conserved = [&](const SpanLedger& ledger) {
        const auto s = ledger.snapshot();
        require(s.leased + s.available + s.pending + s.quarantined == s.mapped);
        require(s.requested <= s.leased);
    };
    const auto original = std::make_shared<SpanFence>();
    const auto consumer_a = std::make_shared<SpanFence>();
    const auto consumer_b = std::make_shared<SpanFence>();

    SpanLedger ledger(4096, 0, 1);
    require(!ledger.reserve(1, 1));
    require(!ledger.reserve(SIZE_MAX, 0));
    require(!ledger.reserve(4097, 0));
    require(ledger.fit_size(513, 0) == 4096 && !ledger.fit_size(1, 1));
    const auto whole = ledger.reserve(4096, 0);
    require(whole && ledger.publish(*whole, true));
    require(!ledger.releasable());
    require(ledger.retire(*whole, original));
    require(ledger.pending_fences().size() == 1 && ledger.pending_fences()[0] == original);
    require(!ledger.retire(*whole, original));
    require(!ledger.releasable());
    ledger.observe(original, SpanCompletion::not_ready);
    require(ledger.snapshot().pending == 4096);
    const auto a = ledger.reserve(513, 0);
    const auto b = ledger.reserve(0, 0);
    require(a && a->offset == 0 && a->size == 1024 && a->dependencies.size() == 1);
    require(b && b->offset == 1024 && b->size == 512 && b->dependencies[0] == original);
    conserved(ledger);
    require(ledger.publish(*a, true) && ledger.publish(*b, true));
    require(!ledger.quarantine_idle_backing());
    require(!ledger.publish(*a, true));
    require(ledger.retire(*a, consumer_a));
    require(ledger.retire(*b, consumer_b));
    require(ledger.pending_fences().size() == 3);
    conserved(ledger);
    // Distinct pending fences must not be discarded/coalesced into a large fit.
    require(!ledger.reserve(3072, 0));
    ledger.observe(original, SpanCompletion::complete);
    require(ledger.snapshot().available == 2560 && ledger.snapshot().pending == 1536);
    require(!ledger.releasable());
    ledger.observe(consumer_a, SpanCompletion::complete);
    require(!ledger.reserve(3584, 0)); // Middle 512-byte slice is still pending.
    ledger.observe(consumer_b, SpanCompletion::complete);
    require(ledger.releasable());
    const auto joined = ledger.reserve(4096, 0);
    require(joined && joined->offset == 0 && joined->dependencies.empty());
    require(!ledger.cancel_unsubmitted(*whole)); // Stale generation of lease.
    require(ledger.cancel_unsubmitted(*joined));
    require(!ledger.commit_backing_release());
    require(ledger.begin_backing_release());
    require(!ledger.reserve(512, 0) && !ledger.begin_backing_release());
    require(ledger.snapshot().pending == 4096);
    conserved(ledger);
    require(ledger.commit_backing_release());
    require(!ledger.commit_backing_release() && !ledger.reserve(1, 0));
    conserved(ledger);

    // Same pending fence can safely cover adjacent pieces after rollback.
    SpanLedger rollback(2048, 0, 2);
    const auto initial = rollback.reserve(2048, 0);
    require(rollback.publish(*initial, true) && rollback.retire(*initial, original));
    const auto split = rollback.reserve(512, 0);
    require(rollback.cancel_unsubmitted(*split));
    const auto restored = rollback.reserve(2048, 0);
    require(restored && restored->dependencies.size() == 1);
    require(!rollback.publish(*restored, false));
    require(rollback.snapshot().quarantined == 2048);
    rollback.observe(original, SpanCompletion::complete);
    require(!rollback.releasable() && !rollback.reserve(512, 0));
    conserved(rollback);

    SpanLedger failed(1024, 0, 3);
    const auto live = failed.reserve(512, 0);
    require(!failed.publish(*initial, true)); // Foreign backing token.
    auto stale = *live;
    ++stale.generation;
    require(!failed.publish(stale, true));
    require(failed.publish(*live, true));
    require(!failed.retire(*live, {})); // Missing new completion proof.
    require(failed.snapshot().quarantined == 512 && !failed.releasable());
    const auto other = failed.reserve(512, 0);
    require(other && failed.publish(*other, true) && failed.retire(*other, original));
    failed.observe(original, SpanCompletion::failed);
    require(failed.snapshot().quarantined == 1024);
    conserved(failed);

    SpanLedger best_fit(4096, 0, 4);
    const auto left = best_fit.reserve(512, 0);
    const auto middle = best_fit.reserve(512, 0);
    require(best_fit.publish(*left, true) && best_fit.publish(*middle, true));
    require(best_fit.retire(*left, original));
    const auto selected = best_fit.reserve(512, 0);
    require(selected && selected->offset == 0 && selected->dependencies[0] == original);
    conserved(best_fit);

    // Repeated split/rollback must conserve one physical backing, not sum
    // a minimum VMM mapping for each small request.
    SpanLedger repeated(8192, 0, 5);
    for (size_t i = 0; i < 200; ++i) {
        auto x = repeated.reserve(i, 0);
        require(x && repeated.snapshot().mapped == 8192);
        conserved(repeated);
        require(repeated.cancel_unsubmitted(*x));
        require(!repeated.cancel_unsubmitted(*x));
        require(repeated.releasable());
    }
    require(repeated.begin_backing_release() && repeated.commit_backing_release());
    SpanLedger release_failure(1024, 0, 6);
    require(release_failure.begin_backing_release());
    require(release_failure.fail_backing_release());
    require(!release_failure.reserve(512, 0) && !release_failure.begin_backing_release());
    require(!release_failure.commit_backing_release());
    require(release_failure.snapshot().quarantined == 1024);
    conserved(release_failure);

    // Dropping a caller's event reference must not discard a pending fence.
    SpanLedger retained(1024, 0, 7);
    auto fence = std::make_shared<SpanFence>();
    std::weak_ptr<const SpanFence> weak = fence;
    const auto retained_lease = retained.reserve(1024, 0);
    require(retained.publish(*retained_lease, true) && retained.retire(*retained_lease, fence));
    fence.reset();
    require(!weak.expired());
    retained.observe(weak.lock(), SpanCompletion::complete);
    require(weak.expired() && retained.releasable());
    SpanLedger empty_failure(1024, 0, 8);
    require(empty_failure.quarantine_idle_backing());
    require(!empty_failure.fit_size(512, 0) && !empty_failure.releasable());
    conserved(empty_failure);
    std::printf("Span ledger CPU contracts: %s (%zu checks)\n", failures ? "FAIL" : "PASS", checks);
    return failures ? 1 : 0;
}
