// Pure CPU ownership/transaction contracts; no CUDA or payload allocation.
#include "ddo_vmm_span_ledger.h"
#include <cstdio>
#include <cstdlib>
#include <new>

static long fail_after = -1;
void* operator new(std::size_t size) {
    if (fail_after == 0) throw std::bad_alloc();
    if (fail_after > 0) --fail_after;
    if (void* result = std::malloc(size ? size : 1)) return result;
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

int main() {
    using namespace ddo_vmm_detail;
    size_t checks = 0, failures = 0;
    auto require = [&](bool condition) { ++checks; failures += !condition; };
    auto conserved = [&](const SpanLedger& ledger) {
        const auto s = ledger.snapshot();
        require(s.mapped == s.leased + s.available + s.pending + s.quarantined);
        require(s.requested <= s.leased);
    };
    auto live = [&](SpanLedger& ledger, size_t bytes) {
        auto lease = ledger.reserve(bytes, 0);
        if (!lease) throw std::bad_alloc();
        require(ledger.publish(*lease, true));
        return *lease;
    };
    const auto a = std::make_shared<SpanFence>();
    const auto b = std::make_shared<SpanFence>();
    const auto completion = std::make_shared<SpanFence>();

    SpanLedger joined(4096, 0, 1);
    const auto left = live(joined, 1024);
    const auto middle = live(joined, 2048);
    const auto right = live(joined, 1024);
    require(joined.retire(left, a) && joined.retire(middle, b) && joined.retire(right, a));
    require(!joined.fit_size(2560, 0) && !joined.reserve(2560, 0));
    require(joined.fit_size_joined(2560, 0) == 4096);
    auto lease = joined.reserve_joined(2560, 0);
    require(lease && lease->offset == 0 && lease->size == 2560
            && lease->dependencies.size() == 2);
    require(lease->dependencies[0] == a && lease->dependencies[1] == b);
    require(joined.snapshot().leased == 2560 && joined.snapshot().pending == 1536);
    // Observations cannot release an unpublished reservation.
    joined.observe(a, SpanCompletion::complete);
    require(joined.snapshot().leased == 2560 && !joined.begin_backing_release());
    require(joined.cancel_unsubmitted(*lease));
    require(!joined.cancel_unsubmitted(*lease));
    require(joined.snapshot().pending == 3072 && joined.snapshot().available == 1024);
    conserved(joined);
    lease = joined.reserve_joined(4096, 0);
    require(lease && lease->dependencies.size() == 2);
    require(joined.publish(*lease, true));
    require(joined.retire(*lease, completion));
    require(joined.pending_fences().size() == 1
            && joined.pending_fences()[0] == completion);
    joined.observe(completion, SpanCompletion::complete);
    require(joined.releasable());
    conserved(joined);

    // Partial dependency submission must quarantine, never cancel as unused.
    SpanLedger partial(2048, 0, 2);
    const auto p = live(partial, 1024), q = live(partial, 1024);
    require(partial.retire(p, a) && partial.retire(q, b));
    auto quarantined = partial.reserve_joined(2048, 0);
    require(quarantined && quarantined->dependencies.size() == 2);
    require(!partial.publish(*quarantined, false));
    require(!partial.cancel_unsubmitted(*quarantined));
    partial.observe(a, SpanCompletion::complete);
    partial.observe(b, SpanCompletion::complete);
    require(partial.snapshot().quarantined == 2048 && !partial.releasable());
    require(!partial.reserve_joined(512, 0));
    conserved(partial);

    // Available + pending may join; live/reserved/quarantined barriers may not.
    SpanLedger barriers(4096, 0, 3);
    const auto barrier = live(barriers, 1024);
    const auto pending = live(barriers, 1024);
    require(barriers.retire(pending, b));
    require(!barriers.fit_size_joined(3584, 0));
    auto mixed = barriers.reserve_joined(3072, 0);
    require(mixed && mixed->offset == 1024 && mixed->dependencies.size() == 1);
    require(!barriers.reserve_joined(1536, 0));
    require(barriers.cancel_unsubmitted(*mixed));
    require(!barriers.retire(barrier, {}));
    require(!barriers.fit_size_joined(4096, 0));
    conserved(barriers);
    require(!barriers.reserve_joined(SIZE_MAX, 0));
    require(!barriers.reserve_joined(512, 1));

    // Prefer the unchanged single-span fit; deduplicate shared dependencies.
    SpanLedger prefer(4096, 0, 4);
    const auto first = live(prefer, 1024), second = live(prefer, 1024);
    require(prefer.retire(first, a) && prefer.retire(second, b));
    auto single = prefer.reserve_joined(1024, 0);
    require(single && single->offset == 0 && single->dependencies.size() == 1);
    require(prefer.cancel_unsubmitted(*single));
    auto all = prefer.reserve_joined(4096, 0);
    require(all && all->dependencies.size() == 2);
    require(prefer.cancel_unsubmitted(*all));
    const auto repeated = prefer.reserve_joined(4096, 0);
    require(repeated && repeated->dependencies.size() == 2);
    require(prefer.cancel_unsubmitted(*repeated));
    conserved(prefer);

    // Any throwing metadata allocation must leave the original partition intact.
    size_t injected_failures = 0, successful_transactions = 0;
    for (long point = 0; point < 16; ++point) {
        SpanLedger transaction(4096, 0, static_cast<uint64_t>(point + 10));
        const auto x = live(transaction, 1024), y = live(transaction, 2048);
        require(transaction.retire(x, a) && transaction.retire(y, b));
        const auto before = transaction.snapshot();
        fail_after = point;
        try {
            auto result = transaction.reserve_joined(2560, 0);
            fail_after = -1;
            require(result && result->dependencies.size() == 2);
            require(transaction.cancel_unsubmitted(*result));
            ++successful_transactions;
        } catch (const std::bad_alloc&) {
            fail_after = -1;
            ++injected_failures;
            const auto after = transaction.snapshot();
            require(after.pending == before.pending && after.available == before.available
                    && after.leased == before.leased && after.reserved_count == 0);
            require(transaction.pending_fences().size() == 2);
            require(!transaction.fit_size(2560, 0));
        }
        conserved(transaction);
    }
    require(injected_failures > 0 && successful_transactions > 0);
    std::printf("Joined span CPU contracts: %s (%zu checks, %zu metadata failures)\n",
                failures ? "FAIL" : "PASS", checks, injected_failures);
    return failures ? 1 : 0;
}
