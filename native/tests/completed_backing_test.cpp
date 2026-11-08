#include "ddo_vmm_completed_backing.h"
#include <cstdlib>
#include <iostream>

namespace {
using namespace ddo_vmm_detail;
int checks = 0;
void require(bool value) {
    ++checks;
    if (!value) { std::cerr << "FAIL check " << checks << '\n'; std::exit(1); }
}
auto pending(SpanLedger& ledger) {
    const auto lease = ledger.reserve(4096, 0);
    require(lease.has_value());
    require(ledger.publish(*lease, true));
    auto fence = std::make_shared<SpanFence>();
    require(ledger.retire(*lease, fence));
    return fence;
}
void conserved(const SpanLedger& ledger) {
    const auto s = ledger.snapshot();
    require(s.mapped == s.leased + s.available + s.pending + s.quarantined);
}
}
int main() {
    using namespace ddo_vmm_detail;
    static_assert(arena_pool_release_limit == 16);
    // Fresh idle backing needs no event or synchronization.
    {
        SpanLedger ledger(4096, 0, 1);
        int queries = 0, releases = 0;
        auto run = [&] {
            return release_completed_backing(ledger,
                [&](auto) { ++queries; return SpanCompletion::complete; },
                [&] {
                    ++releases;
                    require(!ledger.reserve(512, 0));
                    require(!ledger.begin_backing_release());
                    return true;
                });
        };
        require(run() == CompletedRelease::released);
        require(queries == 0 && releases == 1);
        require(ledger.snapshot().mapped == 0);
        require(run() == CompletedRelease::unavailable);
        require(releases == 1);
        conserved(ledger);
    }
    // Reserved, live and failed-publication owners cannot be released.
    for (int state = 0; state < 3; ++state) {
        SpanLedger ledger(4096, 0, 2);
        const auto lease = ledger.reserve(512, 0);
        require(lease.has_value());
        if (state) require(ledger.publish(*lease, state == 1) == (state == 1));
        int queries = 0, releases = 0;
        require(release_completed_backing(ledger,
            [&](auto) { ++queries; return SpanCompletion::complete; },
            [&] { ++releases; return true; }) == CompletedRelease::unavailable);
        require(queries == 0 && releases == 0);
        require(ledger.snapshot().mapped == 4096);
        conserved(ledger);
    }
    // Pending consumers are queried, never waited for or force-released.
    {
        SpanLedger ledger(4096, 0, 3);
        auto fence = pending(ledger);
        int queries = 0, releases = 0;
        auto release = [&] { ++releases; return true; };
        require(release_completed_backing(ledger, [&](const auto& seen) {
            require(seen == fence); ++queries; return SpanCompletion::not_ready;
        }, release) == CompletedRelease::unavailable);
        require(queries == 1 && releases == 0);
        require(ledger.snapshot().pending == 4096);
        require(release_completed_backing(ledger, [&](const auto& seen) {
            require(seen == fence); ++queries; return SpanCompletion::complete;
        }, release) == CompletedRelease::released);
        require(queries == 2 && releases == 1);
        require(ledger.snapshot().mapped == 0);
        conserved(ledger);
    }
    // Completion failure quarantines and retains ownership without native free.
    {
        SpanLedger ledger(4096, 0, 4);
        auto fence = pending(ledger);
        int releases = 0;
        require(release_completed_backing(ledger,
            [&](auto) { return SpanCompletion::failed; },
            [&] { ++releases; return true; }) == CompletedRelease::failed);
        require(releases == 0 && ledger.snapshot().mapped == 4096);
        require(ledger.snapshot().quarantined == 4096);
        require(!ledger.reserve(512, 0));
        conserved(ledger);
    }
    // Failed native unmap/release cannot become available or be retried.
    {
        SpanLedger ledger(4096, 0, 5);
        int releases = 0;
        auto run = [&] {
            return release_completed_backing(ledger,
                [&](auto) { return SpanCompletion::complete; },
                [&] { ++releases; return false; });
        };
        require(run() == CompletedRelease::failed);
        require(releases == 1 && ledger.snapshot().quarantined == 4096);
        require(run() == CompletedRelease::unavailable);
        require(releases == 1 && ledger.snapshot().mapped == 4096);
        conserved(ledger);
    }
    // A partially completed backing remains owned while any consumer is pending.
    {
        SpanLedger ledger(8192, 0, 6);
        const auto lease_a = ledger.reserve(4096, 0);
        const auto lease_b = ledger.reserve(4096, 0);
        require(lease_a && lease_b);
        require(ledger.publish(*lease_a, true) && ledger.publish(*lease_b, true));
        const auto first = std::make_shared<SpanFence>();
        const auto second = std::make_shared<SpanFence>();
        require(ledger.retire(*lease_a, first) && ledger.retire(*lease_b, second));
        int queries = 0, releases = 0;
        require(release_completed_backing(ledger, [&](const auto& fence) {
            ++queries;
            return fence == first ? SpanCompletion::complete : SpanCompletion::not_ready;
        }, [&] { ++releases; return true; }) == CompletedRelease::unavailable);
        require(queries == 2 && releases == 0);
        require(ledger.snapshot().mapped == 8192);
        require(ledger.snapshot().pending == 4096 && ledger.snapshot().available == 4096);
        conserved(ledger);
    }
    std::cout << "PASS " << checks << " completed-backing CPU assertions; no CUDA calls\n";
}
