#include "ddo_vmm_span_ledger.h"
#include <iostream>
#include <stdexcept>
using namespace ddo_vmm_detail;
constexpr size_t MiB = 1024 * 1024;
size_t checks = 0;
void require(bool ok) {
    ++checks;
    if (!ok) throw std::runtime_error("CPU geometry contract failed");
}
SpanLedger::Lease live(SpanLedger& ledger, size_t bytes) {
    auto lease = ledger.reserve_joined(bytes, 0);
    require(lease.has_value());
    require(ledger.publish(*lease, true));
    return *lease;
}
void complete(SpanLedger& ledger, const SpanLedger::Lease& lease) {
    auto fence = std::make_shared<SpanFence>();
    require(ledger.retire(lease, fence));
    ledger.observe(fence, SpanCompletion::complete);
}
void check_geometry(const SpanLedger& ledger, size_t requested = 688 * MiB) {
    const auto before = ledger.snapshot();
    const auto geometry = ledger.geometry(requested, 0);
    const auto after = ledger.snapshot();
    require(geometry.fits_reusable == ledger.fit_size_joined(requested, 0).has_value());
    require(before.mapped == after.mapped && before.requested == after.requested
        && before.leased == after.leased && before.available == after.available
        && before.pending == after.pending && before.quarantined == after.quarantined
        && before.live_count == after.live_count && before.reserved_count == after.reserved_count);
}
int main() {
    {
        SpanLedger ledger(1024 * MiB, 0, 1);
        check_geometry(ledger);
        auto geometry = ledger.geometry(688 * MiB, 0);
        require(geometry.largest_available == 1024 * MiB);
        require(geometry.largest_reusable == 1024 * MiB);
        require(geometry.available_runs == 1 && geometry.reusable_runs == 1);
        require(geometry.whole_backing_releasable);
        auto lease = ledger.reserve(512 * MiB, 0);
        require(lease.has_value());
        geometry = ledger.geometry(688 * MiB, 0);
        require(geometry.largest_available == 512 * MiB && !geometry.fits_reusable);
        require(!geometry.whole_backing_releasable);
        require(ledger.cancel_unsubmitted(*lease));
        check_geometry(ledger);
    }
    {
        SpanLedger ledger(1024 * MiB, 0, 2);
        const auto left = live(ledger, 400 * MiB);
        const auto middle = live(ledger, 224 * MiB);
        const auto right = live(ledger, 400 * MiB);
        complete(ledger, left);
        complete(ledger, right);
        auto geometry = ledger.geometry(688 * MiB, 0);
        require(ledger.snapshot().available == 800 * MiB);
        require(geometry.largest_available == 400 * MiB && geometry.largest_reusable == 400 * MiB);
        require(geometry.available_runs == 2 && geometry.reusable_runs == 2);
        require(!geometry.fits_available && !geometry.fits_reusable);
        check_geometry(ledger);
        complete(ledger, middle);
        check_geometry(ledger);
        require(ledger.geometry(688 * MiB, 0).largest_available == 1024 * MiB);
    }
    {
        SpanLedger ledger(1024 * MiB, 0, 3);
        const auto lease = live(ledger, 400 * MiB);
        auto fence = std::make_shared<SpanFence>();
        require(ledger.retire(lease, fence));
        const auto owners = fence.use_count();
        const auto geometry = ledger.geometry(688 * MiB, 0);
        require(fence.use_count() == owners);
        require(geometry.largest_available == 624 * MiB);
        require(geometry.largest_reusable == 1024 * MiB);
        require(!geometry.fits_available && geometry.fits_reusable);
        require(!geometry.whole_backing_releasable);
        check_geometry(ledger);
        ledger.observe(fence, SpanCompletion::not_ready);
        require(ledger.geometry(688 * MiB, 0).largest_available == 624 * MiB);
        auto joined = ledger.reserve_joined(688 * MiB, 0);
        require(joined.has_value() && joined->dependencies.size() == 1);
        require(ledger.cancel_unsubmitted(*joined));
        ledger.observe(fence, SpanCompletion::complete);
        check_geometry(ledger);
        require(ledger.geometry(688 * MiB, 0).whole_backing_releasable);
    }
    {
        SpanLedger ledger(1024 * MiB, 0, 4);
        const auto lease = live(ledger, 400 * MiB);
        auto fence = std::make_shared<SpanFence>();
        require(ledger.retire(lease, fence));
        ledger.observe(fence, SpanCompletion::failed);
        check_geometry(ledger);
        const auto geometry = ledger.geometry(688 * MiB, 0);
        require(geometry.largest_reusable == 624 * MiB);
        require(!geometry.fits_reusable && !geometry.whole_backing_releasable);
    }
    {
        SpanLedger ledger(1024 * MiB, 0, 5);
        require(!ledger.geometry(688 * MiB, 1).eligible);
        require(!ledger.geometry(SIZE_MAX, 0).eligible);
        require(ledger.geometry(0, 0).fits_available);
        require(ledger.begin_backing_release());
        require(!ledger.geometry(688 * MiB, 0).eligible);
        check_geometry(ledger);
        require(ledger.commit_backing_release());
        require(!ledger.geometry(688 * MiB, 0).eligible);
        check_geometry(ledger);
    }
    {
        SpanLedger ledger(4096, 0, 6);
        const auto first = live(ledger, 1);
        const auto middle = live(ledger, 513);
        const auto last = live(ledger, 2560);
        complete(ledger, first);
        complete(ledger, last);
        for (size_t request = 0; request <= 4608; request += 257)
            check_geometry(ledger, request);
        require(!ledger.geometry(2561, 0).fits_reusable);
        require(ledger.geometry(2560, 0).fits_available);
        complete(ledger, middle);
        check_geometry(ledger, 4096);
    }
    std::cout << "{\"status\":\"pass\",\"cases\":6,\"checks\":" << checks
        << ",\"cuda\":false,\"payload_bytes\":0,\"runtime_installed\":false}\n";
}
