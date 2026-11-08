#pragma once
#include "ddo_vmm_span_ledger.h"
#include <cstdint>
#include <mutex>
namespace ddo_vmm_detail {
// Last materialization refusal of one storage. Operation: 0 none, 1 create,
// 2 map, 3 access. Access statuses are the original first call, the single
// NOT_READY barrier and its continuation; a false flag means not observed.
struct SegmentedRefusal {
    bool observed = false, capacity = false, rolled_back = false;
    int operation = 0, status = 0, rollback_status = 0;
    size_t unit = SIZE_MAX, fresh = 0, released = 0, retained = 0, missing = 0;
    bool access_observed = false, barrier_observed = false, continuation_observed = false;
    int access_first = 0, access_barrier = 0, access_continuation = 0;
};
enum class SegmentedUnitKind { unavailable, discarded, resident, retained };
// CPU integration capability only. No native factory, public ABI or runtime enable.
// Caller owns pool lock on entry/return; implementations unlock only after proof
// and admission, and relock before returning, including refusal/failure paths.
struct SegmentedPoolStorage {
    virtual ~SegmentedPoolStorage() = default; // Never performs Driver cleanup.
    virtual bool materialize(const SpanLedger::Lease&, const SpanLedger&,
                             std::unique_lock<std::mutex>&) noexcept = 0;
    virtual bool discard(size_t, const SpanLedger&, std::unique_lock<std::mutex>&) noexcept = 0;
    virtual bool close_va(std::unique_lock<std::mutex>&) noexcept = 0;
    virtual size_t page_count() const noexcept = 0;
    virtual size_t physical_charge() const noexcept = 0;
    virtual int error() const noexcept = 0;
    virtual bool physical_snapshot(const SpanLedger&, SpanLedger::Snapshot&) const { return false; }
    // Capacity-transaction observations; defaults report "unavailable".
    virtual SegmentedRefusal last_refusal() const noexcept { return {}; }
    virtual SegmentedUnitKind unit_kind(size_t) const noexcept { return SegmentedUnitKind::unavailable; }
    virtual size_t unit_bytes() const noexcept { return 0; }
};
}
