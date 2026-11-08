#pragma once
#include "ddo_vmm_span_ledger.h"
#include <vector>
#include <type_traits>

namespace ddo_dead_pages {
// Owner serializes entry and excludes reserve/reclaim/close for this transaction.
// Driver calls run outside the canonical pool lock. This storage seam is not
// registered with the runtime pool and never owns a second admission budget.
template<class Driver> class Storage {
public:
    using Handle = typename Driver::Handle;
    enum class State { discarded, resident, retained_failure };
    struct Unit { Handle handle{}; State state = State::discarded; bool mapped = false; };
    // Last materialization refusal. Operation: 0 none, 1 create, 2 map,
    // 3 access. Rollback applies only to Driver-classified capacity refusals.
    struct Refusal {
        bool observed = false, capacity = false, rolled_back = false;
        int operation = 0, status = 0, rollback_status = 0;
        size_t unit = SIZE_MAX, fresh = 0, released = 0, retained = 0;
        size_t missing = 0; // Units the lease lacked at claim time (deficit D).
    };
private:
    Driver& driver_;
    const ddo_vmm_detail::SpanLedger* ledger_;
    uintptr_t base_;
    size_t page_, extent_, charged_ = 0;
    int device_;
    uint64_t generation_;
    std::vector<Unit> units_;
    bool failed_ = false;
    int error_ = 0;
    Refusal refusal_;
    bool fail(int error) { failed_ = true; error_ = error; return false; }
    bool capacity_refusal(int status) const noexcept {
        if constexpr (requires { driver_.capacity_refusal(status); }) return driver_.capacity_refusal(status);
        else return false; // Drivers without classification keep retention.
    }
    // Undo this call's never-published fresh units, newest first, outside the
    // pool lock. Each successful release credits one unit. Any failure stops
    // the walk: remaining units and the unclaimed tail stay charged/retained.
    template<class Credit>
    bool rollback_fresh(const std::vector<size_t>& fresh, size_t untried, Credit& credit) {
        for (auto it = fresh.rbegin(); it != fresh.rend(); ++it) {
            auto& u = units_[*it];
            if (u.mapped) {
                if (const int status = driver_.unmap(base_ + *it * page_, page_)) {
                    refusal_.rollback_status = status; break;
                }
                u.mapped = false;
            }
            if (u.handle) {
                if (const int status = driver_.release(u.handle)) {
                    refusal_.rollback_status = status; break;
                }
                u.handle = {};
            }
            u.state = State::discarded;
            charged_ -= page_; credit(page_);
            ++refusal_.released;
        }
        refusal_.retained = refusal_.fresh - refusal_.released;
        if (refusal_.rollback_status) return false;
        if (untried) { charged_ -= untried * page_; credit(untried * page_); }
        refusal_.rolled_back = true;
        return true;
    }
public:
    // Caller owns a reserved VA extent. No handle or budget is claimed here.
    Storage(Driver& driver, const ddo_vmm_detail::SpanLedger& ledger, uintptr_t address,
            size_t extent, size_t granularity, int device,
            uint64_t generation) : driver_(driver), ledger_(&ledger), base_(address), page_(granularity), extent_(extent),
            device_(device), generation_(generation) {
        if (!address || !granularity || granularity % 512 || address % granularity || !extent || extent % granularity
            || extent > UINTPTR_MAX - address || device < 0 || !generation || extent / granularity > 4096
            || !ledger.dead_range(0, extent, device, generation))
            throw std::invalid_argument("segmented storage identity/geometry");
        units_.resize(extent / granularity);
    }
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    // Explicit caller retention on failure; destructor never frees handles/VA.
    size_t charged() const { return charged_; }
    int error() const { return error_; }
    bool failed() const { return failed_; }
    const Unit& unit(size_t index) const { return units_.at(index); }
    const Refusal& last_refusal() const noexcept { return refusal_; }
    size_t page() const noexcept { return page_; }
    bool physical_snapshot(const ddo_vmm_detail::SpanLedger& ledger,
            ddo_vmm_detail::SpanLedger::Snapshot& output) const {
        if (failed_ || &ledger != ledger_) return false;
        ddo_vmm_detail::SpanLedger::Snapshot result;
        for (size_t i = 0; i < units_.size(); ++i) {
            if (units_[i].state == State::discarded) continue;
            if (units_[i].state != State::resident) return false;
            const auto part = ledger.range_snapshot(i * page_, page_);
            if (!part) return false;
            result.mapped += page_; result.leased += part->leased;
            result.available += part->available; result.pending += part->pending;
            result.quarantined += part->quarantined;
        }
        if (result.mapped != charged_) return false;
        const auto full = ledger.snapshot();
        result.requested = full.requested; result.live_count = full.live_count;
        result.reserved_count = full.reserved_count;
        output = result; return true;
    }

    // Actual ledger reservation must cover the range and remain held until
    // publication or rollback. Claim is the canonical native admission callback.
    // Drivers without capacity classification never roll back, so no credit
    // path is needed. Capacity-classifying drivers must supply one.
    template<class Claim>
    bool materialize(const ddo_vmm_detail::SpanLedger::Lease& lease,
            const ddo_vmm_detail::SpanLedger& ledger, Claim claim)
        requires (!requires(Driver& d) { d.capacity_refusal(0); }) {
        auto credit = [](size_t) noexcept {};
        return materialize(lease, ledger, claim, credit);
    }
    // Credit(bytes) returns rolled-back charge; it is invoked outside the pool
    // lock, exactly like discard credit.
    template<class Claim, class Credit>
    bool materialize(const ddo_vmm_detail::SpanLedger::Lease& lease,
            const ddo_vmm_detail::SpanLedger& ledger, Claim claim, Credit credit) {
        static_assert(std::is_nothrow_invocable_r_v<bool, Claim, size_t>);
        static_assert(std::is_nothrow_invocable_v<Credit, size_t>);
        static_assert(noexcept(driver_.identity(device_)) && noexcept(driver_.create(page_, units_[0].handle))
            && noexcept(driver_.map(base_, page_, units_[0].handle)) && noexcept(driver_.access(base_, page_)));
        refusal_ = {};
        if (failed_ || &ledger != ledger_ || lease.owner != &ledger || lease.generation != generation_
            || !lease.size || lease.offset > extent_ || lease.size > extent_ - lease.offset
            || ledger.snapshot().mapped != extent_ || !driver_.identity(device_)
            || !ledger.owns_reservation(lease, device_)) return false;
        const auto first = lease.offset / page_, last = (lease.offset + lease.size - 1) / page_;
        size_t missing = 0;
        for (size_t i = first; i <= last; ++i) {
            if (units_[i].state == State::retained_failure) return false;
            if (units_[i].state == State::discarded) missing += page_;
        }
        // Request-local rollback list, allocated before any claim or Driver call.
        std::vector<size_t> fresh;
        try { fresh.reserve(missing / page_); } catch (...) { return false; }
        if (!claim(missing)) return false;
        charged_ += missing; // Conservative reservation retained on partial failure.
        size_t attempted = 0;
        auto refuse = [&](int operation, int status, size_t index) {
            refusal_.observed = true; refusal_.operation = operation;
            refusal_.status = status; refusal_.unit = index;
            refusal_.capacity = capacity_refusal(status);
            refusal_.missing = missing / page_;
            if (!refusal_.capacity) return fail(status); // Unchanged retention.
            refusal_.fresh = fresh.size();
            if (!rollback_fresh(fresh, missing / page_ - attempted, credit)) return fail(status);
            return false; // Clean refusal: storage stays healthy, original status kept in refusal_.
        };
        for (size_t i = first; i <= last; ++i) {
            auto& u = units_[i];
            if (u.state == State::resident) continue;
            u.state = State::retained_failure;
            fresh.push_back(i); ++attempted; // Capacity reserved above; push cannot allocate.
            if (const int status = driver_.create(page_, u.handle)) return refuse(1, status, i);
            if (!u.handle) return fail(-1); // Invalid successful Driver result; publish nothing.
            if (const int status = driver_.map(base_ + i * page_, page_, u.handle)) return refuse(2, status, i);
            u.mapped = true;
            if (const int status = driver_.access(base_ + i * page_, page_)) return refuse(3, status, i);
            u.state = State::resident;
        }
        return true;
    }
    // Proof is rechecked after the caller has claimed maintenance exclusion.
    // Budget credit follows successful handle release, never the attempt.
    template<class Credit, class Begin>
    bool discard(size_t index, const ddo_vmm_detail::SpanLedger& ledger, Credit credit, Begin begin) {
        static_assert(std::is_nothrow_invocable_v<Credit, size_t>);
        static_assert(std::is_nothrow_invocable_v<Begin>);
        static_assert(noexcept(driver_.identity(device_)) && noexcept(driver_.unmap(base_, page_))
            && noexcept(driver_.release(units_[0].handle)));
        if (failed_ || &ledger != ledger_ || index >= units_.size() || !driver_.identity(device_)
            || units_[index].state != State::resident
            || !ledger.dead_range(index * page_, page_, device_, generation_)) return false;
        begin(); // Caller releases pool serialization only after the full proof.
        auto& u = units_[index];
        u.state = State::retained_failure;
        if (const int status = driver_.unmap(base_ + index * page_, page_)) return fail(status);
        u.mapped = false;
        if (const int status = driver_.release(u.handle)) return fail(status);
        u.handle = {};
        u.state = State::discarded;
        charged_ -= page_;
        credit(page_);
        return true;
    }
    template<class Credit, class Begin>
    bool close_dead(const ddo_vmm_detail::SpanLedger& ledger, Credit credit, Begin begin) {
        static_assert(std::is_nothrow_invocable_v<Credit, size_t> && std::is_nothrow_invocable_v<Begin>);
        if (failed_ || &ledger != ledger_ || !ledger.release_claimed() || !driver_.identity(device_)) return false;
        begin();
        for (size_t i = 0; i < units_.size(); ++i) {
            auto& u = units_[i];
            if (u.state == State::discarded) continue;
            u.state = State::retained_failure;
            if (const int status = driver_.unmap(base_ + i * page_, page_)) return fail(status);
            u.mapped = false;
            if (const int status = driver_.release(u.handle)) return fail(status);
            u.handle = {}; u.state = State::discarded;
            charged_ -= page_; credit(page_);
        }
        return true;
    }
};
}
