#pragma once
#include "ddo_vmm_segmented_pool_cpu.h"
#include "factory.h"

namespace ddo_vmm_detail {
// Explicit experimental registration only. No automatic size/lifetime policy.
template<class Driver> class SegmentedDriverStorage final : public SegmentedPoolStorage {
    struct Account {
        std::mutex& mutex;
        size_t va = 0, physical = 0;
        bool claim_va(size_t bytes) noexcept {
            std::scoped_lock lock(mutex);
            if (va) return false;
            va = bytes; return true;
        }
        void credit_va(size_t bytes) noexcept {
            std::scoped_lock lock(mutex); va -= bytes;
        }
    } account_;
    Driver driver_;
    ddo_dead_pages::FactoryOwner<Driver, Account> factory_;
    int device_;
    int (*admit_)(int, size_t) noexcept;
    size_t extent_pages_;
    int driver_error_ = 0, admission_error_ = 0;
public:
    template<class Context>
    SegmentedDriverStorage(std::mutex& mutex, Context context, int device,
            size_t extent, size_t page, uint64_t generation,
            int (*admit)(int, size_t) noexcept)
        : account_{mutex}, driver_(context, device), factory_(driver_, account_, extent, page, device, generation),
          device_(device), admit_(admit), extent_pages_(extent / page) {
        if (!admit_) throw std::invalid_argument("missing canonical admission");
    }
    bool open() noexcept {
        const bool result = factory_.open([]() noexcept { return true; });
        if (!result) driver_error_ = factory_.error();
        return result;
    }
    auto& ledger() { return factory_.ledger(); }
    uintptr_t address() const noexcept { return factory_.address(); }
    size_t va_charge() const noexcept { return account_.va; }
    int admission_error() const noexcept { return admission_error_; }
    bool physical_snapshot(const SpanLedger& ledger, SpanLedger::Snapshot& output) const override {
        return !error() && factory_.storage().physical_snapshot(ledger, output);
    }
    bool materialize(const SpanLedger::Lease& lease, const SpanLedger& ledger,
            std::unique_lock<std::mutex>& lock) noexcept override {
        bool result = factory_.storage().materialize(lease, ledger, [&](size_t bytes) noexcept {
            // Admission may call arena/pool recovery. Never hold pool serialization
            // across that callback; the lease and retiring owner remain retained.
            lock.unlock();
            admission_error_ = bytes ? admit_(device_, bytes) : 0;
            lock.lock();
            if (admission_error_ || bytes > SIZE_MAX - account_.physical) return false;
            account_.physical += bytes; lock.unlock(); return true;
        }, [&](size_t bytes) noexcept {
            // Rollback credit; Driver work in materialize runs unlocked.
            lock.lock(); account_.physical -= bytes; lock.unlock();
        });
        if (!result) driver_error_ = factory_.storage().error();
        if (!lock.owns_lock()) lock.lock();
        return result;
    }
    SegmentedRefusal last_refusal() const noexcept override {
        if (factory_.state() != decltype(factory_)::State::ready) return {};
        const auto& r = factory_.storage().last_refusal();
        SegmentedRefusal out;
        out.observed = r.observed; out.capacity = r.capacity; out.rolled_back = r.rolled_back;
        out.operation = r.operation; out.status = r.status; out.rollback_status = r.rollback_status;
        out.unit = r.unit; out.fresh = r.fresh; out.released = r.released; out.retained = r.retained;
        out.missing = r.missing;
        if (r.operation == 3) {
            const auto& a = driver_.last_access();
            out.access_observed = a.first_observed; out.access_first = a.first;
            out.barrier_observed = a.barrier_observed; out.access_barrier = a.barrier;
            out.continuation_observed = a.continuation_observed; out.access_continuation = a.continuation;
        }
        return out;
    }
    SegmentedUnitKind unit_kind(size_t index) const noexcept override {
        if (factory_.state() != decltype(factory_)::State::ready || index >= extent_pages_)
            return SegmentedUnitKind::unavailable;
        using State = typename ddo_dead_pages::Storage<Driver>::State;
        switch (factory_.storage().unit(index).state) {
        case State::discarded: return SegmentedUnitKind::discarded;
        case State::resident: return SegmentedUnitKind::resident;
        default: return SegmentedUnitKind::retained;
        }
    }
    size_t unit_bytes() const noexcept override {
        return factory_.state() == decltype(factory_)::State::ready ? factory_.storage().page() : 0;
    }
    bool discard(size_t index, const SpanLedger& ledger, std::unique_lock<std::mutex>& lock) noexcept override {
        bool result = factory_.storage().discard(index, ledger, [&](size_t bytes) noexcept {
            lock.lock(); account_.physical -= bytes; lock.unlock();
        }, [&]() noexcept { lock.unlock(); });
        if (!result) driver_error_ = factory_.storage().error();
        if (!lock.owns_lock()) lock.lock();
        return result;
    }
    bool close_va(std::unique_lock<std::mutex>& lock) noexcept override {
        bool result = factory_.storage().close_dead(factory_.ledger(), [&](size_t bytes) noexcept {
            lock.lock(); account_.physical -= bytes; lock.unlock();
        }, [&]() noexcept { lock.unlock(); });
        if (!result) driver_error_ = factory_.storage().error();
        if (result) result = factory_.close();
        if (!lock.owns_lock()) lock.lock();
        return result;
    }
    size_t page_count() const noexcept override { return extent_pages_; }
    size_t physical_charge() const noexcept override { return account_.physical; }
    int error() const noexcept override {
        return driver_error_ ? driver_error_ : factory_.cleanup_error();
    }
};
}
