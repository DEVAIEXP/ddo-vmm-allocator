#pragma once
#include "dead_pages.h"
#include <memory>

namespace ddo_dead_pages {
// Externally serialized owner. Account is the caller's canonical VA admission
// interface, not a second budget. Physical admission remains in Storage.
template<class Driver, class Account> class FactoryOwner {
public:
    enum class State { cold, ready, refused, retained_failure, closed };
private:
    Driver& driver_;
    Account& account_;
    size_t extent_, page_;
    int device_;
    uint64_t generation_;
    uintptr_t address_ = 0;
    bool va_charged_ = false;
    State state_ = State::cold;
    int error_ = 0, cleanup_error_ = 0;
    std::unique_ptr<ddo_vmm_detail::SpanLedger> ledger_;
    std::unique_ptr<Storage<Driver>> storage_;
public:
    FactoryOwner(Driver& driver, Account& account, size_t extent, size_t page,
            int device, uint64_t generation) : driver_(driver), account_(account),
            extent_(extent), page_(page), device_(device), generation_(generation) {
        if (!page || page % 512 || !extent || extent % page || extent / page > 4096
            || device < 0 || !generation) throw std::invalid_argument("factory geometry");
    }
    FactoryOwner(const FactoryOwner&) = delete;
    FactoryOwner& operator=(const FactoryOwner&) = delete;
    // Destruction does not hide failed Driver ownership. Caller retains failures.
    template<class Metadata> bool open(Metadata metadata) noexcept {
        static_assert(std::is_nothrow_invocable_r_v<bool, Metadata>);
        if (state_ != State::cold) return false;
        if (!driver_.identity(device_)) { error_ = -2; state_ = State::refused; return false; }
        if (!account_.claim_va(extent_)) { state_ = State::refused; return false; }
        va_charged_ = true;
        error_ = driver_.reserve(extent_, page_, address_);
        if (!error_ && (!address_ || address_ % page_ || extent_ > UINTPTR_MAX - address_)) error_ = -1;
        if (!error_) {
            try {
                if (!metadata()) throw std::bad_alloc();
                ledger_ = std::make_unique<ddo_vmm_detail::SpanLedger>(extent_, device_, generation_);
                storage_ = std::make_unique<Storage<Driver>>(driver_, *ledger_, address_, extent_, page_, device_, generation_);
                state_ = State::ready; return true;
            } catch (...) { error_ = -3; }
        }
        // A nonzero output is retained even on a failed reserve. Do not assume
        // an error means that no resource was created.
        if (address_) cleanup_error_ = driver_.free_va(address_, extent_);
        if (cleanup_error_) { state_ = State::retained_failure; return false; }
        address_ = 0; account_.credit_va(extent_); va_charged_ = false;
        state_ = State::refused; return false;
    }
    // Physical close and completion proof must have succeeded first. The pool
    // commits its single ledger only after this checked VA close returns.
    bool close() noexcept {
        if (state_ != State::ready || !ledger_->release_claimed() || storage_->failed()
            || storage_->charged()) return false;
        cleanup_error_ = driver_.free_va(address_, extent_);
        if (cleanup_error_) { state_ = State::retained_failure; return false; }
        address_ = 0; account_.credit_va(extent_); va_charged_ = false;
        state_ = State::closed; return true;
    }
    State state() const noexcept { return state_; }
    int error() const noexcept { return error_; }
    int cleanup_error() const noexcept { return cleanup_error_; }
    uintptr_t address() const noexcept { return state_ == State::ready ? address_ : 0; }
    uintptr_t retained_address() const noexcept { return address_; }
    bool va_charged() const noexcept { return va_charged_; }
    auto& ledger() { if (state_ != State::ready) throw std::logic_error("owner not ready"); return *ledger_; }
    auto& storage() { if (state_ != State::ready) throw std::logic_error("owner not ready"); return *storage_; }
    const auto& storage() const { if (state_ != State::ready) throw std::logic_error("owner not ready"); return *storage_; }
};
}
