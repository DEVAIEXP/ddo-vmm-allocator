#pragma once
#include <array>
#include <memory>
#include <stdexcept>

namespace ddo_vmm_detail {

// Dispatcher-owned fixed metadata; no additional I/O workers or source bytes.
// Adoption must precede enqueue, including calls that partially enqueue/throw.
// Retirement requires proof of ALL-chunk completion, even for failed enqueue.
template<class Work, class Status>
class SourceReadFlights {
public:
    static constexpr size_t max_capacity = 8;
    struct Flight { std::shared_ptr<const Work> work; Status submitted{}; };
    explicit SourceReadFlights(size_t limit) : limit_(limit) {
        if (!limit || limit > max_capacity) throw std::invalid_argument("invalid source flight limit");
    }
    bool full() const { return count_ == limit_; }
    bool empty() const { return count_ == 0; }
    size_t size() const { return count_; }
    size_t limit() const { return limit_; }
    Flight* adopt(std::shared_ptr<const Work> work, Status status) noexcept {
        if (!work || full()) return nullptr;
        for (size_t i = 0; i < limit_; ++i) if (!flights_[i].work) {
            flights_[i] = {std::move(work), status};
            ++count_;
            return &flights_[i];
        }
        return nullptr;
    }
    template<class Ready, class Retire>
    size_t reap(Ready&& ready, Retire&& retire) {
        size_t reaped = 0;
        for (size_t i = 0; i < limit_; ++i) {
            auto& flight = flights_[i];
            if (!flight.work || !ready(flight)) continue;
            retire(flight); // on throw preserve the flight and its owner
            flight.work.reset();
            --count_;
            ++reaped;
        }
        return reaped;
    }
private:
    std::array<Flight, max_capacity> flights_{};
    size_t limit_, count_ = 0;
};

} // namespace ddo_vmm_detail
