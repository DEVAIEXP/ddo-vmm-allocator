#pragma once

#include <condition_variable>
#include <future>
#include <mutex>

namespace ddo_source_io {

// Caller holds the ring lock. The slot remains busy until the last waiter
// retires; close waits for that retirement before freeing storage/the ring.
// A shared future gives every waiter the same result, including exceptions.
template<class Slot>
auto wait_read_without_ring_lock(Slot& slot, std::unique_lock<std::mutex>& lock,
                                 std::condition_variable& retired) {
    auto completion = slot.file_read;
    ++slot.read_waiters;
    retired.notify_all();
    lock.unlock();
    auto finish = [&] {
        lock.lock();
        if (--slot.read_waiters == 0) slot.read_in_flight = false;
        retired.notify_all();
    };
    try {
        auto result = completion.get();
        finish();
        return result;
    } catch (...) {
        // get() can throw; retirement must still precede storage release.
        if (!lock.owns_lock()) finish();
        throw;
    }
}

} // namespace ddo_source_io
