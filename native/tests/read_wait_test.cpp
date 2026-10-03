#include "ddo_vmm_read_wait.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

struct Slot {
    std::shared_future<int> file_read;
    bool read_in_flight = true;
    size_t read_waiters = 0;
    bool busy() const { return read_in_flight || read_waiters; }
};

int main() {
    using namespace std::chrono_literals;
    int failures = 0;
    auto check = [&](bool value, const char* name) {
        if (!value) { ++failures; std::cerr << name << '\n'; }
    };
    // Repeat success and partial-submission-style failure. The future is not
    // ready while foreground admissions obtain the SAME ring mutex.
    for (int iteration = 0; iteration < 32; ++iteration) {
        std::mutex mutex;
        std::condition_variable changed;
        std::promise<int> completion;
        Slot slot{completion.get_future().share()};
        int results[2] = {};
        bool exceptions[2] = {};
        std::thread waiters[2];
        for (int i = 0; i < 2; ++i) {
            waiters[i] = std::thread([&, i] {
                std::unique_lock lock(mutex);
                try {
                    results[i] = ddo_source_io::wait_read_without_ring_lock(slot, lock, changed);
                } catch (const std::runtime_error&) { exceptions[i] = true; }
            });
        }
        {
            std::unique_lock lock(mutex);
            check(changed.wait_for(lock, 5s, [&] { return slot.read_waiters == 2; }),
                  "both waiters must release the global admission lock");
            check(slot.busy(), "pending read cannot be recycled/unregistered");
        }
        std::promise<void> close_entered;
        bool closed = false;
        std::thread closer([&] {
            std::unique_lock lock(mutex);
            close_entered.set_value();
            changed.wait(lock, [&] { return slot.read_waiters == 0; });
            closed = true;
        });
        close_entered.get_future().wait();
        {
            std::lock_guard lock(mutex);
            check(!closed, "close must retain storage and mutex while waiters own them");
            check(slot.read_waiters == 2, "close cannot retire active waiters");
        }
        const bool fail = iteration % 2;
        if (fail) completion.set_exception(std::make_exception_ptr(std::runtime_error("partial read")));
        else completion.set_value(43);
        for (auto& waiter : waiters) waiter.join();
        closer.join();
        check(closed && !slot.busy(), "all completions must retire before release");
        check(exceptions[0] == fail && exceptions[1] == fail,
              "concurrent waiters must observe the same failure");
        if (!fail) check(results[0] == 43 && results[1] == 43, "shared result must not be consumed");
        // Fresh generation only after retirement; old completion cannot clear it.
        std::promise<int> next;
        slot.file_read = next.get_future().share();
        slot.read_in_flight = true;
        next.set_value(66);
        std::unique_lock lock(mutex);
        check(ddo_source_io::wait_read_without_ring_lock(slot, lock, changed) == 66,
              "fresh generation must return its own payload");
        check(!slot.busy(), "fresh generation must retire");
    }
    std::cout << "read wait: 32 lifecycle iterations, failures=" << failures << '\n';
    return failures ? 1 : 0;
}
