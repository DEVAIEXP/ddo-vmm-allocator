#include "ddo_vmm_execution_registry.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <thread>


using execution_reclaim::Result;
void require(bool ok) { if (!ok) throw std::runtime_error("pressure CPU contract failed"); }
int main() {
    unsigned cases = 0;
    auto check = [&](bool ok) { require(ok); ++cases; };
    struct Owner { unsigned visits = 0; } first, second;
    execution_pressure::Registry<Owner> registry;
    check(registry.visit([](Owner*) { return Result::success; }) == Result::busy);
    registry.add(&first); registry.add(&second);
    check(registry.visit([&](Owner* owner) {
        ++owner->visits; return owner == &first ? Result::busy : Result::success;
    }) == Result::success && first.visits == 1 && second.visits == 1);
    check(registry.visit([&](Owner* owner) { ++owner->visits; return Result::failed; }) == Result::failed
        && first.visits == 2 && second.visits == 1);
    check(registry.close(&first, [] { return Result::busy; }) == Result::busy);
    check(registry.close(&first, [] { return Result::failed; }) == Result::failed);
    check(registry.visit([&](Owner* owner) {
        require(owner == &first);
        Result other_lookup = Result::failed;
        std::thread other([&] { other_lookup = registry.visit([](Owner*) { return Result::success; }); });
        other.join(); require(other_lookup == Result::busy);
        return Result::success;
    }) == Result::success);

    // A live lookup pins registry lifetime until close can remove the owner.
    std::promise<void> entered, release;
    auto entered_future = entered.get_future(); auto release_future = release.get_future();
    std::atomic<bool> close_entered{false}, close_finished{false};
    Result visit_result = Result::failed, close_result = Result::failed;
    std::thread lookup([&] {
        visit_result = registry.visit([&](Owner* owner) {
            if (owner != &first) return Result::failed;
            entered.set_value();
            return release_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready
                ? Result::success : Result::failed;
        });
    });
    const bool lookup_started = entered_future.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    std::thread closer([&] {
        close_entered = true;
        close_result = registry.close(&first, [&] { close_finished = true; return Result::success; });
    });
    while (!close_entered.load()) std::this_thread::yield();
    const bool protected_while_looked_up = !close_finished.load();
    const auto concurrent_visit = registry.visit([](Owner*) { return Result::success; });
    release.set_value(); lookup.join(); closer.join();
    check(lookup_started && protected_while_looked_up && concurrent_visit == Result::busy);
    check(visit_result == Result::success && close_result == Result::success && close_finished.load());
    check(registry.close(&first, [] { return Result::success; }) == Result::failed);
    check(registry.visit([&](Owner* owner) { return owner == &second ? Result::success : Result::failed; })
        == Result::success);
    check(registry.close(&second, [] { return Result::success; }) == Result::success);
    check(registry.visit([](Owner*) { return Result::success; }) == Result::busy);
    std::printf("{\"status\":\"pass\",\"cases\":%u,\"cuda_loaded\":false}\n", cases);
}
