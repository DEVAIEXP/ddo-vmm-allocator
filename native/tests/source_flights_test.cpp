#include "ddo_vmm_source_scheduler.h"
#include "ddo_vmm_source_flights.h"
#include <future>
#include <iostream>
#include <vector>

int main() {
    using namespace ddo_vmm_detail;
    using Flights = SourceReadFlights<SourceReadScheduler::Work, int>;
    int failures = 0;
    auto check = [&](bool value, const char* label) {
        if (!value) { ++failures; std::cerr << label << '\n'; }
    };
    for (int iteration = 0; iteration < 32; ++iteration) {
        SourceReadScheduler scheduler(43, 10, 10);
        Flights flights(2);
        auto owner = std::make_shared<int>(66);
        std::weak_ptr<int> retained = owner;
        std::promise<int> first, second, third;
        auto a = first.get_future().share();
        auto b = second.get_future().share();
        auto c = third.get_future().share();
        auto r1 = scheduler.request({1, 1, 20, 1, 0, 2}, owner, false);
        auto r2 = scheduler.request({1, 1, 20, 1, 2, 2}, owner, false);
        auto r3 = scheduler.request({1, 1, 20, 1, 4, 2}, owner, false);
        auto w1 = scheduler.take();
        auto w2 = scheduler.take();
        auto* f1 = flights.adopt(w1, 0);
        auto* f2 = flights.adopt(w2, 7); // partial enqueue failure is NOT drained
        check(f1 && f2 && flights.full(), "two pending windows admitted without waiting for either");
        check(!flights.adopt(w1, 0) && flights.size() == 2, "fixed flight limit preserved");
        owner.reset(); w1.reset(); w2.reset();
        auto ready = [&](const Flights::Flight& flight) {
            return (flight.work->token.generation == r1.token.generation ? a : b)
                .wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        };
        int retired = 0;
        bool partial_failed = false;
        auto retire = [&](Flights::Flight& flight) {
            auto completion = flight.work->token.generation == r1.token.generation ? a : b;
            auto status = completion.get();
            if (flight.submitted) status = flight.submitted;
            if (status == 7) partial_failed = true;
            scheduler.acknowledge(flight.work->token, status == 0);
            scheduler.cancel(flight.work->token);
            ++retired;
        };
        check(flights.reap(ready, retire) == 0 && retired == 0,
              "failed partial enqueue retains owner until ALL chunks complete");
        second.set_value(0);
        check(flights.reap(ready, retire) == 1 && flights.size() == 1,
              "later completion retires without waiting for oldest");
        SourceReadState state{};
        check(scheduler.state(r1.token, state) && state == SourceReadState::reading,
              "oldest remains owned and reading");
        check(!scheduler.state(r2.token, state), "failed submission is retired, never published ready");
        check(partial_failed, "successful chunk completion cannot erase partial submission failure");
        // Demand promotion is retained when the second slot becomes available.
        scheduler.request({1, 1, 20, 1, 4, 2}, retained.lock(), true);
        auto w3 = scheduler.take();
        check(w3 && w3->token.generation == r3.token.generation, "queued demand admitted to free flight");
        flights.adopt(w3, 0); w3.reset();
        scheduler.begin_close();
        check(!scheduler.drained() && !retained.expired(), "close retains submitted source owners");
        // A retirement callback failure must not discard the flight's owner.
        try { flights.reap([](const auto&) { return true; }, [](auto&) { throw 1; }); }
        catch (int) {}
        check(flights.size() == 2 && !retained.expired(), "failed retirement preserves flight ownership");
        third.set_value(0);
        first.set_value(0);
        flights.reap([](const auto&) { return true; }, [&](auto& flight) {
            auto completion = flight.work->token.generation == r1.token.generation ? a : c;
            completion.get(); // proof precedes acknowledgement on close, too
            scheduler.acknowledge(flight.work->token, false);
            scheduler.cancel(flight.work->token);
        });
        check(flights.empty() && scheduler.drained() && retained.expired(),
              "close releases owners only after all submitted completions are proven");
    }
    // Full native window geometry: no more than eight flights, independent of
    // a larger admitted host cache. All owners still require explicit drain.
    Flights eight(8);
    SourceReadScheduler scheduler(99, 9, 9);
    for (int i = 0; i < 9; ++i)
        scheduler.request({1, 1, 20, 1, static_cast<uint64_t>(i * 2), 2},
                          std::make_shared<int>(i), false);
    for (int i = 0; i < 8; ++i) check(eight.adopt(scheduler.take(), 0) != nullptr, "eight flights admitted");
    check(eight.full() && scheduler.snapshot().queued == 1, "ninth stays queued, no capacity expansion");
    scheduler.begin_close();
    eight.reap([](const auto&) { return true; }, [&](auto& f) { scheduler.acknowledge(f.work->token, true); });
    check(eight.empty() && scheduler.drained(), "eight-flight shutdown drains all owners");
    std::cout << "source flights: 32 lifecycle iterations plus eight-flight bound, failures=" << failures << '\n';
    return failures ? 1 : 0;
}
