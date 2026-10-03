#include "ddo_vmm_source_scheduler.h"

#include <cstdio>
#include <stdexcept>

using namespace ddo_vmm_detail;

static int checks = 0, failures = 0;
static void require(bool value, const char* name) {
    ++checks;
    if (!value) { ++failures; std::printf("FAIL: %s\n", name); }
}
static SourceWindowKey window(uint64_t offset) { return {7, 9, 1024, 11, offset, 64}; }
static auto owner() { return std::make_shared<const int>(42); }

int main() {
    SourceReadState observed{};
    {
        SourceReadScheduler scheduler(1, 2, 4);
        auto a = scheduler.request(window(0), owner(), false);
        auto b = scheduler.request(window(64), owner(), false);
        require(a.status == SourceReadAdmission::accepted && b.status == SourceReadAdmission::accepted, "bounded insertion");
        require(scheduler.request(window(128), owner(), false).status == SourceReadAdmission::deferred, "speculative backpressure without blocking");
        auto duplicate = scheduler.request(window(64), owner(), true);
        require(duplicate.status == SourceReadAdmission::existing && duplicate.token.generation == b.token.generation, "deduplicate and promote changed-order demand");
        auto next = scheduler.take();
        require(next && next->token.generation == b.token.generation, "demand precedes earlier speculation");
        require(scheduler.acknowledge(next->token, true), "read completion proof");
        require(scheduler.state(b.token, observed) && observed == SourceReadState::ready, "completed read remains reusable");
        require(!scheduler.acknowledge(b.token, true), "duplicate acknowledgement rejected");
        auto c = scheduler.request(window(128), owner(), false);
        auto demand = scheduler.request(window(192), owner(), true);
        require(demand.status == SourceReadAdmission::accepted, "demand displaces only queued speculation");
        require(!scheduler.state(c.token, observed) && scheduler.state(a.token, observed), "newest queued speculation cancelled, older preserved");
        auto work = scheduler.take();
        require(work && work->token.generation == demand.token.generation, "changed order consumed without architectural assumption");
        require(!scheduler.acknowledge({99, work->token.generation}, true), "foreign owner rejected");
        require(scheduler.acknowledge(work->token, false), "failed read completion retained");
        require(scheduler.state(demand.token, observed) && observed == SourceReadState::failed, "read failure not ready");
        require(scheduler.request(window(192), owner(), true).token.generation == demand.token.generation, "no silent retry of failed source");
        require(scheduler.cancel(demand.token), "explicit failed-source removal");
        auto retry = scheduler.request(window(192), owner(), true);
        require(retry.token.generation != demand.token.generation, "retry receives new generation");
        require(!scheduler.cancel(demand.token), "stale cancellation cannot steal retry");
    }
    {
        SourceReadScheduler scheduler(2, 1, 1);
        auto a = scheduler.request(window(0), owner(), true);
        require(scheduler.request(window(64), owner(), true).status == SourceReadAdmission::deferred, "full demand queue preserves required source and fallback");
        auto work = scheduler.take();
        require(scheduler.request(window(64), owner(), true).status == SourceReadAdmission::deferred, "entry ceiling includes running work");
        scheduler.acknowledge(a.token, true);
        require(scheduler.request(window(64), owner(), true).status == SourceReadAdmission::deferred, "entry ceiling includes ready ownership");
        scheduler.cancel(a.token);
        require(scheduler.request(window(64), owner(), true).status == SourceReadAdmission::accepted, "capacity returns only after owner retirement");
    }
    {
        SourceReadScheduler scheduler(3, 2, 2);
        auto retained = owner();
        std::weak_ptr<const int> weak = retained;
        auto a = scheduler.request(window(0), retained, true);
        retained.reset();
        auto work = scheduler.take();
        scheduler.cancel(a.token); // Simulated partial enqueue failure, chunks still active.
        require(!weak.expired() && scheduler.snapshot().reading == 1, "partial enqueue/cancellation retains active read storage");
        scheduler.begin_close();
        require(!scheduler.drained() && !weak.expired(), "close does not free active workers");
        require(!scheduler.take(), "close schedules no new work");
        require(scheduler.request(window(64), owner(), true).status == SourceReadAdmission::closed, "closed admission rejected");
        require(scheduler.acknowledge(a.token, false) && scheduler.drained(), "all-chunk completion drains cancelled read");
        require(!weak.expired(), "worker lease outlives ledger acknowledgement");
        work.reset();
        require(weak.expired(), "last read owner releases after proven completion");
        require(!scheduler.state(a.token, observed), "cancelled read cannot publish ready after close");
    }
    {
        SourceReadScheduler scheduler(7, 1, 1);
        auto a = scheduler.request(window(0), owner(), false);
        auto work = scheduler.take();
        scheduler.cancel(a.token);
        auto demand = scheduler.request(window(0), owner(), true);
        require(demand.status == SourceReadAdmission::existing
            && demand.token.generation == a.token.generation, "cancelled active speculation reclaimed by actual demand without duplicate I/O");
        require(!scheduler.take(), "reclaimed reading work not submitted twice");
        require(scheduler.acknowledge(work->token, true), "reclaimed worker completion acknowledged");
        require(scheduler.state(a.token, observed) && observed == SourceReadState::ready,
            "reclaimed demand survives earlier cancellation");
    }
    {
        SourceReadScheduler scheduler(8, 1, 1);
        auto a = scheduler.request(window(0), owner(), false);
        auto work = scheduler.take();
        scheduler.cancel(a.token);
        scheduler.request(window(0), owner(), true);
        scheduler.acknowledge(work->token, false);
        require(scheduler.state(a.token, observed) && observed == SourceReadState::failed,
            "reclaimed failed read remains failed for demand fallback");
    }
    {
        SourceReadScheduler scheduler(4, 2, 2);
        auto retained = owner();
        std::weak_ptr<const int> weak = retained;
        scheduler.request(window(0), retained, false);
        retained.reset();
        scheduler.begin_close();
        require(scheduler.drained() && weak.expired(), "queued work closes without pretending I/O was submitted");
        scheduler.begin_close();
        require(scheduler.drained(), "close is idempotent");
    }
    {
        SourceReadScheduler scheduler(5, 2, 2);
        auto original = window(0), changed = original;
        ++changed.modified;
        auto a = scheduler.request(original, owner(), false);
        auto b = scheduler.request(changed, owner(), false);
        require(a.token.generation != b.token.generation, "immutable file version is part of deduplication identity");
        try { scheduler.request({7, 9, 8, 11, 7, 2}, owner(), true); require(false, "range validation"); }
        catch (const std::invalid_argument&) { require(true, "range validation"); }
        try { scheduler.request(window(0), {}, true); require(false, "owner validation"); }
        catch (const std::invalid_argument&) { require(true, "owner validation"); }
        require(!SourceWindowKey{7, 9, 8, 11, UINT64_MAX, 2}.valid(), "offset overflow rejected");
        require(!SourceWindowKey{7, 9, 8, 11, 0, 0}.valid(), "zero window rejected");
    }
    for (int round = 0; round < 32; ++round) {
        SourceReadScheduler scheduler(6, 2, 3);
        auto a = scheduler.request(window(0), owner(), false);
        auto b = scheduler.request(window(64), owner(), true);
        auto work = scheduler.take();
        require(work->token.generation == b.token.generation, "bounded demand progress");
        require(scheduler.acknowledge(work->token, true), "bounded demand completion");
        scheduler.cancel(b.token);
        auto remaining = scheduler.take();
        require(remaining->token.generation == a.token.generation, "speculation makes progress after demand");
        scheduler.acknowledge(remaining->token, true);
        scheduler.begin_close();
        require(scheduler.drained(), "repeated lifecycle drains");
    }
    std::printf("Native source scheduler CPU contracts: %s (%d checks, %d failures)\n",
        failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
