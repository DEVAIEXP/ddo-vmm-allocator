#include "ddo_vmm_execution_reclaim.h"
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
using execution_reclaim::Result;
void require(bool ok) { if (!ok) throw std::runtime_error("reclaim CPU contract failed"); }
int main() {
    using execution_reclaim::Pressure;
    require(execution_reclaim::assess(4, 1024, 764, 256) == Pressure::retain);
    require(execution_reclaim::assess(4, 1024, 765, 256) == Pressure::reclaim);
    require(execution_reclaim::assess(4, 1024, 2048, 256) == Pressure::reclaim);
    require(execution_reclaim::assess(0, 1024, 0, 256) == Pressure::unavailable);
    require(execution_reclaim::assess(4, 0, 0, 256) == Pressure::unavailable);
    require(execution_reclaim::assess(UINT64_MAX, UINT64_MAX, 0, 0) == Pressure::retain);
    require(execution_reclaim::assess(UINT64_MAX, UINT64_MAX, 0, 1) == Pressure::reclaim);
    unsigned cases = 0;
    for (const auto query_result : {Result::success, Result::busy, Result::failed}) {
        for (bool leased : {false, true}) {
            bool poisoned = false, materialized = true;
            unsigned queries = 0, releases = 0;
            const auto result = execution_reclaim::run(leased, poisoned, materialized, 2,
                [&] { ++queries; return query_result; },
                [&](size_t) { ++releases; return true; });
            if (leased) require(result == Result::busy && queries == 0 && releases == 0 && materialized);
            else if (query_result == Result::success)
                require(result == Result::success && queries == 1 && releases == 2 && !materialized && !poisoned);
            else require(result == query_result && releases == 0 && materialized
                && poisoned == (query_result == Result::failed));
            ++cases;
        }
    }
    for (size_t fail_at : {size_t(0), size_t(1)}) {
        bool poisoned = false, materialized = true;
        size_t releases = 0;
        auto release = [&](size_t index) { ++releases; return index != fail_at; };
        volatile auto completion = Result::success;
        auto query = [&] { return completion; };
        require(execution_reclaim::run(false, poisoned, materialized, 2, query, release) == Result::failed);
        require(poisoned && materialized && releases == fail_at + 1);
        require(execution_reclaim::run(false, poisoned, materialized, 2, query, release) == Result::failed);
        require(releases == fail_at + 1);
        ++cases;
    }
    bool poisoned = false, materialized = false;
    unsigned unexpected_calls = 0;
    volatile auto completion = Result::success;
    require(execution_reclaim::run(false, poisoned, materialized, 2,
        [&] { ++unexpected_calls; return completion; },
        [&](size_t) { ++unexpected_calls; return true; }) == Result::success);
    require(unexpected_calls == 0);
    std::printf("{\"status\":\"pass\",\"cases\":%u,\"cuda_loaded\":false}\n", cases + 1);
}
