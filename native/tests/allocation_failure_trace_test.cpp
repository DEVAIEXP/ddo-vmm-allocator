#include "ddo_vmm_allocation_failure_trace.h"
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace {
int calls = 0;
const char* observed_stage = nullptr;
bool observed_exception = false;
char observed_error[1024] = {};
void sink(const char* stage, size_t bytes, int device, uintptr_t stream, int status, bool exception, const char* error) noexcept {
    ++calls;
    observed_stage = stage;
    observed_exception = exception;
    std::snprintf(observed_error, sizeof(observed_error), "%s", error);
    assert(bytes == 1024 && device == 0 && stream == 12 && status == 8);
}
}

int main() {
    using namespace ddo_vmm_detail;
    {
        AllocationFailureTrace trace(1024, 0, 12, sink);
        trace.stage("pressure_prepare_after_eviction", 8);
        assert(trace.finish(nullptr) == nullptr);
    }
    assert(calls == 1 && std::strcmp(observed_stage, "pressure_prepare_after_eviction") == 0);
    {
        AllocationFailureTrace trace(1024, 0, 12, sink);
        int owned = 0;
        assert(trace.finish(&owned) == &owned);
    }
    assert(calls == 1);
    {
        AllocationFailureTrace trace(1024, 0, 12, sink);
        trace.stage("record_insert", 8);
        trace.exception();
    }
    assert(calls == 2 && observed_exception);
    {
        NoopAllocationFailureTrace trace(1024, 0, 12, sink);
        trace.stage("ignored", 8);
        trace.exception();
        trace.finish(nullptr);
    }
    assert(calls == 2);
    {
        AllocationFailureTrace trace(1024, 0, 12, sink);
        char original[] = "cuMemCreate: CUDA_ERROR_OUT_OF_MEMORY";
        trace.region_failure(8, original);
        original[0] = 'X';
    }
    assert(calls == 3);
    assert(std::strcmp(observed_stage, "region_create_no_evictable_weight") == 0);
    assert(std::strcmp(observed_error, "cuMemCreate: CUDA_ERROR_OUT_OF_MEMORY") == 0);
}
