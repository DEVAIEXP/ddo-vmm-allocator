#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace ddo_vmm_detail {
using AllocationFailureSink = void (*)(const char*, size_t, int, uintptr_t, int, bool, const char*) noexcept;

// Host-only diagnostic state. No CUDA calls, retries, ownership or policy changes.
class AllocationFailureTrace {
public:
    AllocationFailureTrace(size_t bytes, int device, uintptr_t stream, AllocationFailureSink sink)
        : bytes_(bytes), device_(device), stream_(stream), sink_(sink) {}
    ~AllocationFailureTrace() {
        if (!succeeded_ && sink_) sink_(stage_, bytes_, device_, stream_, status_, exception_, error_);
    }
    void stage(const char* value, int status = 0) noexcept { stage_ = value; status_ = status; }
    // Copy before eviction or another region call can overwrite thread-local error text.
    void region_failure(int status, const char* error) noexcept {
        stage_ = "region_create_no_evictable_weight";
        status_ = status;
        std::snprintf(error_, sizeof(error_), "%s", error ? error : "");
    }
    void exception() noexcept { exception_ = true; }
    void* finish(void* pointer) noexcept { succeeded_ = pointer != nullptr; return pointer; }
private:
    size_t bytes_;
    int device_;
    uintptr_t stream_;
    AllocationFailureSink sink_;
    const char* stage_ = "entry";
    int status_ = 0;
    bool exception_ = false;
    bool succeeded_ = false;
    char error_[1024] = {};
};

struct NoopAllocationFailureTrace {
    NoopAllocationFailureTrace(size_t, int, uintptr_t, AllocationFailureSink) {}
    void stage(const char*, int = 0) noexcept {}
    void region_failure(int, const char*) noexcept {}
    void exception() noexcept {}
    void* finish(void* pointer) noexcept { return pointer; }
};
} // namespace ddo_vmm_detail
