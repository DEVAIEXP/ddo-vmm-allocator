#pragma once
#include "ddo_vmm_pool_lifetime_trace.h"

namespace ddo_vmm_detail {
// Separate schema/exports from allocation history. Caller holds the pool mutex.
// Fields: sequence, action, pointer, backing, generation, lease, producer ID,
// consumer ID, identity-known mask, raw stream, event, CUDA status, lifetime cursor.
template<size_t Capacity = 8192>
class PoolDependencyTrace {
public:
    bool start() noexcept { return history_.start(); }
    void record(uint64_t action, uint64_t pointer, uint64_t backing,
                uint64_t generation, uint64_t lease, uint64_t producer,
                uint64_t consumer, uint64_t known, uint64_t stream,
                uint64_t event, uint64_t status, uint64_t lifetime_cursor) noexcept {
        history_.record(action, pointer, backing, generation, lease, producer,
                        consumer, known, stream, event, status, lifetime_cursor);
    }
    int snapshot(uint64_t* report, size_t count, uint64_t* rows, size_t capacity) const noexcept {
        return history_.snapshot(report, count, rows, capacity);
    }
private:
    PoolLifetimeTrace<Capacity> history_;
};
} // namespace ddo_vmm_detail
