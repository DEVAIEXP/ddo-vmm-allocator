#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace ddo_vmm_detail {
// Caller serializes record/snapshot/start with the pool mutex. No CUDA, heap
// allocation, file I/O, event waits or allocator policy belongs here.
template<size_t Capacity = 8192>
class PoolLifetimeTrace {
public:
    static constexpr size_t fields = 13;
    using Row = std::array<uint64_t, fields>;
    static_assert(Capacity > 0);

    bool start() noexcept {
        if (enabled_) return false;
        enabled_ = true;
        return true;
    }
    uint64_t total_sequence() const noexcept { return sequence_; }
    // Action 6: an existing failed allocation/admission call, not a fence failure.
    // offset=reason (1 admission, 2 region create), requested=rounded bytes,
    // generation=device ordinal, status=DDO status (not a CUresult).
    void trigger(uint64_t reason, uint64_t requested, uint64_t device,
                 uint64_t stream, uint64_t status) noexcept {
        record(6, 0, 0, 0, reason, 0, requested, device, 0, stream, 0, status);
    }
    void record(uint64_t action, uint64_t pointer, uint64_t backing, uint64_t mapped,
        uint64_t offset, uint64_t size, uint64_t requested, uint64_t generation,
        uint64_t lease, uint64_t stream_handle, uint64_t fence, uint64_t status) noexcept {
        if (!enabled_ || sequence_ == UINT64_MAX) return;
        const auto sequence = ++sequence_;
        rows_[(sequence - 1) % Capacity] = {sequence, action, pointer, backing, mapped,
            offset, size, requested, generation, lease, stream_handle, fence, status};
    }
    // Oldest retained row first. report[0..3]: schema, enabled, total, retained.
    // A cursor older than the first row has lost evidence, never implicit zero.
    int snapshot(uint64_t* report, size_t report_count, uint64_t* rows,
                 size_t row_capacity) const noexcept {
        const size_t retained = sequence_ < Capacity ? static_cast<size_t>(sequence_) : Capacity;
        if (!report || report_count != 4 || row_capacity < retained || (retained && !rows)) return 1;
        report[0] = 1; report[1] = enabled_; report[2] = sequence_; report[3] = retained;
        const uint64_t first = sequence_ - retained + 1;
        for (size_t index = 0; index < retained; ++index)
            for (size_t field = 0; field < fields; ++field)
                rows[index * fields + field] = rows_[(first + index - 1) % Capacity][field];
        return 0;
    }
private:
    std::array<Row, Capacity> rows_{};
    uint64_t sequence_ = 0;
    bool enabled_ = false;
};
} // namespace ddo_vmm_detail
