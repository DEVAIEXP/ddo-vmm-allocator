#pragma once
#include "ddo_vmm_mapping_trace.h"
#include <array>
#include <limits>
#include <mutex>
#include <vector>

namespace ddo_vmm_detail {
// CPU-only storage and call-through contract. No driver calls, waits or policy.
class MappingTrace {
public:
    using Record = ddo_vmm_mapping_record_v1;
    using Header = std::array<uint64_t, 16>;
    int start(size_t capacity, size_t byte_limit, const Header& identity) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if (header_[1] || !capacity || capacity > byte_limit / sizeof(Record)) return 1;
            std::vector<Record> storage(capacity);
            rows_.swap(storage);
            header_ = identity;
            header_[0] = 1; header_[1] = 1; header_[2] = capacity;
            header_[3] = header_[4] = header_[5] = header_[14] = header_[15] = 0;
            return 0;
        } catch (...) { return 1; }
    }
    int stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (header_[1] == 0 || header_[5]) return 1;
        header_[1] = 2;
        return 0;
    }
    int snapshot(uint64_t* header, size_t count, Record* rows, size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!header || count != header_.size() || capacity < header_[3]
            || (header_[3] && !rows)) return 1;
        for (size_t i = 0; i < header_.size(); ++i) header[i] = header_[i];
        for (size_t i = 0; i < header_[3]; ++i) rows[i] = rows_[i];
        return 0;
    }
    template<class Call, class Clock>
    auto measure(Record row, Call&& call, Clock&& clock) {
        row.sequence = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (header_[1] == 1) {
                if (header_[15] == std::numeric_limits<uint64_t>::max()) {
                    ++header_[4];
                } else {
                    row.sequence = ++header_[15];
                    ++header_[5];
                }
            }
        }
        // The driver runs outside the diagnostic lock, exactly once.
        if (!row.sequence) return call();
        row.begin_ticks = clock();
        const auto result = call(); // CUDA C functions do not throw C++ exceptions.
        row.end_ticks = clock();
        row.result = static_cast<uint64_t>(result);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --header_[5];
            if (!row.begin_ticks || row.end_ticks < row.begin_ticks) ++header_[14];
            if (header_[3] == header_[2]) ++header_[4];
            else rows_[header_[3]++] = row;
        }
        return result;
    }
private:
    std::mutex mutex_;
    Header header_{};
    std::vector<Record> rows_;
};
} // namespace ddo_vmm_detail
