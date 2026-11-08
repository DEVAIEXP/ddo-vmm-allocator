#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <inttypes.h>

namespace ddo_vmm_detail {
// Failure-only metadata. Caller holds the pool lock; no CUDA calls or ownership
// changes. Overflow explicitly invalidates completeness; never grow the buffer.
template<size_t Capacity = 64>
struct PoolGeometryFailure {
    static_assert(Capacity > 0 && Capacity <= 64);
    static constexpr size_t fields = 21;
    using Row = std::array<uint64_t, fields>;
    std::array<Row, Capacity> rows{};
    size_t count = 0, total = 0;
    bool available = false, overflow = false;

    template<class Owners>
    void capture(const Owners& owners, size_t requested, int device, uintptr_t context) noexcept {
        count = 0;
        total = owners.size();
        available = true;
        overflow = total > Capacity;
        for (const auto& owner : owners) {
            if (count == Capacity) break;
            const auto snapshot = owner->ledger.snapshot();
            const auto geometry = owner->ledger.geometry(requested, owner->device);
            const bool context_match = reinterpret_cast<uintptr_t>(owner->context) == context;
            const bool device_match = owner->device == device;
            const bool eligible = context_match && device_match && !owner->retiring && geometry.eligible;
            rows[count++] = {static_cast<uint64_t>(owner->address), geometry.generation,
                static_cast<uint64_t>(owner->device), context_match, device_match, owner->retiring, eligible,
                snapshot.mapped, snapshot.requested, snapshot.leased, snapshot.available, snapshot.pending,
                snapshot.quarantined, geometry.largest_available, geometry.largest_reusable,
                geometry.fits_available, geometry.fits_reusable, geometry.whole_backing_releasable,
                snapshot.live_count, snapshot.reserved_count, geometry.spans};
        }
    }

    // At most 64*21 uint64 values: serialized output is below 32 KiB.
    bool write(FILE* output, size_t requested, size_t mapped, int device,
               uintptr_t context, uintptr_t stream, int status, int reason) const noexcept {
        if (!output) return false;
        if (std::fprintf(output,
            "{\"event\":\"pool_geometry_failure\",\"schema\":1,\"request_bytes\":%zu,"
            "\"mapped_bytes\":%zu,\"device\":%d,\"context\":%" PRIuPTR ",\"stream\":%" PRIuPTR ","
            "\"status\":%d,\"reason\":%d,\"available\":%s,\"backings_total\":%zu,"
            "\"rows_count\":%zu,\"overflow\":%s,\"rows\":[",
            requested, mapped, device, context, stream, status, reason,
            available ? "true" : "false", total, count, overflow ? "true" : "false") < 0) return false;
        for (size_t row = 0; row < count; ++row) {
            if (std::fputs(row ? ",[" : "[", output) == EOF) return false;
            for (size_t field = 0; field < fields; ++field)
                if (std::fprintf(output, field ? ",%" PRIu64 : "%" PRIu64, rows[row][field]) < 0) return false;
            if (std::fputc(']', output) == EOF) return false;
        }
        return std::fputs("]}\n", output) != EOF;
    }
};
static_assert(sizeof(PoolGeometryFailure<>) <= 12 * 1024);
} // namespace ddo_vmm_detail
