#include "ddo_vmm_mapping_trace_core.h"
#include <array>
#include <cstdlib>

using ddo_vmm_detail::MappingTrace;
void require(bool value) { if (!value) std::abort(); }
int main() {
    MappingTrace trace;
    uint64_t time = 10;
    int calls = 0, clock_calls = 0;
    auto clock = [&] { ++clock_calls; return time++; };
    auto success = [&] { ++calls; return 0; };
    MappingTrace::Record row{};
    row.action = 1; row.address = 4096; row.bytes = 8192; row.allocation = 77;
    row.allocation_offset = 2048; row.flags = 5; row.device = UINT64_MAX;
    row.sequence = 99; // Caller state must never enable a disabled recorder.
    require(trace.measure(row, success, clock) == 0 && calls == 1 && clock_calls == 0);
    MappingTrace::Header header{};
    header[6] = 7; header[7] = 123; header[8] = 1000;
    require(trace.start(0, 256, header) == 1);
    require(trace.start(2, 255, header) == 1);
    require(trace.start(2, 256, header) == 0);
    require(trace.start(2, 256, header) == 1);
    require(trace.measure(row, [&] {
        ++calls;
        require(trace.stop() == 1); // Refuse mutation while the driver is active.
        return 7; // Preserve non-success without retry or cleanup.
    }, clock) == 7);
    require(trace.measure(row, success, clock) == 0);
    require(trace.measure(row, success, clock) == 0); // Capacity: preserve rows, report loss.
    MappingTrace::Header output{};
    output.fill(99);
    std::array<MappingTrace::Record, 2> rows{};
    require(trace.snapshot(output.data(), 15, rows.data(), 2) == 1 && output[0] == 99);
    require(trace.snapshot(output.data(), 16, rows.data(), 1) == 1 && output[0] == 99);
    require(trace.snapshot(nullptr, 16, rows.data(), 2) == 1);
    require(trace.snapshot(output.data(), 16, rows.data(), 2) == 0);
    require(output[0] == 1 && output[1] == 1 && output[2] == 2 && output[3] == 2);
    require(output[4] == 1 && output[5] == 0 && output[6] == 7 && output[7] == 123);
    require(rows[0].sequence == 1 && rows[0].result == 7);
    require(rows[0].address == 4096 && rows[0].bytes == 8192 && rows[0].allocation == 77);
    require(rows[0].allocation_offset == 2048 && rows[0].flags == 5);
    require(rows[0].begin_ticks == 10 && rows[0].end_ticks == 11);
    require(rows[1].sequence == 2 && calls == 4 && clock_calls == 6);
    require(trace.stop() == 0 && trace.stop() == 0);
    require(trace.start(2, 256, header) == 1); // Never reset a capture silently.
    require(trace.measure(row, success, clock) == 0 && calls == 5 && clock_calls == 6);
    MappingTrace invalid_clock;
    require(invalid_clock.start(1, 128, header) == 0);
    require(invalid_clock.measure(row, success, [] { return uint64_t{0}; }) == 0);
    require(invalid_clock.snapshot(output.data(), 16, rows.data(), 2) == 0 && output[14] == 1);
    return 0;
}
