// CPU contracts for the checked-allocation policy (no Torch, CUDA or native DLL).
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "checked_allocation.h"

using namespace ddo_vmm_bridge;

namespace {
void report_for(std::uint64_t* report, std::uint64_t bytes, std::uint64_t termination) {
    for (std::size_t i = 0; i < kReportFields; ++i) report[i] = 0;
    report[kRecorded] = 1;
    report[kRequest] = bytes;
    report[kTermination] = termination;
}

bool contains(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }
}  // namespace

int main() {
    int marker = 0, calls = 0;
    void* pointer = checked_allocation([&](int value) -> void* { ++calls; return value == 1 ? &marker : nullptr; },
                                       []() { throw std::logic_error("must not reject"); }, 1);
    assert(pointer == &marker && calls == 1);

    calls = 0;
    bool thrown = false;
    try {
        checked_allocation([&]() -> void* { ++calls; return nullptr; }, []() { throw std::runtime_error("refused"); });
    } catch (const std::runtime_error&) { thrown = true; }
    assert(thrown && calls == 1);

    std::uint64_t report[kReportFields];
    for (std::uint64_t termination : {2ull, 3ull, 4ull, 8ull}) {
        report_for(report, 360448000, termination);
        const auto refusal = classify_refusal(360448000, 0, 0, report);
        assert(refusal.capacity && contains(refusal.message, "343.8 MiB"));
        assert(contains(refusal.message, termination_name(termination)));
    }
    for (std::uint64_t termination : {1ull, 5ull, 6ull, 7ull, 99ull}) {
        report_for(report, 360448000, termination);
        const auto refusal = classify_refusal(360448000, 0, 0, report);
        assert(!refusal.capacity && contains(refusal.message, "not a capacity refusal"));
    }
    report_for(report, 1622016000, 3);  // Stale report for another request size.
    assert(!classify_refusal(360448000, 0, 0, report).capacity);
    report_for(report, 360448000, 3);
    report[kRecorded] = 0;  // Nothing recorded yet.
    assert(!classify_refusal(360448000, 0, 0, report).capacity);
    report_for(report, 360448000, 3);
    assert(!classify_refusal(360448000, 0, 2, report).capacity);  // Lock busy.
    assert(!classify_refusal(360448000, 0, -1, nullptr).capacity);  // Export missing.
    std::puts("Checked allocation CPU contracts passed");
    return 0;
}
