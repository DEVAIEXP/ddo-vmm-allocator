#pragma once
#include <cstddef>

namespace ddo_vmm_detail {
// Observation only: does not select cache entries or change allocation sizes.
struct AllocationFootprint {
    size_t requested = 0;
    size_t minimum_mapped = 0;
    size_t mapped = 0;

    bool valid() const { return requested <= minimum_mapped && minimum_mapped <= mapped; }
    size_t granularity_padding() const { return minimum_mapped - requested; }
    size_t reuse_excess() const { return mapped - minimum_mapped; }
};
} // namespace ddo_vmm_detail
