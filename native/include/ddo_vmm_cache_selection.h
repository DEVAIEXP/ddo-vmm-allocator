#pragma once
#include <cstddef>

namespace ddo_vmm_detail {
// Reuse only an identical physical size class. Never give a smaller request
// ownership of an unsplit larger mapping. Device identity is also mandatory.
template <typename Cache>
auto find_exact_cached_allocation(Cache& cache, size_t mapped_size, int device) {
    const auto [first, last] = cache.equal_range(mapped_size);
    for (auto candidate = first; candidate != last; ++candidate) {
        if (candidate->second.device_ordinal == device)
            return candidate;
    }
    return cache.end();
}
} // namespace ddo_vmm_detail
