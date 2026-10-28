#pragma once

namespace ddo_vmm_detail {
// Preserve driver results. Only an explicit OOM can use existing pressure recovery.
// Callbacks allow CPU contracts without loading a CUDA library or context.
template <typename Result, typename Create, typename Evict>
Result allocation_attempt(Create create, Evict evict, Result out_of_memory,
                          bool allow_pressure) {
    for (;;) {
        const Result result = create();
        if (result != out_of_memory || !allow_pressure || !evict()) return result;
    }
}
} // namespace ddo_vmm_detail
