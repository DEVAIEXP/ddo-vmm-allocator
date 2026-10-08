#pragma once
#include <stdint.h>

#if defined(_WIN32)
#if defined(DDO_EXECUTION_OPS_BUILD)
#define DDO_OP_EXPORT __declspec(dllexport)
#else
#define DDO_OP_EXPORT __declspec(dllimport)
#endif
#else
#define DDO_OP_EXPORT __attribute__((visibility("default")))
#endif

// Experimental ABI 1. Float32 (dtype=0) and bfloat16 (dtype=1) only.
// No allocations, synchronization, tensor retention, or allocator policy.
// All pointers must be device pointers owned by the caller until stream completion.
// Output must not alias any input. The caller validates capacities and device identity.
// Return values are cudaError_t integers; success means submission, not completion.
extern "C" {
DDO_OP_EXPORT int ddo_execution_ops_abi();
// Contiguous [rows, channels]. Scale/shift have modulation_rows rows, with
// channels contiguous and an explicit row stride (in elements). Groups repeat
// consecutively; one modulation row can broadcast across all input rows.
// Optional affine weight/bias use the same dtype; bias requires weight.
DDO_OP_EXPORT int ddo_rms_modulate(
    const void* x, const void* scale, const void* shift,
    const void* weight, const void* bias, void* output,
    int64_t rows, int64_t channels, int64_t modulation_rows,
    int64_t modulation_stride, float eps, int dtype, void* stream);

struct DdoRotaryView {
    const void* input;
    void* output;
    const float* cos;
    const float* sin;
    int64_t batch, tokens, heads, head_dim;
    // Frequencies contiguous: interleaved [freq_batch,tokens,heads*head_dim],
    // split [freq_batch,heads,tokens,head_dim/2]. freq_batch is 1 or batch.
    int64_t freq_batch;
};
// Both Q/K are contiguous [batch,tokens,heads*head_dim]. Different sequence
// lengths are supported. One launch processes both independently, no attention.
DDO_OP_EXPORT int ddo_rotary_pair(
    DdoRotaryView query, DdoRotaryView key, int split, int dtype, void* stream);
}
