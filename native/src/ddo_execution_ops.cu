#include "ddo_execution_ops.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <initializer_list>
#include <limits>

namespace {
template<class T> __device__ float read_value(T x) { return float(x); }
template<class T> __device__ T rounded(float x) { return T(x); }
template<> __device__ __nv_bfloat16 rounded(float x) { return __float2bfloat16_rn(x); }

template<class T> __global__ void rms_modulate(
    const T* x, const T* scale, const T* shift, const T* weight, const T* bias,
    T* output, int64_t channels, int64_t repeats, int64_t stride, float eps) {
    __shared__ float sums[256];
    const int64_t row = blockIdx.x;
    float sum = 0.0f;
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        float v = read_value(x[row * channels + c]);
        sum = __fadd_rn(sum, __fmul_rn(v, v));
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (threadIdx.x < step) sums[threadIdx.x] += sums[threadIdx.x + step];
        __syncthreads();
    }
    const float inv = rsqrtf(sums[0] / float(channels) + eps);
    const int64_t mod = (row / repeats) * stride;
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        T normalized = rounded<T>(__fmul_rn(read_value(x[row * channels + c]), inv));
        if (weight) normalized = rounded<T>(__fmul_rn(read_value(normalized), read_value(weight[c])));
        if (bias) normalized = rounded<T>(__fadd_rn(read_value(normalized), read_value(bias[c])));
        T factor = rounded<T>(__fadd_rn(1.0f, read_value(scale[mod + c])));
        T product = rounded<T>(__fmul_rn(read_value(normalized), read_value(factor)));
        output[row * channels + c] = rounded<T>(__fadd_rn(read_value(product), read_value(shift[mod + c])));
    }
}

template<class T> __global__ void rotary_pair(DdoRotaryView q, DdoRotaryView k, bool split) {
    DdoRotaryView v = blockIdx.y == 0 ? q : k;
    const int64_t width = v.heads * v.head_dim;
    const int64_t count = v.batch * v.tokens * width;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t c = i % width;
        int64_t t = (i / width) % v.tokens;
        int64_t b = i / (width * v.tokens);
        int64_t fb = v.freq_batch == 1 ? 0 : b;
        int64_t paired, freq;
        bool negative;
        if (split) {
            int64_t h = c / v.head_dim, d = c % v.head_dim;
            int64_t half = v.head_dim / 2;
            negative = d < half;
            paired = i + (negative ? half : -half);
            freq = ((fb * v.heads + h) * v.tokens + t) * half + d % half;
        } else {
            negative = (c % 2) == 0;
            paired = i + (negative ? 1 : -1);
            freq = (fb * v.tokens + t) * width + c;
        }
        float a = read_value(static_cast<const T*>(v.input)[i]);
        float other = read_value(static_cast<const T*>(v.input)[paired]);
        float product = __fmul_rn(a, v.cos[freq]);
        // Split layout preserves the explicit FP32 addcmul operation.
        float result = split ? fmaf(negative ? -v.sin[freq] : v.sin[freq], other, product)
                             : __fadd_rn(product, __fmul_rn(negative ? -other : other, v.sin[freq]));
        static_cast<T*>(v.output)[i] = rounded<T>(result);
    }
}

bool valid_rotary(DdoRotaryView v) {
    if (!v.input || !v.output || !v.cos || !v.sin || v.input == v.output) return false;
    if (v.batch <= 0 || v.tokens <= 0 || v.heads <= 0 || v.head_dim <= 0 || v.head_dim % 2) return false;
    if (v.freq_batch != 1 && v.freq_batch != v.batch) return false;
    int64_t count = v.batch;
    for (int64_t n : {v.tokens, v.heads, v.head_dim}) {
        if (count > std::numeric_limits<int64_t>::max() / n) return false;
        count *= n;
    }
    return true;
}
}

extern "C" int ddo_execution_ops_abi() { return 1; }
extern "C" int ddo_rms_modulate(
    const void* x, const void* scale, const void* shift, const void* weight, const void* bias,
    void* output, int64_t rows, int64_t channels, int64_t modulation_rows,
    int64_t stride, float eps, int dtype, void* stream) {
    if (!x || !scale || !shift || !output || output == x || (bias && !weight) ||
        rows <= 0 || rows > 2147483647 || channels <= 0 || modulation_rows <= 0 ||
        rows % modulation_rows || stride < channels || !(eps > 0.0f) || !std::isfinite(eps) ||
        rows > std::numeric_limits<int64_t>::max() / channels ||
        modulation_rows > std::numeric_limits<int64_t>::max() / stride || dtype < 0 || dtype > 1)
        return int(cudaErrorInvalidValue);
#define DDO_RMS_LAUNCH(T) rms_modulate<T><<<unsigned(rows), 256, 0, static_cast<cudaStream_t>(stream)>>>( \
    static_cast<const T*>(x), static_cast<const T*>(scale), static_cast<const T*>(shift), \
    static_cast<const T*>(weight), static_cast<const T*>(bias), static_cast<T*>(output), \
    channels, rows / modulation_rows, stride, eps)
    if (dtype == 0) { DDO_RMS_LAUNCH(float); } else { DDO_RMS_LAUNCH(__nv_bfloat16); }
#undef DDO_RMS_LAUNCH
    return int(cudaGetLastError());
}

extern "C" int ddo_rotary_pair(DdoRotaryView q, DdoRotaryView k, int split, int dtype, void* stream) {
    if (!valid_rotary(q) || !valid_rotary(k) || split < 0 || split > 1 || dtype < 0 || dtype > 1)
        return int(cudaErrorInvalidValue);
    dim3 grid(1024, 2);
    if (dtype == 0) rotary_pair<float><<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(q, k, split != 0);
    else rotary_pair<__nv_bfloat16><<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(q, k, split != 0);
    return int(cudaGetLastError());
}
