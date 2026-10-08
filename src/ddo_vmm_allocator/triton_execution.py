"""Opt-in generic inference kernels; no residency or weight-placement policy.

Import this module only when explicitly requested. Triton is optional. Outputs
are PyTorch allocations and launches use PyTorch's current CUDA stream. Inputs
remain owned by the calling forward; no private stream, caching or retention.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _modulate(Norm, Scale, Shift, Out,
                  TOKENS: tl.constexpr, CHANNELS: tl.constexpr,
                  XS0: tl.constexpr, XS1: tl.constexpr,
                  SB: tl.constexpr, ST: tl.constexpr, SS0: tl.constexpr, SS1: tl.constexpr,
                  HB: tl.constexpr, HT: tl.constexpr, HS0: tl.constexpr, HS1: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    batch, token = row // TOKENS, row % TOKENS
    channels = tl.arange(0, BLOCK)
    valid = channels < CHANNELS
    norm = tl.load(Norm + batch * XS0 + token * XS1 + channels, valid, 0).to(tl.float32)
    dtype = Out.dtype.element_ty
    # Normalization/affine arithmetic already ran through the original norm.
    # Preserve the separately rounded eager modulation stages.
    scale = tl.load(Scale + (batch % SB) * SS0 + (token % ST) * SS1 + channels, valid, 0).to(tl.float32)
    shift = tl.load(Shift + (batch % HB) * HS0 + (token % HT) * HS1 + channels, valid, 0).to(tl.float32)
    factor = (1.0 + scale).to(dtype).to(tl.float32)
    product = (norm * factor).to(dtype).to(tl.float32)
    tl.store(Out + row * CHANNELS + channels, (product + shift).to(dtype), valid)


@triton.jit
def _rotary_one(X, Cos, Sin, Out, B: tl.constexpr, T: tl.constexpr,
                H: tl.constexpr, D: tl.constexpr, FB: tl.constexpr,
                CS0: tl.constexpr, CS1: tl.constexpr, CS2: tl.constexpr, CS3: tl.constexpr,
                SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr, SS3: tl.constexpr,
                SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    width = H * D
    valid = indices < B * T * width
    channel = indices % width
    token = (indices // width) % T
    batch = indices // (width * T)
    if SPLIT:
        head, dim = channel // D, channel % D
        negative = dim < D // 2
        partner = indices + tl.where(negative, D // 2, -D // 2)
        frequency = dim % (D // 2)
        ci = (batch % FB) * CS0 + head * CS1 + token * CS2 + frequency * CS3
        si = (batch % FB) * SS0 + head * SS1 + token * SS2 + frequency * SS3
    else:
        negative = channel % 2 == 0
        partner = indices + tl.where(negative, 1, -1)
        ci = (batch % FB) * CS0 + token * CS1 + channel * CS2
        si = (batch % FB) * SS0 + token * SS1 + channel * SS2
    value = tl.load(X + indices, valid, 0).to(tl.float32)
    other = tl.load(X + partner, valid, 0).to(tl.float32)
    cos = tl.load(Cos + ci, valid, 0).to(tl.float32)
    sin = tl.load(Sin + si, valid, 0).to(tl.float32)
    product = value * cos
    if SPLIT:
        result = tl.fma(tl.where(negative, -sin, sin), other, product)
    else:
        result = product + tl.where(negative, -other, other) * sin
    tl.store(Out + indices, result.to(Out.dtype.element_ty), valid)


@triton.jit
def _rotary_pair(Q, QC, QS, QO, K, KC, KS, KO,
                 QB: tl.constexpr, QT: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
                 QFB: tl.constexpr, QCS: tl.constexpr, QSS: tl.constexpr,
                 KB: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
                 KFB: tl.constexpr, KCS: tl.constexpr, KSS: tl.constexpr,
                 SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    if tl.program_id(1) == 0:
        _rotary_one(Q, QC, QS, QO, QB, QT, QH, QD, QFB,
                    QCS[0], QCS[1], QCS[2], QCS[3], QSS[0], QSS[1], QSS[2], QSS[3], SPLIT, BLOCK)
    else:
        _rotary_one(K, KC, KS, KO, KB, KT, KH, KD, KFB,
                    KCS[0], KCS[1], KCS[2], KCS[3], KSS[0], KSS[1], KSS[2], KSS[3], SPLIT, BLOCK)


def _activation(x):
    if x.ndim != 3 or x.dtype not in (torch.float32, torch.bfloat16) or not x.is_cuda:
        raise ValueError("Execution kernels require CUDA FP32/BF16 [batch, tokens, channels]")
    if not all(x.shape) or x.stride(-1) != 1 or torch.is_grad_enabled():
        raise ValueError("Execution kernels require nonempty channel-contiguous inference tensors")


def _modulation_shape(tensor, x):
    if tensor.device != x.device or tensor.dtype != x.dtype or tensor.ndim != 3:
        raise ValueError("Modulation must have the activation device/dtype and rank 3")
    if tensor.shape[0] not in (1, x.shape[0]) or tensor.shape[1] not in (1, x.shape[1]):
        raise ValueError("Modulation batch/token dimensions must match or broadcast")
    if tensor.shape[2] != x.shape[2] or tensor.stride(-1) != 1:
        raise ValueError("Modulation channels must match and be contiguous")
    return tensor.shape[:2], tensor.stride()[:2]


def _rotary_layout(x, freqs, layout):
    cos, sin = freqs
    if layout not in ("interleaved", "split"):
        raise ValueError("Unsupported rotary layout")
    rank = 4 if layout == "split" else 3
    if cos.ndim != rank or sin.shape != cos.shape or sin.dtype != cos.dtype:
        raise ValueError("Rotary frequency shapes/dtypes must agree with the layout")
    if cos.device != x.device or sin.device != x.device or cos.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Rotary frequencies require the activation device and FP32/BF16 dtype")
    if cos.shape[0] not in (1, x.shape[0]):
        raise ValueError("Rotary frequency batch must match or broadcast")
    if layout == "split":
        heads, tokens, half = cos.shape[1:]
        dim = half * 2
        if tokens != x.shape[1] or heads * dim != x.shape[2]:
            raise ValueError("Split rotary geometry does not match activation")
    else:
        heads, dim = 1, x.shape[2]
        if tuple(cos.shape[1:]) != tuple(x.shape[1:]) or dim % 2:
            raise ValueError("Interleaved rotary geometry does not match activation")
    return (x.shape[0], x.shape[1], heads, dim, cos.shape[0],
            tuple(cos.stride()) + (0,) * (4 - rank), tuple(sin.stride()) + (0,) * (4 - rank))


class TritonExecutionOps:
    """Inference-only callbacks with submission counters, not completion metrics."""

    def __init__(self):
        self.rms_submissions = 0
        self.rotary_pair_submissions = 0

    def rms_modulate(self, norm, x, scale, shift):
        """Keep the original norm forward; fuse only its subsequent modulation."""
        _activation(x)
        if norm.training or not math.isfinite(norm.eps) or norm.eps <= 0:
            raise ValueError("Fused RMS modulation requires an eval norm and finite positive eps")
        (sb, st), (ss0, ss1) = _modulation_shape(scale, x)
        (hb, ht), (hs0, hs1) = _modulation_shape(shift, x)
        for parameter in (norm.weight, norm.bias):
            if parameter is not None and (parameter.shape != (x.shape[2],) or
                    parameter.dtype != x.dtype or parameter.device != x.device or not parameter.is_contiguous()):
                raise ValueError("Fused norm affine parameters must match activation channels/device/dtype")
        with torch.cuda.device(x.device):
            normalized = norm(x)
            if normalized.shape != x.shape or normalized.dtype != x.dtype or normalized.device != x.device:
                raise ValueError("Original norm must preserve activation shape/device/dtype")
            _activation(normalized)
            output = torch.empty_like(normalized, memory_format=torch.contiguous_format)
            _modulate[(x.shape[0] * x.shape[1],)](
                normalized, scale, shift, output,
                x.shape[1], x.shape[2], normalized.stride(0), normalized.stride(1),
                sb, st, ss0, ss1, hb, ht, hs0, hs1,
                triton.next_power_of_2(x.shape[2]), enable_fp_fusion=False)
        self.rms_submissions += 1
        return output

    def rotary_pair(self, query, key, query_freqs, key_freqs, layout):
        _activation(query)
        _activation(key)
        if query.device != key.device or query.dtype != key.dtype or not query.is_contiguous() or not key.is_contiguous():
            raise ValueError("Paired rotary requires contiguous Q/K with matching device/dtype")
        qlayout = _rotary_layout(query, query_freqs, layout)
        klayout = _rotary_layout(key, key_freqs, layout)
        qout, kout = torch.empty_like(query), torch.empty_like(key)
        grid = (triton.cdiv(max(query.numel(), key.numel()), 256), 2)
        with torch.cuda.device(query.device):
            _rotary_pair[grid](query, *query_freqs, qout, key, *key_freqs, kout,
                               *qlayout, *klayout, layout == "split", 256, enable_fp_fusion=False)
        self.rotary_pair_submissions += 1
        return qout, kout

    def snapshot(self):
        return {"backend": "triton", "rms_modulate_submissions": self.rms_submissions,
                "rotary_pair_submissions": self.rotary_pair_submissions,
                "normalization_backend": "original_norm_forward",
                "modulation_backend": "triton"}
