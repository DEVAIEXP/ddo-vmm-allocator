"""Experimental independent fused RMSNorm/split-RoPE compute operations.

Opt-in only: no residency policy, private stream, source retention or imports
from the external benchmark kernel. CUDA execution needs a separate gate.
"""
import math

import torch
import triton
import triton.language as tl
from triton.language.extra import cuda


@triton.jit
def _grouped_square_sum(values, WIDTH: tl.constexpr):
    # Explicit logical lanes make accumulation order independent of tl.sum's
    # general tree. This is a prototype arithmetic contract, not a claim about
    # physical lane placement or compiled equivalence.
    lanes = tl.arange(0, 128)
    accumulated = tl.full((128,), 0, tl.float32)
    for group in tl.static_range(WIDTH // 512):
        for member in tl.static_range(4):
            value = tl.gather(values, group * 512 + lanes * 4 + member, axis=0)
            accumulated = tl.fma(value, value, accumulated)
    for shift in tl.static_range(5):
        distance = 16 >> shift
        other = tl.gather(accumulated, tl.minimum(lanes + distance, 127), axis=0)
        accumulated = accumulated + tl.where(lanes % 32 + distance < 32, other, 0.0)
    # Combine warp leaders in a prescribed two-level order. Only logical lane
    # zero is retained; the final masked reduction adds zeros, not partial sums.
    upper = tl.gather(accumulated, tl.minimum(lanes + 64, 127), axis=0)
    accumulated = accumulated + upper
    upper = tl.gather(accumulated, tl.minimum(lanes + 32, 127), axis=0)
    accumulated = accumulated + upper
    return tl.sum(tl.where(lanes == 0, accumulated, 0.0), axis=0)


@triton.jit
def _normalize_row(Input, Weight, offset, WIDTH: tl.constexpr, EPS: tl.constexpr):
    columns = tl.arange(0, WIDTH)
    values = tl.load(Input + offset + columns).to(tl.float32)
    inverse = cuda.libdevice.rsqrt(_grouped_square_sum(values, WIDTH) / WIDTH + EPS)
    weights = tl.load(Weight + columns).to(tl.float32)
    # PyTorch Q/K affine output rounds only after normalization and weight.
    return ((values * inverse) * weights).to(Input.dtype.element_ty).to(tl.float32)


@triton.jit
def _rotate_row(normalized, cosine, sine, WIDTH: tl.constexpr, HEADS: tl.constexpr):
    half: tl.constexpr = WIDTH // HEADS // 2
    first, second = tl.split(normalized.reshape(HEADS, 2, half).permute(0, 2, 1))
    cosine = cosine.reshape(HEADS, half)
    sine = sine.reshape(HEADS, half)
    first_out = tl.fma(-sine, second, first * cosine)
    second_out = tl.fma(sine, first, second * cosine)
    return tl.join(first_out, second_out).permute(0, 2, 1).reshape(WIDTH)


@triton.jit
def _single(Input, Weight, Cos, Sin, Output, TOKENS: tl.constexpr,
            WIDTH: tl.constexpr, HEADS: tl.constexpr, FB: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    batch, token = row // TOKENS, row % TOKENS
    frequencies = tl.arange(0, WIDTH // 2)
    table_offset = ((batch % FB) * TOKENS + token) * (WIDTH // 2)
    cosine = tl.load(Cos + table_offset + frequencies)
    sine = tl.load(Sin + table_offset + frequencies)
    normalized = _normalize_row(Input, Weight, row * WIDTH, WIDTH, EPS)
    rotated = _rotate_row(normalized, cosine, sine, WIDTH, HEADS)
    tl.store(Output + row * WIDTH + tl.arange(0, WIDTH), rotated.to(Output.dtype.element_ty))


@triton.jit
def _paired(Query, Key, QWeight, KWeight, Cos, Sin, QOut, KOut,
            TOKENS: tl.constexpr, WIDTH: tl.constexpr, HEADS: tl.constexpr,
            FB: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    batch, token = row // TOKENS, row % TOKENS
    frequencies = tl.arange(0, WIDTH // 2)
    table_offset = ((batch % FB) * TOKENS + token) * (WIDTH // 2)
    # One frequency load expression per table, reused by both Q/K branches.
    cosine = tl.load(Cos + table_offset + frequencies)
    sine = tl.load(Sin + table_offset + frequencies)
    qnorm = _normalize_row(Query, QWeight, row * WIDTH, WIDTH, EPS)
    knorm = _normalize_row(Key, KWeight, row * WIDTH, WIDTH, EPS)
    qrotated = _rotate_row(qnorm, cosine, sine, WIDTH, HEADS)
    krotated = _rotate_row(knorm, cosine, sine, WIDTH, HEADS)
    columns = tl.arange(0, WIDTH)
    tl.store(QOut + row * WIDTH + columns, qrotated.to(QOut.dtype.element_ty))
    tl.store(KOut + row * WIDTH + columns, krotated.to(KOut.dtype.element_ty))


def _validate(inputs, weight, cos, sin, heads, eps):
    if (inputs.ndim != 3 or inputs.dtype != torch.bfloat16 or not inputs.is_cuda or
            not inputs.is_contiguous() or not all(inputs.shape) or torch.is_grad_enabled()):
        raise ValueError("Fused Q/K requires contiguous, nonempty CUDA BF16 inference inputs")
    batch, tokens, width = inputs.shape
    if width not in (2048, 4096) or heads != 32 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Experimental Q/K geometry is restricted to widths 2048/4096 and 32 heads")
    if (weight.shape != (width,) or weight.dtype != inputs.dtype or weight.device != inputs.device or
            not weight.is_contiguous()):
        raise ValueError("Q/K weights must match the live activation width/device/dtype")
    for table in (cos, sin):
        if (table.ndim != 4 or table.shape[0] not in (1, batch) or
                table.shape[1:] != (tokens, heads, width // heads // 2) or
                table.dtype != torch.float32 or table.device != inputs.device or not table.is_contiguous()):
            raise ValueError("Q/K frequency tables require contiguous FP32 [batch,tokens,heads,half] layout")
    if cos.shape != sin.shape:
        raise ValueError("Cosine and sine shapes must agree")
    return batch, tokens, width


@torch.library.custom_op("ddo_triton_qk_bench::single", mutates_args=(), device_types="cuda")
def rms_norm_split_rope(inputs: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor,
                        sin: torch.Tensor, heads: int, eps: float) -> torch.Tensor:
    batch, tokens, width = _validate(inputs, weight, cos, sin, heads, eps)
    with torch.cuda.device(inputs.device):
        output = torch.empty_like(inputs)
        _single[(batch * tokens,)](inputs, weight, cos, sin, output, tokens, width,
                                  heads, cos.shape[0], eps,
                                  num_warps=4, enable_fp_fusion=False)
    return output


@rms_norm_split_rope.register_fake
def fake_single(inputs, weight, cos, sin, heads, eps):
    return torch.empty_like(inputs)


@torch.library.custom_op("ddo_triton_qk_bench::paired", mutates_args=(), device_types="cuda")
def rms_norm_split_rope_qk(query: torch.Tensor, key: torch.Tensor, qweight: torch.Tensor,
                           kweight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                           heads: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, width = _validate(query, qweight, cos, sin, heads, eps)
    _validate(key, kweight, cos, sin, heads, eps)
    if query.shape != key.shape or query.device != key.device:
        raise ValueError("Dual Q/K requires matching activation shapes/devices and shared tables")
    with torch.cuda.device(query.device):
        qout, kout = torch.empty_like(query), torch.empty_like(key)
        _paired[(batch * tokens,)](query, key, qweight, kweight, cos, sin, qout, kout,
                                  tokens, width, heads, cos.shape[0], eps,
                                  num_warps=4, enable_fp_fusion=False)
    return qout, kout


@rms_norm_split_rope_qk.register_fake
def fake_paired(query, key, qweight, kweight, cos, sin, heads, eps):
    return torch.empty_like(query), torch.empty_like(key)
