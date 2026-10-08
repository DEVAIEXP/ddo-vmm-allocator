"""Opt-in opaque wrappers around the existing modulation and rotary kernels.

No new arithmetic, residency policy, private stream or invocation counters.
The normalizer remains outside the modulation op so its original forward can
be preserved by the host's opaque normalization boundary during compilation.
"""

import torch
import triton

from . import triton_execution as original


@torch.library.custom_op("ddo_triton_execution_bench::modulate", mutates_args=(), device_types="cuda")
def modulate(normalized: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    original._activation(normalized)
    (sb, st), (ss0, ss1) = original._modulation_shape(scale, normalized)
    (hb, ht), (hs0, hs1) = original._modulation_shape(shift, normalized)
    with torch.cuda.device(normalized.device):
        output = torch.empty_like(normalized, memory_format=torch.contiguous_format)
        original._modulate[(normalized.shape[0] * normalized.shape[1],)](
            normalized, scale, shift, output,
            normalized.shape[1], normalized.shape[2], normalized.stride(0), normalized.stride(1),
            sb, st, ss0, ss1, hb, ht, hs0, hs1,
            triton.next_power_of_2(normalized.shape[2]), enable_fp_fusion=False)
    return output


@modulate.register_fake
def _modulate_fake(normalized, scale, shift):
    return torch.empty_like(normalized, memory_format=torch.contiguous_format)


@torch.library.custom_op("ddo_triton_execution_bench::rotary_pair", mutates_args=(), device_types="cuda")
def rotary_pair(query: torch.Tensor, key: torch.Tensor, qcos: torch.Tensor, qsin: torch.Tensor,
                kcos: torch.Tensor, ksin: torch.Tensor, layout: str) -> tuple[torch.Tensor, torch.Tensor]:
    original._activation(query)
    original._activation(key)
    if query.device != key.device or query.dtype != key.dtype or not query.is_contiguous() or not key.is_contiguous():
        raise ValueError("Paired rotary requires contiguous Q/K with matching device/dtype")
    qlayout = original._rotary_layout(query, (qcos, qsin), layout)
    klayout = original._rotary_layout(key, (kcos, ksin), layout)
    with torch.cuda.device(query.device):
        qout, kout = torch.empty_like(query), torch.empty_like(key)
        grid = (triton.cdiv(max(query.numel(), key.numel()), 256), 2)
        original._rotary_pair[grid](query, qcos, qsin, qout, key, kcos, ksin, kout,
                                   *qlayout, *klayout, layout == "split", 256, enable_fp_fusion=False)
    return qout, kout


@rotary_pair.register_fake
def _rotary_pair_fake(query, key, qcos, qsin, kcos, ksin, layout):
    return torch.empty_like(query), torch.empty_like(key)


class CompilableTritonExecutionOps:
    """Traceable callbacks; kernel validation and launches remain opaque."""

    def rms_modulate(self, norm, inputs, scale, shift):
        return modulate(norm(inputs), scale, shift)

    def rotary_pair(self, query, key, query_freqs, key_freqs, layout):
        return rotary_pair(query, key, *query_freqs, *key_freqs, layout)
