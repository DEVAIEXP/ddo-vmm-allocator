"""Experimental CPU contract for composite weights, not native transport.

Descriptors retain no loader tensors. Callers own payloads until native source
registration exists. Arithmetic and tensor dispatch stay with the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version

import torch


@dataclass(frozen=True)
class PayloadSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    size_bytes: int


@dataclass(frozen=True)
class Float8WeightOnlySpec:
    """Narrow opt-in descriptor; does not attach hooks or modify parameters."""

    payloads: tuple[PayloadSpec, ...]
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    logical_dtype: torch.dtype
    block_size: tuple[int, ...]
    kernel_preference: object

    @property
    def physical_bytes(self):
        return sum(item.size_bytes for item in self.payloads)

    def reconstruct_cpu(self, payloads):
        """Rebuild a wrapper over caller-owned CPU buffers, without copying."""
        _validate_payloads(payloads, self.payloads)
        return self.reconstruct_views(payloads)

    def reconstruct_views(self, payloads):
        """Wrap supplied views; GPU use requires its own native acceptance gate.

        Does not allocate/upload GPU payloads, wait on events, or perform math.
        Meta views are only for invalidating a retired component.
        """
        cls = _supported_backend()
        if set(payloads) != {item.name for item in self.payloads}:
            raise ValueError("Payload names differ from the contract")
        device = payloads["qdata"].device
        if device.type not in ("cpu", "cuda", "meta"):
            raise ValueError("Unsupported payload view device")
        _validate_payloads(payloads, self.payloads, device=device)
        wrapper = cls.__tensor_unflatten__(dict(payloads), {
            "block_size": list(self.block_size), "mm_config": None,
            "act_quant_kwargs": None, "kernel_preference": self.kernel_preference,
            "dtype": self.logical_dtype,
        }, self.shape, self.stride)
        if tuple(wrapper.shape) != self.shape or tuple(wrapper.stride()) != self.stride:
            raise ValueError("Reconstructed wrapper layout differs from the contract")
        if wrapper.dtype != self.logical_dtype:
            raise ValueError("Reconstructed wrapper logical dtype differs from the contract")
        return wrapper

    def validate_wrapper_view(self, wrapper):
        cls = _supported_backend()
        if type(wrapper) is not cls or wrapper.requires_grad:
            raise ValueError("Unsupported published wrapper type or gradient ownership")
        names, metadata = wrapper.__tensor_flatten__()
        expected = {"block_size": list(self.block_size), "mm_config": None,
                    "act_quant_kwargs": None, "kernel_preference": self.kernel_preference,
                    "dtype": self.logical_dtype}
        if set(names) != {item.name for item in self.payloads} or metadata != expected:
            raise ValueError("Published wrapper metadata differs from the contract")
        if tuple(wrapper.shape) != self.shape or tuple(wrapper.stride()) != self.stride:
            raise ValueError("Published wrapper geometry differs from the contract")
        if wrapper.dtype != self.logical_dtype:
            raise ValueError("Published wrapper logical dtype differs from the contract")
        _validate_payloads({name: getattr(wrapper, name) for name in names}, self.payloads,
                           device=wrapper.device)


def _supported_backend():
    # Do not import TorchAO during ordinary allocator use.
    if version("torchao") != "0.18.0":
        raise ValueError("This experimental contract requires TorchAO 0.18.0")
    from torchao.quantization import Float8Tensor
    return Float8Tensor


def _validate_payloads(payloads, specs, *, device=None):
    if set(payloads) != {item.name for item in specs}:
        raise ValueError("Payload names differ from the contract")
    intervals = []
    for spec in specs:
        tensor = payloads[spec.name]
        if type(tensor) is not torch.Tensor or (
                tensor.device != device if device is not None else tensor.device.type != "cpu"):
            raise ValueError("Plain payload tensors on one admitted device are required")
        if tensor.layout != torch.strided or not tensor.is_contiguous() or tensor.requires_grad:
            raise ValueError("Payload layout or gradient ownership is unsupported")
        if tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype:
            raise ValueError("Payload shape or physical dtype differs from the contract")
        if tensor.numel() * tensor.element_size() != spec.size_bytes or spec.size_bytes <= 0:
            raise ValueError("Payload byte length differs from the contract")
        # Refuse shared backing storage, even for disjoint views. Future alias
        # support must preserve whole-storage identity and native ownership.
        if tensor.device.type != "meta":
            storage = tensor.untyped_storage()
            intervals.append((storage.data_ptr(), storage.data_ptr() + storage.nbytes()))
    intervals.sort()
    if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("Shared or overlapping payload storage is unsupported")


def describe_float8_weight_only_cpu(weight):
    """Return immutable storage metadata; never retain or mutate the weight.

    Only the verified per-row E4M3 BF16 weight-only recipe is admitted. This does
    not authorize native attachment, checkpoint loading, or CUDA execution.
    """
    cls = _supported_backend()
    if type(weight) is not cls:
        raise ValueError("Unsupported quantized wrapper type")
    if weight.device.type != "cpu" or weight.requires_grad:
        raise ValueError("CPU inference-only weight required")
    names, metadata = weight.__tensor_flatten__()
    required = {"block_size", "mm_config", "act_quant_kwargs", "kernel_preference", "dtype"}
    if set(names) != {"qdata", "scale"} or set(metadata) != required:
        raise ValueError("Unknown quantized wrapper metadata")
    if metadata["act_quant_kwargs"] is not None or metadata["mm_config"] is not None:
        raise ValueError("Activation quantization or custom matmul configuration is unsupported")
    from torchao.quantization.quantize_.common import KernelPreference
    if metadata["kernel_preference"] is not KernelPreference.AUTO:
        raise ValueError("Only the verified automatic kernel preference is admitted")
    shape = tuple(weight.shape)
    if len(shape) != 2 or min(shape) <= 0 or metadata["block_size"] != [1, shape[1]]:
        raise ValueError("Only nonempty per-row matrix weights are admitted")
    if weight.dtype != torch.bfloat16 or metadata["dtype"] != weight.dtype:
        raise ValueError("Only BF16 logical weights are admitted")
    if weight.qdata.dtype != torch.float8_e4m3fn or weight.scale.dtype != torch.float32:
        raise ValueError("Only E4M3 data and FP32 scales are admitted")
    if tuple(weight.qdata.shape) != shape or tuple(weight.scale.shape) != (shape[0], 1):
        raise ValueError("Per-row payload dimensions are inconsistent")
    payloads = {name: getattr(weight, name) for name in names}
    specs = tuple(PayloadSpec(name, tuple(tensor.shape), tensor.dtype,
                              tensor.numel() * tensor.element_size())
                  for name, tensor in payloads.items())
    _validate_payloads(payloads, specs)
    return Float8WeightOnlySpec(specs, shape, tuple(weight.stride()), weight.dtype,
                               tuple(metadata["block_size"]), metadata["kernel_preference"])
