"""Portable foundations for the DDO CUDA VMM allocator."""

from .platform import CudaVmmCapabilities, detect_cuda_vmm_capabilities
from .tensor_owners import TensorOwnerGroup, capture_model_tensor_owners

# Bootstrap imports torch_allocator through this package before allocator
# installation. Adapter imports may transitively query CUDA via optional
# quantization backends; retain the public API without eagerly importing them.
_ADAPTER_EXPORTS = {
    "TransformerAttachReport": "diffusers_adapter",
    "attach_transformer_blocks": "diffusers_adapter",
    "ModelArenaAttachReport": "model_arena_adapter",
    "ModelArenaModuleAdapter": "model_arena_adapter",
    "ModelArenaWeightStore": "model_arena_adapter",
    "attach_model_linear_modules": "model_arena_adapter",
    "inspect_model_linear_layout": "model_arena_adapter",
}


def __getattr__(name):
    if name not in _ADAPTER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{_ADAPTER_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_ADAPTER_EXPORTS))

__all__ = [
    "CudaVmmCapabilities",
    "TensorOwnerGroup",
    "capture_model_tensor_owners",
    "detect_cuda_vmm_capabilities",
    "TransformerAttachReport",
    "attach_transformer_blocks",
    "ModelArenaAttachReport",
    "ModelArenaModuleAdapter",
    "ModelArenaWeightStore",
    "attach_model_linear_modules",
    "inspect_model_linear_layout",
]
