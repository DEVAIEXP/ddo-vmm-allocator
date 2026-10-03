"""Portable foundations for a DDO CUDA VMM allocator."""

from .platform import CudaVmmCapabilities, detect_cuda_vmm_capabilities
from .tensor_owners import TensorOwnerGroup, capture_model_tensor_owners
from .diffusers_adapter import TransformerAttachReport, attach_transformer_blocks
from .model_arena_adapter import (
    ModelArenaAttachReport,
    ModelArenaModuleAdapter,
    ModelArenaWeightStore,
    attach_model_linear_modules,
    inspect_model_linear_layout,
)

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
