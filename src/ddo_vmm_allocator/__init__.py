"""Portable foundations for a DDO CUDA VMM allocator."""

from .platform import CudaVmmCapabilities, detect_cuda_vmm_capabilities
from .diffusers_adapter import TransformerAttachReport, attach_transformer_blocks
from .model_arena_adapter import (
    ModelArenaAttachReport,
    ModelArenaModuleAdapter,
    ModelArenaWeightStore,
    attach_model_linear_modules,
)

__all__ = [
    "CudaVmmCapabilities",
    "detect_cuda_vmm_capabilities",
    "TransformerAttachReport",
    "attach_transformer_blocks",
    "ModelArenaAttachReport",
    "ModelArenaModuleAdapter",
    "ModelArenaWeightStore",
    "attach_model_linear_modules",
]
