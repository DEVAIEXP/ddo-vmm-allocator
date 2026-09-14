"""Portable foundations for a DDO CUDA VMM allocator."""

from .platform import CudaVmmCapabilities, detect_cuda_vmm_capabilities
from .diffusers_adapter import TransformerAttachReport, attach_transformer_blocks

__all__ = [
    "CudaVmmCapabilities",
    "detect_cuda_vmm_capabilities",
    "TransformerAttachReport",
    "attach_transformer_blocks",
]
