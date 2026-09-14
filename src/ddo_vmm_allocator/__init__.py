"""Portable foundations for a DDO CUDA VMM allocator."""

from .platform import CudaVmmCapabilities, detect_cuda_vmm_capabilities

__all__ = [
    "CudaVmmCapabilities",
    "detect_cuda_vmm_capabilities",
]
