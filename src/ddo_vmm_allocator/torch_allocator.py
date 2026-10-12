"""Opt-in installation of the native VMM allocator into a fresh PyTorch process."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class TorchAllocatorInstallError(RuntimeError):
    """Raised when the global PyTorch allocator cannot be safely replaced."""


_installed_allocator: Any | None = None
_installed_library_path: Path | None = None


def installed_library_path() -> Path | None:
    """Actual installed allocator library, not a mutable environment setting."""
    return _installed_library_path


def install_pytorch_vmm_allocator(
    library_path: str | os.PathLike[str] | None = None,
) -> Any:
    """Install the DLL through PyTorch's official pluggable-allocator API.

    This must run before PyTorch initializes CUDA, and it may be called at most
    once in a process.  Keeping the allocator in a module-level reference is
    required because PyTorch can free allocations after user-level references
    would otherwise be gone.
    """

    global _installed_allocator, _installed_library_path
    if _installed_allocator is not None:
        return _installed_allocator

    resolved_path = library_path or os.environ.get("DDO_VMM_LIBRARY_PATH")
    if not resolved_path:
        raise TorchAllocatorInstallError(
            "library path is required; pass library_path or set DDO_VMM_LIBRARY_PATH"
        )
    path = Path(resolved_path).expanduser().resolve(strict=True)

    from ._library import load_library, require_exports

    library = load_library(path, TorchAllocatorInstallError)
    require_exports(library, path,
                    ("ddo_vmm_torch_malloc", "ddo_vmm_torch_free"),
                    TorchAllocatorInstallError)

    import torch

    if not torch.cuda.is_available():
        raise TorchAllocatorInstallError("PyTorch CUDA is unavailable")
    if torch.cuda.is_initialized():
        raise TorchAllocatorInstallError(
            "CUDA is already initialized; start a fresh process through the DDO VMM launcher"
        )

    allocator = torch.cuda.memory.CUDAPluggableAllocator(
        str(path), "ddo_vmm_torch_malloc", "ddo_vmm_torch_free"
    )
    torch.cuda.memory.change_current_allocator(allocator)
    _installed_allocator = allocator
    _installed_library_path = path
    return allocator
