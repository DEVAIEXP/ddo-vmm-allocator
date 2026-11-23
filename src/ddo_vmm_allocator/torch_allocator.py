"""Opt-in installation of the native VMM allocator into a fresh PyTorch process."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


class TorchAllocatorInstallError(RuntimeError):
    """Raised when the global PyTorch allocator cannot be safely replaced."""


_installed_allocator: Any | None = None
_installed_library_path: Path | None = None
# The configured library handle is retained for the process lifetime so the
# cold configuration and the pluggable allocator share one loaded module.
_configured_library: Any | None = None
_pool_configuration: dict[str, Any] = {"segmented_auto": False, "growable_reserve_bytes": 0}

# Written by the DDO bootstrap for its child process; explicit arguments win.
SEGMENTED_AUTO_ENVIRONMENT = "DDO_VMM_SEGMENTED_AUTO"
GROWABLE_RESERVE_ENVIRONMENT = "DDO_VMM_GROWABLE_RESERVE_BYTES"
MAX_GROWABLE_RESERVE_BYTES = 64 * 1024**3
_POOL_CONFIGURATION_EXPORTS = ("ddo_exp_segmented_configure_auto", "ddo_exp_segmented_configure_growable")

# Optional Torch-version-specific checked registration (native/torch_bridge).
# "auto" uses it only on an exact identity match, "0" never, "1" requires it.
TORCH_BRIDGE_ENVIRONMENT = "DDO_VMM_TORCH_BRIDGE"
TORCH_BRIDGE_IDENTITY = "ddo_vmm_torch_bridge.json"
TORCH_BRIDGE_ABI = 1
_allocation_boundary: str | None = None


def installed_library_path() -> Path | None:
    """Actual installed allocator library, not a mutable environment setting."""
    return _installed_library_path


def installed_allocation_boundary() -> str | None:
    """``"checked"`` when a final native refusal raises in PyTorch, ``"unguarded"``
    when the plain pluggable allocator is used, ``None`` before installation."""
    return _allocation_boundary


def _torch_bridge_mode() -> str:
    value = os.environ.get(TORCH_BRIDGE_ENVIRONMENT, "").strip().lower() or "auto"
    if value not in ("auto", "0", "1"):
        raise TorchAllocatorInstallError(f"{TORCH_BRIDGE_ENVIRONMENT} must be auto, 0 or 1, got {value!r}")
    return value


def _load_torch_bridge(torch: Any, library_path: Path, mode: str) -> Any | None:
    """Load the checked bridge next to the native library on an exact identity match.

    Without an identity file, or for another PyTorch/CPython build, the plain
    pluggable allocator stays in use (``mode="1"`` refuses instead). A matching
    identity whose module bytes differ is a damaged installation and is refused.
    """
    if mode == "0":
        return None
    identity_path = library_path.parent / TORCH_BRIDGE_IDENTITY
    if not identity_path.is_file():
        if mode == "1":
            raise TorchAllocatorInstallError(f"{TORCH_BRIDGE_ENVIRONMENT}=1 but no bridge identity at {identity_path}")
        return None
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TorchAllocatorInstallError(f"unreadable Torch bridge identity {identity_path}") from error
    expected = {
        "abi": TORCH_BRIDGE_ABI,
        "torch_version": torch.__version__,
        "torch_git": getattr(torch.version, "git_version", None),
        "python": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "platform": sys.platform,
    }
    mismatched = sorted(key for key, value in expected.items() if identity.get(key) != value)
    if mismatched:
        if mode == "1":
            raise TorchAllocatorInstallError(f"Torch bridge identity does not match this process: {mismatched}")
        return None
    module_path = library_path.parent / str(identity.get("module", ""))
    if not module_path.is_file():
        raise TorchAllocatorInstallError(f"Torch bridge module missing: {module_path}")
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    if digest != identity.get("sha256"):
        raise TorchAllocatorInstallError(f"Torch bridge module hash mismatch: {module_path}")
    spec = importlib.util.spec_from_file_location("ddo_vmm_torch_bridge", module_path)
    if spec is None or spec.loader is None:
        raise TorchAllocatorInstallError(f"cannot load Torch bridge module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "BRIDGE_ABI", None) != TORCH_BRIDGE_ABI:
        raise TorchAllocatorInstallError("Torch bridge ABI mismatch")
    return module


def _register_allocator(torch: Any, library: Any, path: Path, bridge: Any | None) -> tuple[Any, str]:
    """Register the already-loaded native callbacks with PyTorch.

    The plain path is PyTorch's own ``CUDAPluggableAllocator``. The checked
    path registers the same exports of the same loaded module through
    ``createCustomAllocator`` and only adds the null-result exception.
    """
    if bridge is None:
        allocator = torch.cuda.memory.CUDAPluggableAllocator(
            str(path), "ddo_vmm_torch_malloc", "ddo_vmm_torch_free"
        )
        torch.cuda.memory.change_current_allocator(allocator)
        return allocator, "unguarded"
    import ctypes

    def address(name: str) -> int:
        function = getattr(library, name, None)
        return 0 if function is None else int(ctypes.cast(function, ctypes.c_void_p).value or 0)

    bridge.install(address("ddo_vmm_torch_malloc"), address("ddo_vmm_torch_free"),
                   address("ddo_exp_segmented_last_refusal"))
    return bridge, "checked"


def installed_pool_configuration() -> dict[str, Any]:
    """Cold pool configuration applied by the installer (copy)."""
    return dict(_pool_configuration)


def _environment_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    if value not in ("0", "1"):
        raise TorchAllocatorInstallError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _environment_bytes(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as error:
        raise TorchAllocatorInstallError(f"{name} must be an integer byte count, got {value!r}") from error


def _resolve_pool_configuration(segmented_auto: bool | None, growable_reserve_bytes: int | None) -> tuple[bool, int]:
    if segmented_auto is None:
        segmented_auto = bool(_environment_flag(SEGMENTED_AUTO_ENVIRONMENT))
    if growable_reserve_bytes is None:
        growable_reserve_bytes = _environment_bytes(GROWABLE_RESERVE_ENVIRONMENT) or 0
    growable_reserve_bytes = int(growable_reserve_bytes)
    if growable_reserve_bytes < 0 or growable_reserve_bytes > MAX_GROWABLE_RESERVE_BYTES:
        raise TorchAllocatorInstallError(
            f"growable reserve must be within [0, {MAX_GROWABLE_RESERVE_BYTES}] bytes, got {growable_reserve_bytes}")
    if growable_reserve_bytes and not segmented_auto:
        raise TorchAllocatorInstallError("a growable reserve requires automatic segmented selection")
    return bool(segmented_auto), growable_reserve_bytes


def _apply_pool_configuration(library: Any, segmented_auto: bool, growable_reserve_bytes: int) -> None:
    """Configure the native pool before its first allocation (cold only)."""
    if not segmented_auto:
        return
    import ctypes

    configure_auto = library.ddo_exp_segmented_configure_auto
    configure_auto.argtypes = [ctypes.c_int]
    configure_auto.restype = ctypes.c_int
    status = configure_auto(1)
    if status != 0:
        raise TorchAllocatorInstallError(f"automatic segmented selection was refused (status {status})")
    if growable_reserve_bytes:
        configure_growable = library.ddo_exp_segmented_configure_growable
        configure_growable.argtypes = [ctypes.c_uint64]
        configure_growable.restype = ctypes.c_int
        status = configure_growable(growable_reserve_bytes)
        if status != 0:
            raise TorchAllocatorInstallError(f"growable reserve configuration was refused (status {status})")


def install_pytorch_vmm_allocator(
    library_path: str | os.PathLike[str] | None = None,
    *,
    segmented_auto: bool | None = None,
    growable_reserve_bytes: int | None = None,
    show_recovery_warns: bool | None = None,
) -> Any:
    """Install the DLL through PyTorch's official pluggable-allocator API.

    This must run before PyTorch initializes CUDA, and it may be called at most
    once in a process.  Keeping the allocator in a module-level reference is
    required because PyTorch can free allocations after user-level references
    would otherwise be gone.

    ``segmented_auto`` enables automatic segmented pool owners and
    ``growable_reserve_bytes`` sizes one growable shared owner (0 disables it;
    it requires ``segmented_auto``).  ``None`` reads the values the DDO
    bootstrap writes for its child; without them both stay disabled.  The
    configuration is applied before the allocator is installed and requires a
    library built with the segmented pool features; it is never ignored.
    """

    global _installed_allocator, _installed_library_path, _configured_library, _allocation_boundary
    if _installed_allocator is not None:
        return _installed_allocator

    resolved_path = library_path or os.environ.get("DDO_VMM_LIBRARY_PATH")
    if not resolved_path:
        raise TorchAllocatorInstallError(
            "library path is required; pass library_path or set DDO_VMM_LIBRARY_PATH"
        )
    path = Path(resolved_path).expanduser().resolve(strict=True)
    auto, reserve = _resolve_pool_configuration(segmented_auto, growable_reserve_bytes)
    bridge_mode = _torch_bridge_mode()
    if show_recovery_warns is None:
        show_recovery_warns = bool(_environment_flag("DDO_SHOW_RECOVERY_WARNS"))
    if type(show_recovery_warns) is not bool:
        raise TorchAllocatorInstallError("show_recovery_warns must be a boolean")

    from ._library import load_library, require_exports

    library = load_library(path, TorchAllocatorInstallError)
    require_exports(library, path,
                    ("ddo_vmm_torch_malloc", "ddo_vmm_torch_free"),
                    TorchAllocatorInstallError)
    if auto:
        require_exports(library, path, _POOL_CONFIGURATION_EXPORTS, TorchAllocatorInstallError)

    # Optional on older artifacts. Do not claim their console behavior changed.
    notifications = getattr(library, "ddo_vmm_set_recovery_warns_v1", None)
    if notifications is not None:
        import ctypes
        notifications.argtypes, notifications.restype = [ctypes.c_int], ctypes.c_int
        if notifications(int(show_recovery_warns)) != 0:
            raise TorchAllocatorInstallError("native recovery notification configuration failed")

    import torch

    if not torch.cuda.is_available():
        raise TorchAllocatorInstallError("PyTorch CUDA is unavailable")
    if torch.cuda.is_initialized():
        raise TorchAllocatorInstallError(
            "CUDA is already initialized; start a fresh process through the DDO VMM launcher"
        )

    # Load (but do not install) the optional bridge before mutating native state.
    bridge = _load_torch_bridge(torch, path, bridge_mode)
    _apply_pool_configuration(library, auto, reserve)
    allocator, boundary = _register_allocator(torch, library, path, bridge)
    _configured_library = library
    _pool_configuration.update(segmented_auto=auto, growable_reserve_bytes=reserve)
    _installed_allocator = allocator
    _installed_library_path = path
    _allocation_boundary = boundary
    return allocator
