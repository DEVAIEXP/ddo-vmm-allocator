"""Explicit ctypes bridge for the portable CUDA Driver VMM library.

This module deliberately exposes regions as opaque handles.  It does not turn
an address into a PyTorch tensor yet; doing that safely requires the later
allocator adapter and lifetime integration.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import platform
from typing import Self


class NativeVmmError(RuntimeError):
    """Raised when a CUDA Driver VMM operation performed by the DLL fails."""


@dataclass(frozen=True)
class NativeVmmRegionInfo:
    address: int
    requested_size: int
    mapped_size: int


@dataclass(frozen=True)
class WindowsVideoMemoryInfo:
    """Per-process WDDM budgets for the CUDA adapter currently in use."""

    local_budget_bytes: int
    local_usage_bytes: int
    nonlocal_budget_bytes: int
    nonlocal_usage_bytes: int
    available_physical_bytes: int
    available_commit_bytes: int


class _WindowsVideoMemoryInfo(ctypes.Structure):
    _fields_ = [
        ("local_budget_bytes", ctypes.c_uint64),
        ("local_usage_bytes", ctypes.c_uint64),
        ("nonlocal_budget_bytes", ctypes.c_uint64),
        ("nonlocal_usage_bytes", ctypes.c_uint64),
        ("available_physical_bytes", ctypes.c_uint64),
        ("available_commit_bytes", ctypes.c_uint64),
    ]


class NativeVmmLibrary:
    """Loaded C ABI VMM library with explicit region ownership."""

    def __init__(self, library_path: str | os.PathLike[str]) -> None:
        self.path = Path(library_path).expanduser().resolve(strict=True)
        loader = ctypes.WinDLL if platform.system() == "Windows" else ctypes.CDLL
        self._library = loader(str(self.path))

        self._library.ddo_vmm_region_create.argtypes = [
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._library.ddo_vmm_region_create.restype = ctypes.c_int
        self._library.ddo_vmm_region_create_ex.argtypes = [
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._library.ddo_vmm_region_create_ex.restype = ctypes.c_int
        self._library.ddo_vmm_region_destroy.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_region_destroy.restype = None
        self._library.ddo_vmm_region_evict.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_region_evict.restype = ctypes.c_int
        self._library.ddo_vmm_region_materialize.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_region_materialize.restype = ctypes.c_int
        self._library.ddo_vmm_region_address.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_region_address.restype = ctypes.c_size_t
        self._library.ddo_vmm_region_requested_size.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_region_requested_size.restype = ctypes.c_size_t
        self._library.ddo_vmm_region_mapped_size.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_region_mapped_size.restype = ctypes.c_size_t
        self._library.ddo_vmm_region_is_materialized.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_region_is_materialized.restype = ctypes.c_int
        self._library.ddo_vmm_device_granularity.argtypes = [ctypes.c_int]
        self._library.ddo_vmm_device_granularity.restype = ctypes.c_size_t
        self._library.ddo_vmm_pressure_register.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_pressure_register.restype = None
        self._library.ddo_vmm_pressure_unregister.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_pressure_unregister.restype = None
        self._library.ddo_vmm_pressure_set_pinned.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._library.ddo_vmm_pressure_set_pinned.restype = None
        self._library.ddo_vmm_pressure_touch.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_pressure_touch.restype = None
        self._library.ddo_vmm_pressure_evict_one.argtypes = [ctypes.c_int]
        self._library.ddo_vmm_pressure_evict_one.restype = ctypes.c_int
        self._library.ddo_vmm_windows_memory_query.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_WindowsVideoMemoryInfo),
        ]
        self._library.ddo_vmm_windows_memory_query.restype = ctypes.c_int
        self._library.ddo_vmm_dxgi_budget_guard_set.argtypes = [
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        self._library.ddo_vmm_dxgi_budget_guard_set.restype = ctypes.c_int
        self._library.ddo_vmm_last_error.argtypes = []
        self._library.ddo_vmm_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        value = self._library.ddo_vmm_last_error()
        return value.decode("utf-8", errors="replace") if value else "unknown native VMM error"

    def device_granularity(self, device_ordinal: int = 0) -> int:
        result = int(self._library.ddo_vmm_device_granularity(device_ordinal))
        if result == 0:
            raise NativeVmmError(self._last_error())
        return result

    def pressure_evict_one(self, device_ordinal: int = 0) -> bool:
        return bool(self._library.ddo_vmm_pressure_evict_one(device_ordinal))

    def windows_memory_info(self, device_ordinal: int = 0) -> WindowsVideoMemoryInfo:
        """Return WDDM local/shared budgets associated with a CUDA device."""

        info = _WindowsVideoMemoryInfo()
        if self._library.ddo_vmm_windows_memory_query(device_ordinal, ctypes.byref(info)) != 0:
            raise NativeVmmError(self._last_error())
        return WindowsVideoMemoryInfo(
            local_budget_bytes=int(info.local_budget_bytes),
            local_usage_bytes=int(info.local_usage_bytes),
            nonlocal_budget_bytes=int(info.nonlocal_budget_bytes),
            nonlocal_usage_bytes=int(info.nonlocal_usage_bytes),
            available_physical_bytes=int(info.available_physical_bytes),
            available_commit_bytes=int(info.available_commit_bytes),
        )

    def set_dxgi_budget_guard(
        self,
        *,
        device_ordinal: int = 0,
        local_reserve_bytes: int = 0,
        shared_reserve_bytes: int = 0,
        system_reserve_bytes: int = 0,
    ) -> None:
        """Keep the requested memory reserve before materializing another VMM region.

        A zero reserve disables the corresponding limit.  The native layer
        evicts only registered, non-pinned regions while it tries to restore
        these margins.
        """

        values = (local_reserve_bytes, shared_reserve_bytes, system_reserve_bytes)
        if any(value < 0 for value in values):
            raise ValueError("DXGI budget guard reserves must be non-negative")
        if self._library.ddo_vmm_dxgi_budget_guard_set(
            device_ordinal, *values
        ) != 0:
            raise NativeVmmError(self._last_error())

    def create_region(self, size_bytes: int, device_ordinal: int = 0, *, backing: str = "device") -> "NativeVmmRegion":
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        handle = ctypes.c_void_p()
        if backing not in {"device", "host"}:
            raise ValueError("backing must be 'device' or 'host'")
        status = self._library.ddo_vmm_region_create_ex(
            device_ordinal, size_bytes, 1 if backing == "host" else 0, ctypes.byref(handle)
        )
        if status != 0 or not handle.value:
            raise NativeVmmError(self._last_error())
        return NativeVmmRegion(self, handle, backing=backing)


class NativeVmmRegion:
    """A mapped CUDA VMM region that must be closed explicitly."""

    def __init__(self, owner: NativeVmmLibrary, handle: ctypes.c_void_p, *, backing: str = "device") -> None:
        self._owner = owner
        self._handle: ctypes.c_void_p | None = handle
        self.backing = backing
        self._owner._library.ddo_vmm_pressure_register(handle)

    @property
    def closed(self) -> bool:
        return self._handle is None

    @property
    def info(self) -> NativeVmmRegionInfo:
        if self._handle is None:
            raise NativeVmmError("VMM region is closed")
        return NativeVmmRegionInfo(
            address=int(self._owner._library.ddo_vmm_region_address(self._handle)),
            requested_size=int(self._owner._library.ddo_vmm_region_requested_size(self._handle)),
            mapped_size=int(self._owner._library.ddo_vmm_region_mapped_size(self._handle)),
        )

    @property
    def materialized(self) -> bool:
        if self._handle is None:
            raise NativeVmmError("VMM region is closed")
        return bool(self._owner._library.ddo_vmm_region_is_materialized(self._handle))

    def evict(self) -> None:
        """Unmap and release backing while retaining this object's virtual address."""

        if self._handle is None:
            raise NativeVmmError("VMM region is closed")
        if self._owner._library.ddo_vmm_region_evict(self._handle) != 0:
            raise NativeVmmError(self._owner._last_error())

    def materialize(self) -> None:
        """Map new device-local backing at the original virtual address."""

        if self._handle is None:
            raise NativeVmmError("VMM region is closed")
        if self._owner._library.ddo_vmm_region_materialize(self._handle) != 0:
            raise NativeVmmError(self._owner._last_error())

    def set_pressure_pinned(self, pinned: bool) -> None:
        if self._handle is None:
            raise NativeVmmError("VMM region is closed")
        self._owner._library.ddo_vmm_pressure_set_pinned(self._handle, int(pinned))

    def touch_pressure(self) -> None:
        if self._handle is None:
            raise NativeVmmError("VMM region is closed")
        self._owner._library.ddo_vmm_pressure_touch(self._handle)

    def close(self) -> None:
        if self._handle is not None:
            self._owner._library.ddo_vmm_pressure_unregister(self._handle)
            self._owner._library.ddo_vmm_region_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exception_type: object, _exception: object, _traceback: object) -> None:
        self.close()


def load_native_vmm(library_path: str | os.PathLike[str] | None = None) -> NativeVmmLibrary:
    """Load the native library explicitly or from ``DDO_VMM_LIBRARY_PATH``."""

    resolved_path = library_path or os.environ.get("DDO_VMM_LIBRARY_PATH")
    if not resolved_path:
        raise NativeVmmError(
            "native VMM library path is required; pass library_path or set DDO_VMM_LIBRARY_PATH"
        )
    return NativeVmmLibrary(resolved_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the DDO native CUDA Driver VMM bridge.")
    parser.add_argument("--library", required=True, help="Path to ddo_vmm.dll or libddo_vmm.so")
    parser.add_argument("--megabytes", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.megabytes <= 0:
        raise SystemExit("--megabytes must be positive")
    library = load_native_vmm(args.library)
    with library.create_region(args.megabytes * 1024**2, args.device) as region:
        info = region.info
        region.evict()
        if region.materialized:
            raise NativeVmmError("VMM region remained materialized after eviction")
        region.materialize()
        if not region.materialized or region.info.address != info.address:
            raise NativeVmmError("VMM region did not rematerialize at its original virtual address")
        print("Native Python-to-VMM bridge: PASS")
        print(f"  library: {library.path}")
        print(f"  virtual address: 0x{info.address:x}")
        print(f"  requested: {info.requested_size / 1024**2:.2f} MiB")
        print(f"  mapped: {info.mapped_size / 1024**2:.2f} MiB")
        print(f"  granularity: {library.device_granularity(args.device) / 1024**2:.2f} MiB")
        print("  evict/rematerialize at same address: PASS")


if __name__ == "__main__":
    main()
