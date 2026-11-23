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
from typing import Any, Self


class NativeVmmError(RuntimeError):
    """Raised when a CUDA Driver VMM operation performed by the DLL fails."""


@dataclass(frozen=True)
class NativeVmmRegionInfo:
    address: int
    requested_size: int
    mapped_size: int


@dataclass(frozen=True)
class NativeModelArenaInfo:
    address: int
    virtual_size: int
    page_size: int
    resident_size: int
    fault_ceiling: int


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

    def torch_memory_stats(self) -> dict[str, int | float]:
        """Optional process-wide ledger gauges; no CUDA calls or completion waits.

        Each export locks independently: this is not an atomic total or peak.
        Allocations being retired outside the ledger and model arenas are excluded.
        """
        pool = self.torch_pool_stats()
        if pool:
            return pool  # Do not reinterpret standalone per-record footprint.
        result = {}
        for name in ("live_allocation_count", "live_mapped_bytes",
                     "cached_allocation_count", "cached_mapped_bytes"):
            fn = getattr(self._library, f"ddo_vmm_torch_{name}", None)
            if fn is None:
                continue  # Unsupported is not zero.
            fn.argtypes = []
            fn.restype = ctypes.c_size_t
            result[name] = int(fn())
        result.update(self.torch_allocation_footprint())
        return result

    def torch_allocation_footprint(self) -> dict[str, int]:
        """Atomic live-ledger decomposition, excluding cache/retiring/model arenas."""
        if self.torch_pool_stats():
            return {}
        fn = getattr(self._library, "ddo_vmm_torch_allocation_footprint_v1", None)
        if fn is None:
            return {}
        fn.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        fn.restype = ctypes.c_int
        values = (ctypes.c_uint64 * 6)()
        if fn(values, len(values)) != 0:
            raise NativeVmmError("native torch allocation footprint query failed")
        requested, minimum, mapped, padding, excess, count = map(int, values)
        if not (requested <= minimum <= mapped and minimum - requested == padding
                and mapped - minimum == excess):
            raise NativeVmmError("native torch allocation footprint is inconsistent")
        return dict(zip(("footprint_requested_bytes", "footprint_minimum_mapped_bytes",
                         "footprint_mapped_bytes", "footprint_granularity_padding_bytes",
                         "footprint_reuse_excess_bytes", "footprint_allocation_count"),
                        (requested, minimum, mapped, padding, excess, count)))

    def torch_release_stats(self) -> dict[str, int | float]:
        """Optional process-lifetime host release counters; never invokes CUDA."""
        if self.torch_pool_stats():
            return {}
        fn = getattr(self._library, "ddo_vmm_torch_release_stats_v1", None)
        if fn is None:
            return {}  # Older DLLs remain supported; absent does not mean zero.
        fn.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        fn.restype = ctypes.c_int
        values = (ctypes.c_uint64 * 6)()
        if fn(values, len(values)) != 0:
            raise NativeVmmError("native torch release statistics query failed")
        return {
            "reset_sync_calls": int(values[0]),
            "reset_sync_seconds": values[1] / 1e9,
            "reset_failures": int(values[2]),
            "cache_eviction_attempts": int(values[3]),
            "cache_eviction_attempted_bytes": int(values[4]),
            "cache_eviction_sync_seconds": values[5] / 1e9,
        }

    def set_recovery_warns(self, enabled: bool) -> bool:
        """Set process-wide recovery notices; False means an older binary lacks control."""
        if type(enabled) is not bool:
            raise TypeError("show_recovery_warns must be a boolean")
        fn = getattr(self._library, "ddo_vmm_set_recovery_warns_v1", None)
        if fn is None:
            return False
        fn.argtypes, fn.restype = [ctypes.c_int], ctypes.c_int
        if fn(int(enabled)) != 0:
            raise NativeVmmError("native recovery notification configuration failed")
        return True

    def set_allocation_diagnostics(self, enabled: bool) -> None:
        """Acquire/release a process-wide diagnostic scope; never initialize CUDA."""
        fn = getattr(self._library, "ddo_vmm_allocation_diagnostics_enable_v1", None)
        if fn is None:
            raise NativeVmmError("allocation diagnostics require the separate diagnostic binary")
        fn.argtypes, fn.restype = [ctypes.c_int], ctypes.c_int
        status = int(fn(int(enabled)))
        if status == 2:
            raise NativeVmmError("allocation diagnostics are unsupported by this binary")
        if status:
            raise NativeVmmError("allocation diagnostic scope is already owned or invalid")

    def allocation_diagnostics_snapshot(self) -> dict[str, int | float]:
        """Optional cumulative native CPU timers; absent is unavailable, not zero."""
        fn = getattr(self._library, "ddo_vmm_allocation_diagnostics_snapshot_v2", None)
        version = 2 if fn is not None else 1
        if fn is None:
            fn = getattr(self._library, "ddo_vmm_allocation_diagnostics_snapshot_v1", None)
        if fn is None:
            return {}
        fn.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        fn.restype = ctypes.c_int
        values = (ctypes.c_uint64 * (36 if version == 2 else 26))()
        status = int(fn(values, len(values)))
        if status == 2:
            return {}
        if status or values[0] != version or values[1] not in (0, 1):
            raise NativeVmmError("invalid allocation diagnostic snapshot/schema")
        result = {"schema_version": version, "enabled": bool(values[1])}
        for index, name in enumerate(("create", "map", "access", "unmap", "release", "reserve", "address_free")):
            calls, ns, failures = values[2 + index * 3:5 + index * 3]
            result.update({name + "_calls": calls, name + "_host_seconds": ns / 1e9,
                           name + "_failed_calls": failures})
        result.update(pool_reuse_calls=int(values[23]), pool_new_backings=int(values[24]),
                      pool_observed_peak_mapped_bytes=int(values[25]))
        if version == 2:
            for offset, name in ((26, "access_retry_sync"), (29, "fault_range")):
                result.update({name + "_calls": int(values[offset]),
                               name + "_host_seconds": values[offset + 1] / 1e9})
            result["access_retry_sync_failed_calls"] = int(values[28])
            for index, name in enumerate(("success", "not_ready", "out_of_memory", "other")):
                result["access_return_" + name + "_calls"] = int(values[32 + index])
        return result

    def torch_pool_stats(self) -> dict[str, int | float]:
        """Optional unique-backing pool snapshot, native schema v1; never CUDA."""
        fn = getattr(self._library, "ddo_vmm_torch_pool_stats_v1", None)
        if fn is None:
            return {}
        fn.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        fn.restype = ctypes.c_int
        values = (ctypes.c_uint64 * 15)()
        status = int(fn(values, len(values)))
        if status == 2:
            return {}
        if status != 0:
            raise NativeVmmError("native torch pool statistics query failed")
        if (values[0] != 1 or values[1] != sum(values[i] for i in (3, 4, 5, 6))
                or values[2] > values[3] or values[10] > values[1]
                or values[14] > values[9]):
            raise NativeVmmError("native torch pool statistics are inconsistent")
        names = ("enabled", "mapped_bytes", "requested_bytes", "leased_bytes",
                 "available_bytes", "pending_bytes", "quarantined_bytes",
                 "live_allocation_count", "reserved_allocation_count", "backing_count",
                 "idle_backing_bytes", "eviction_attempts", "eviction_wait_ns",
                 "failures", "idle_backing_count")
        result = {f"pool_{name}": int(value) for name, value in zip(names, values)}
        result["pool_eviction_wait_seconds"] = result.pop("pool_eviction_wait_ns") / 1e9
        result["pool_schema_version"] = 1
        return result

    def trim_idle_torch_pool(self) -> dict:
        """Explicit native release; unsupported libraries are never replaced."""
        function = getattr(self._library, "ddo_vmm_torch_trim_idle_v1", None)
        if function is None:
            return {"status": "unsupported", "trim_executed": False}
        function.argtypes = []
        function.restype = ctypes.c_int
        before = self.torch_pool_stats()
        status = int(function())
        if status == 2:
            return {"status": "unsupported", "trim_executed": False, "before": before}
        if status != 0:
            raise NativeVmmError("native idle pool trim failed; unsafe backing retained")
        return {"status": "completed", "trim_executed": True,
                "before": before, "after": self.torch_pool_stats()}

    def __init__(self, library_path: str | os.PathLike[str]) -> None:
        self.path = Path(library_path).expanduser().resolve(strict=True)
        from ._library import load_library

        self._library = load_library(self.path, NativeVmmError)
        try:
            self._bind_exports()
        except AttributeError as error:
            raise NativeVmmError(
                f"Incompatible native VMM library '{self.path}': {error}. "
                "Use a DLL built for this allocator package. "
                "The selected library will not be replaced automatically."
            ) from error

    def _bind_exports(self) -> None:
        """Bind existing contracts; newer optional telemetry remains optional."""
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
        self._library.ddo_vmm_region_materialize_range.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._library.ddo_vmm_region_materialize_range.restype = ctypes.c_int
        self._library.ddo_vmm_region_resident_size.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_region_resident_size.restype = ctypes.c_size_t
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
        self._library.ddo_vmm_model_arena_create.argtypes = [
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._library.ddo_vmm_model_arena_create.restype = ctypes.c_int
        self._library.ddo_vmm_model_arena_destroy.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_model_arena_destroy.restype = None
        self._library.ddo_vmm_model_arena_fault_range.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._library.ddo_vmm_model_arena_fault_range.restype = ctypes.c_int
        # Optional symbols preserve loading older DLLs for legacy contracts.
        for name, arguments in (
            ("ddo_vmm_model_arena_enable_streaming", [ctypes.c_void_p, ctypes.c_size_t]),
            ("ddo_vmm_model_arena_record_compute", [ctypes.c_void_p, ctypes.c_void_p]),
            ("ddo_vmm_model_arena_wait_compute", [ctypes.c_void_p, ctypes.c_void_p]),
            ("ddo_vmm_model_arena_drain_compute", [ctypes.c_void_p]),
        ):
            function = getattr(self._library, name, None)
            if function is not None:
                function.argtypes = arguments
                function.restype = ctypes.c_int
        self._library.ddo_vmm_model_arena_prepare_view_range.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._library.ddo_vmm_model_arena_prepare_view_range.restype = ctypes.c_int
        self._library.ddo_vmm_model_arena_release_view_range.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._library.ddo_vmm_model_arena_release_view_range.restype = ctypes.c_int
        self._library.ddo_vmm_model_arena_copy_from_host.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p
        ]
        self._library.ddo_vmm_model_arena_copy_from_host.restype = ctypes.c_int
        self._library.ddo_vmm_model_arena_copy_from_host_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._library.ddo_vmm_model_arena_copy_from_host_async.restype = ctypes.c_int
        self._library.ddo_vmm_model_arena_unpin_range.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._library.ddo_vmm_model_arena_unpin_range.restype = ctypes.c_int
        self._library.ddo_vmm_model_arena_address.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_model_arena_address.restype = ctypes.c_size_t
        self._library.ddo_vmm_model_arena_virtual_size.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_model_arena_virtual_size.restype = ctypes.c_size_t
        self._library.ddo_vmm_model_arena_page_size.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_model_arena_page_size.restype = ctypes.c_size_t
        self._library.ddo_vmm_model_arena_resident_size.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_model_arena_resident_size.restype = ctypes.c_size_t
        self._library.ddo_vmm_model_arena_fault_ceiling.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_model_arena_fault_ceiling.restype = ctypes.c_size_t
        self._library.ddo_vmm_model_arena_materialized_pages.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_model_arena_materialized_pages.restype = ctypes.c_uint64
        self._library.ddo_vmm_model_arena_reused_pages.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_model_arena_reused_pages.restype = ctypes.c_uint64
        self._library.ddo_vmm_model_arena_evicted_pages.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_model_arena_evicted_pages.restype = ctypes.c_uint64
        self._library.ddo_vmm_model_arena_resident_prefix_pages.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_model_arena_resident_prefix_pages.restype = ctypes.c_size_t
        self._library.ddo_vmm_model_arena_tail_evictions.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_model_arena_tail_evictions.restype = ctypes.c_uint64
        self._library.ddo_vmm_model_arena_pressure_evictions.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_model_arena_pressure_evictions.restype = ctypes.c_uint64
        self._library.ddo_vmm_host_ring_create.argtypes = [
            ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)
        ]
        self._library.ddo_vmm_host_ring_create.restype = ctypes.c_int
        self._library.ddo_vmm_host_ring_destroy.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_host_ring_destroy.restype = None
        self._library.ddo_vmm_host_ring_slot_size.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_host_ring_slot_size.restype = ctypes.c_size_t
        self._library.ddo_vmm_host_ring_slot_count.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_host_ring_slot_count.restype = ctypes.c_size_t
        self._library.ddo_vmm_host_ring_source_worker_count.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_host_ring_source_worker_count.restype = ctypes.c_size_t
        self._library.ddo_vmm_host_ring_source_read_chunks.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_host_ring_source_read_chunks.restype = ctypes.c_uint64
        self._library.ddo_vmm_host_ring_source_read_bytes.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_host_ring_source_read_bytes.restype = ctypes.c_uint64
        self._library.ddo_vmm_host_ring_source_read_failures.argtypes = [ctypes.c_void_p]
        self._library.ddo_vmm_host_ring_source_read_failures.restype = ctypes.c_uint64
        self._library.ddo_vmm_host_ring_slot_address.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._library.ddo_vmm_host_ring_slot_address.restype = ctypes.c_size_t
        self._library.ddo_vmm_host_ring_slot_ready.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._library.ddo_vmm_host_ring_slot_ready.restype = ctypes.c_int
        self._library.ddo_vmm_host_ring_wait_slot.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._library.ddo_vmm_host_ring_wait_slot.restype = ctypes.c_int
        self._library.ddo_vmm_host_ring_copy_to_arena_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self._library.ddo_vmm_host_ring_copy_to_arena_async.restype = ctypes.c_int
        self._library.ddo_vmm_host_ring_read_file_to_arena_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self._library.ddo_vmm_host_ring_read_file_to_arena_async.restype = ctypes.c_int
        self._library.ddo_vmm_host_ring_read_file_async.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_size_t]
        self._library.ddo_vmm_host_ring_read_file_async.restype = ctypes.c_int
        self._library.ddo_vmm_host_ring_wait_file_read.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._library.ddo_vmm_host_ring_wait_file_read.restype = ctypes.c_int
        self._library.ddo_vmm_host_ring_copy_ticket_to_arena_async.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_void_p,
        ]
        self._library.ddo_vmm_host_ring_copy_ticket_to_arena_async.restype = ctypes.c_int
        self._library.ddo_vmm_host_ring_copy_batch_to_arena_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self._library.ddo_vmm_host_ring_copy_batch_to_arena_async.restype = ctypes.c_int
        self._library.ddo_vmm_host_ring_wait_slot_async.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p
        ]
        self._library.ddo_vmm_host_ring_wait_slot_async.restype = ctypes.c_int
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

    def open_source_file(self, path: str | os.PathLike[str]):
        """Open a persistent immutable file source without initializing CUDA."""
        from .source import NativeFileSource

        return NativeFileSource.open(self, path)

    def create_source_cache(self, *, slot_bytes: int, slots: int, device: int = 0,
                            host_headroom_bytes: int = 2**30, process_limit_bytes: int = 128 * 2**20,
                            model_windows: bool = False, persistent_buffer: bool = False,
                            registered_limit_bytes: int = 0):
        """Experimental bounded cache for native contract tests; not a preset."""
        from .source_cache import NativeSourceCache

        return NativeSourceCache(self, slot_bytes=slot_bytes, slots=slots, device=device,
                                 host_headroom_bytes=host_headroom_bytes, process_limit_bytes=process_limit_bytes,
                                 model_windows=model_windows, persistent_buffer=persistent_buffer,
                                 registered_limit_bytes=registered_limit_bytes)

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

    def create_model_arena(
        self,
        virtual_size_bytes: int,
        *,
        resident_budget_bytes: int | None = None,
        page_size_bytes: int = 0,
        device_ordinal: int = 0,
        backing: str = "device",
    ) -> "NativeModelArena":
        if virtual_size_bytes <= 0:
            raise ValueError("virtual_size_bytes must be positive")
        if resident_budget_bytes is not None and resident_budget_bytes < 0:
            raise ValueError("resident_budget_bytes must be non-negative")
        if page_size_bytes < 0:
            raise ValueError("page_size_bytes must be non-negative")
        if backing not in {"device", "host"}:
            raise ValueError("backing must be 'device' or 'host'")
        handle = ctypes.c_void_p()
        status = self._library.ddo_vmm_model_arena_create(
            device_ordinal,
            virtual_size_bytes,
            0 if resident_budget_bytes is None else resident_budget_bytes,
            page_size_bytes,
            1 if backing == "host" else 0,
            ctypes.byref(handle),
        )
        if status != 0 or not handle.value:
            raise NativeVmmError(self._last_error())
        return NativeModelArena(self, handle, backing=backing)

    def create_host_ring(
        self,
        slot_size_bytes: int,
        slot_count: int,
        *,
        device_ordinal: int = 0,
    ) -> "NativeHostRing":
        """Allocate registered host-memory slots owned by the native library.

        The returned addresses are intentionally opaque byte buffers.  A later
        transport ticket will own file reads, CUDA-stream copies, and reuse
        synchronization; Python must not retain tensors that alias these slots.
        """

        if slot_size_bytes <= 0:
            raise ValueError("slot_size_bytes must be positive")
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        handle = ctypes.c_void_p()
        status = self._library.ddo_vmm_host_ring_create(
            device_ordinal,
            slot_size_bytes,
            slot_count,
            ctypes.byref(handle),
        )
        if status != 0 or not handle.value:
            raise NativeVmmError(self._last_error())
        return NativeHostRing(self, handle)


class NativeHostRing:
    """Native-owned, CUDA-registered host buffers for future transport tickets."""

    def __init__(self, owner: NativeVmmLibrary, handle: ctypes.c_void_p) -> None:
        self._owner = owner
        self._handle: ctypes.c_void_p | None = handle

    @property
    def closed(self) -> bool:
        return self._handle is None

    def _require_handle(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise NativeVmmError("native host ring is closed")
        return self._handle

    @property
    def slot_size_bytes(self) -> int:
        return int(self._owner._library.ddo_vmm_host_ring_slot_size(self._require_handle()))

    @property
    def slot_count(self) -> int:
        return int(self._owner._library.ddo_vmm_host_ring_slot_count(self._require_handle()))

    @property
    def source_stats(self) -> dict[str, int]:
        """Lifetime counters for the native positioned source transport."""

        handle = self._require_handle()
        return {
            "workers": int(self._owner._library.ddo_vmm_host_ring_source_worker_count(handle)),
            "read_chunks": int(self._owner._library.ddo_vmm_host_ring_source_read_chunks(handle)),
            "read_bytes": int(self._owner._library.ddo_vmm_host_ring_source_read_bytes(handle)),
            "read_failures": int(self._owner._library.ddo_vmm_host_ring_source_read_failures(handle)),
        }

    def slot_address(self, slot_index: int) -> int:
        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("host ring slot index is out of range")
        address = int(
            self._owner._library.ddo_vmm_host_ring_slot_address(
                self._require_handle(), slot_index
            )
        )
        if address == 0:
            raise NativeVmmError(self._owner._last_error())
        return address

    def slot_ready(self, slot_index: int) -> bool:
        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("host ring slot index is out of range")
        return bool(self._owner._library.ddo_vmm_host_ring_slot_ready(self._require_handle(), slot_index))

    def wait_slot(self, slot_index: int) -> None:
        """Wait only for the selected slot's previous DMA before reusing it."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("host ring slot index is out of range")
        status = self._owner._library.ddo_vmm_host_ring_wait_slot(self._require_handle(), slot_index)
        if status != 0:
            raise NativeVmmError(self._owner._last_error())

    def copy_to_arena_async(
        self,
        slot_index: int,
        arena: "NativeModelArena",
        *,
        size_bytes: int,
        slot_offset_bytes: int = 0,
        arena_offset_bytes: int = 0,
        stream: Any = None,
    ) -> None:
        """Enqueue a slot-to-arena DMA without synchronizing either stream.

        The caller must retain the model-arena fault pin and wait for its CUDA
        event before making this slot available to another transport request.
        """

        if size_bytes <= 0 or slot_offset_bytes < 0 or arena_offset_bytes < 0:
            raise ValueError("copy offsets must be non-negative and size_bytes must be positive")
        stream_ptr = 0 if stream is None else getattr(stream, "cuda_stream", stream)
        status = self._owner._library.ddo_vmm_host_ring_copy_to_arena_async(
            self._require_handle(),
            slot_index,
            slot_offset_bytes,
            arena._require_handle(),
            arena_offset_bytes,
            size_bytes,
            ctypes.c_void_p(int(stream_ptr)),
        )
        if status != 0:
            raise NativeVmmError(self._owner._last_error())

    def read_file_to_arena_async(
        self,
        slot_index: int,
        path: str | Path,
        file_offset_bytes: int,
        arena: "NativeModelArena",
        *,
        arena_offset_bytes: int,
        size_bytes: int,
        stream: Any = None,
    ) -> None:
        """Native Windows read into this registered slot followed by queued DMA."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("host ring slot index is out of range")
        if file_offset_bytes < 0 or arena_offset_bytes < 0 or size_bytes <= 0:
            raise ValueError("file offset, arena offset, and size must be valid")
        stream_ptr = 0 if stream is None else getattr(stream, "cuda_stream", stream)
        status = self._owner._library.ddo_vmm_host_ring_read_file_to_arena_async(
            self._require_handle(),
            slot_index,
            str(path).encode("utf-8"),
            file_offset_bytes,
            arena._require_handle(),
            arena_offset_bytes,
            size_bytes,
            ctypes.c_void_p(int(stream_ptr)),
        )
        if status != 0:
            raise NativeVmmError(self._owner._last_error())

    def read_file_async(self, slot_index: int, path: str | Path, file_offset_bytes: int, *, size_bytes: int) -> None:
        if slot_index < 0 or slot_index >= self.slot_count or file_offset_bytes < 0 or size_bytes <= 0:
            raise ValueError("native shard read arguments are invalid")
        status = self._owner._library.ddo_vmm_host_ring_read_file_async(
            self._require_handle(), slot_index, str(path).encode("utf-8"), file_offset_bytes, size_bytes
        )
        if status != 0:
            raise NativeVmmError(self._owner._last_error())

    def wait_file_read(self, slot_index: int) -> None:
        status = self._owner._library.ddo_vmm_host_ring_wait_file_read(self._require_handle(), slot_index)
        if status != 0:
            raise NativeVmmError(self._owner._last_error())

    def copy_ticket_to_arena_async(self, ranges: list[tuple[int, int, int]], arena: "NativeModelArena", *, stream: Any = None) -> None:
        """Complete a native source ticket and enqueue all its H2D ranges at once."""
        if not ranges:
            raise ValueError("ticket ranges must not be empty")
        slots = (ctypes.c_size_t * len(ranges))(*(slot for slot, _, _ in ranges))
        offsets = (ctypes.c_size_t * len(ranges))(*(_offset for _, _offset, _ in ranges))
        sizes = (ctypes.c_size_t * len(ranges))(*(size for _, _, size in ranges))
        stream_ptr = 0 if stream is None else getattr(stream, "cuda_stream", stream)
        status = self._owner._library.ddo_vmm_host_ring_copy_ticket_to_arena_async(
            self._require_handle(), slots, arena._require_handle(), offsets, sizes, len(ranges), ctypes.c_void_p(int(stream_ptr))
        )
        if status != 0:
            raise NativeVmmError(self._owner._last_error())

    def wait_slot_async(self, slot_index: int, *, stream: Any = None) -> None:
        """Make one CUDA stream wait for this slot's native DMA event only."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("host ring slot index is out of range")
        stream_ptr = 0 if stream is None else getattr(stream, "cuda_stream", stream)
        status = self._owner._library.ddo_vmm_host_ring_wait_slot_async(
            self._require_handle(), slot_index, ctypes.c_void_p(int(stream_ptr))
        )
        if status != 0:
            raise NativeVmmError(self._owner._last_error())

    def copy_batch_to_arena_async(
        self,
        slot_index: int,
        arena: "NativeModelArena",
        ranges: list[tuple[int, int, int]],
        *,
        stream: Any = None,
    ) -> None:
        """DMA packed ``(slot_offset, arena_offset, size)`` ranges as one slot ticket."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("host ring slot index is out of range")
        if not ranges:
            raise ValueError("copy batch must contain at least one range")
        if any(slot_offset < 0 or arena_offset < 0 or size <= 0 for slot_offset, arena_offset, size in ranges):
            raise ValueError("copy batch ranges must have non-negative offsets and positive sizes")
        values = tuple(zip(*ranges, strict=True))
        slot_offsets = (ctypes.c_size_t * len(ranges))(*values[0])
        arena_offsets = (ctypes.c_size_t * len(ranges))(*values[1])
        sizes = (ctypes.c_size_t * len(ranges))(*values[2])
        stream_ptr = 0 if stream is None else getattr(stream, "cuda_stream", stream)
        status = self._owner._library.ddo_vmm_host_ring_copy_batch_to_arena_async(
            self._require_handle(),
            slot_index,
            slot_offsets,
            arena._require_handle(),
            arena_offsets,
            sizes,
            len(ranges),
            ctypes.c_void_p(int(stream_ptr)),
        )
        if status != 0:
            raise NativeVmmError(self._owner._last_error())

    def close(self) -> None:
        if self._handle is not None:
            self._owner._library.ddo_vmm_host_ring_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exception_type: object, _exception: object, _traceback: object) -> None:
        self.close()


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

    @property
    def resident_size(self) -> int:
        if self._handle is None:
            raise NativeVmmError("VMM region is closed")
        return int(self._owner._library.ddo_vmm_region_resident_size(self._handle))

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

    def materialize_range(self, offset_bytes: int, size_bytes: int) -> None:
        """Map backing for one byte range inside the reserved virtual address."""

        if self._handle is None:
            raise NativeVmmError("VMM region is closed")
        if offset_bytes < 0 or size_bytes <= 0:
            raise ValueError("offset_bytes must be non-negative and size_bytes must be positive")
        if self._owner._library.ddo_vmm_region_materialize_range(
            self._handle, offset_bytes, size_bytes
        ) != 0:
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


class NativeExecutionPool:
    """Native slot leases and rotation; arenas are borrowed, never destroyed here individually."""

    def __init__(self, library, model, slot_bytes, slots=2):
        self._owner = library
        self._backing = model.backing
        self._handle = ctypes.c_void_p()
        self._arenas = {}
        specs = {
            "create": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)],
            "acquire": [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_void_p)],
            "release": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p],
            "close": [ctypes.POINTER(ctypes.c_void_p)],
        }
        for suffix, args in specs.items():
            fn = getattr(library._library, "ddo_vmm_execution_pool_" + suffix, None)
            if fn is None:
                raise NativeVmmError("native execution pool requires rebuilding allocator")
            fn.argtypes, fn.restype = args, ctypes.c_int
        self._check(library._library.ddo_vmm_execution_pool_create(
            model._require_handle(), slot_bytes, slots, ctypes.byref(self._handle)))
        self.slots = slots
        self.capacity_bytes = ((slot_bytes + model.info.page_size - 1) // model.info.page_size) * model.info.page_size

    def _check(self, status):
        if status:
            raise NativeVmmError(self._owner._last_error())

    def acquire(self, stream):
        if not self._handle.value:
            raise NativeVmmError("native execution pool is closed")
        slot, handle = ctypes.c_size_t(), ctypes.c_void_p()
        self._check(self._owner._library.ddo_vmm_execution_pool_acquire(
            self._handle, int(stream.cuda_stream), ctypes.byref(slot), ctypes.byref(handle)))
        arena = self._arenas.get(slot.value)
        if arena is None:
            arena = NativeModelArena(self._owner, handle, backing=self._backing)
            self._arenas[slot.value] = arena
        return slot.value, arena

    def release(self, slot, stream):
        self._check(self._owner._library.ddo_vmm_execution_pool_release(
            self._handle, slot, int(stream.cuda_stream)))

    @property
    def resident_bytes(self):
        return sum(arena.info.resident_size for arena in self._arenas.values())

    def configure_reclaim(self, *, enabled: bool, margin_bytes: int = 256 * 2**20):
        if not isinstance(enabled, bool) or not isinstance(margin_bytes, int) or not 0 <= margin_bytes < 2**64:
            raise ValueError("reclaim requires a boolean enablement and a uint64 margin")
        fn = getattr(self._owner._library, "ddo_vmm_execution_pool_configure_reclaim", None)
        if fn is None:
            raise NativeVmmError("execution backing reclaim requires a compatible allocator build")
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
        fn.restype = ctypes.c_int
        self._check(fn(self._handle, int(enabled), margin_bytes))

    def reclaim_idle(self, index: int) -> int:
        """Explicit native reclaim; live or pending consumers refuse without waiting."""
        if not isinstance(index, int) or not 0 <= index < self.slots:
            raise ValueError("execution slot index is out of range")
        fn = getattr(self._owner._library, "ddo_vmm_execution_pool_reclaim_idle", None)
        if fn is None:
            raise NativeVmmError("execution backing reclaim requires a compatible allocator build")
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
        fn.restype = ctypes.c_int
        released = ctypes.c_uint64()
        self._check(fn(self._handle, index, ctypes.byref(released)))
        return released.value

    def reclaim_stats(self) -> dict[str, int]:
        fn = getattr(self._owner._library, "ddo_vmm_execution_pool_reclaim_stats", None)
        if fn is None:
            return {}  # Older artifacts do not report this optional feature.
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        fn.restype = ctypes.c_int
        values = (ctypes.c_uint64 * 16)()
        self._check(fn(self._handle, values, len(values)))
        if values[0] != 1 or values[1] != len(values):
            raise NativeVmmError("unsupported execution backing stats schema")
        names = ("enabled", "margin_bytes", "queries", "triggers", "busy", "reclaims",
                 "released_bytes", "recreations", "recreated_bytes", "unavailable", "failures")
        return dict(zip(names, values[2:13]))

    def close(self):
        self._check(self._owner._library.ddo_vmm_execution_pool_close(ctypes.byref(self._handle)))
        for arena in self._arenas.values():
            arena._handle = None  # Native pool already destroyed borrowed owners.
        self._arenas.clear()


class NativeModelArena:
    """One model-sized CUDA virtual address arena with range faults."""

    @property
    def torch_allocator_stats(self) -> dict[str, int | float]:
        result = self._owner.torch_release_stats()
        memory_stats = getattr(self._owner, "torch_memory_stats", None)
        if memory_stats is not None:
            result.update(memory_stats())
        return result

    def __init__(self, owner: NativeVmmLibrary, handle: ctypes.c_void_p, *, backing: str = "device") -> None:
        self._owner = owner
        self._handle: ctypes.c_void_p | None = handle
        self.backing = backing

    @property
    def closed(self) -> bool:
        return self._handle is None

    def _require_handle(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise NativeVmmError("model arena is closed")
        return self._handle

    @property
    def info(self) -> NativeModelArenaInfo:
        if self._handle is None:
            raise NativeVmmError("model arena is closed")
        return NativeModelArenaInfo(
            address=int(self._owner._library.ddo_vmm_model_arena_address(self._handle)),
            virtual_size=int(self._owner._library.ddo_vmm_model_arena_virtual_size(self._handle)),
            page_size=int(self._owner._library.ddo_vmm_model_arena_page_size(self._handle)),
            resident_size=int(self._owner._library.ddo_vmm_model_arena_resident_size(self._handle)),
            fault_ceiling=int(self._owner._library.ddo_vmm_model_arena_fault_ceiling(self._handle)),
        )

    @property
    def residency_counters(self) -> dict[str, int]:
        handle = self._require_handle()
        return {
            "materialized_pages": int(self._owner._library.ddo_vmm_model_arena_materialized_pages(handle)),
            "reused_pages": int(self._owner._library.ddo_vmm_model_arena_reused_pages(handle)),
            "evicted_pages": int(self._owner._library.ddo_vmm_model_arena_evicted_pages(handle)),
            "resident_prefix_pages": int(self._owner._library.ddo_vmm_model_arena_resident_prefix_pages(handle)),
            "tail_evictions": int(self._owner._library.ddo_vmm_model_arena_tail_evictions(handle)),
            "pressure_evictions": int(self._owner._library.ddo_vmm_model_arena_pressure_evictions(handle)),
        }

    def fault_range(self, offset_bytes: int, size_bytes: int) -> int:
        """Materialize, pin, and return the range generation signature."""

        if self._handle is None:
            raise NativeVmmError("model arena is closed")
        if offset_bytes < 0 or size_bytes <= 0:
            raise ValueError("offset_bytes must be non-negative and size_bytes must be positive")
        signature = ctypes.c_uint64()
        if self._owner._library.ddo_vmm_model_arena_fault_range(
            self._handle, offset_bytes, size_bytes, ctypes.byref(signature)
        ) != 0:
            raise NativeVmmError(self._owner._last_error())
        return int(signature.value)

    def enable_streaming(self, execution_buffer_bytes: int) -> None:
        if execution_buffer_bytes <= 0:
            raise ValueError("execution buffer capacity must be positive")
        self._streaming_call("ddo_vmm_model_arena_enable_streaming", execution_buffer_bytes)

    def drain_compute(self) -> None:
        function = getattr(self._owner._library, "ddo_vmm_model_arena_drain_compute", None)
        if function is None:
            raise NativeVmmError("native execution buffer requires rebuilding the allocator DLL")
        if function(self._require_handle()) != 0:
            raise NativeVmmError(self._owner._last_error())

    def record_compute(self, stream: Any) -> None:
        self._streaming_call("ddo_vmm_model_arena_record_compute", int(stream.cuda_stream))

    def wait_compute(self, stream: Any) -> None:
        self._streaming_call("ddo_vmm_model_arena_wait_compute", int(stream.cuda_stream))

    def _streaming_call(self, name: str, argument: int) -> None:
        function = getattr(self._owner._library, name, None)
        if function is None:
            raise NativeVmmError("native execution buffer requires rebuilding the allocator DLL")
        if function(self._require_handle(), argument) != 0:
            raise NativeVmmError(self._owner._last_error())

    def peek_range_signature(self, offset_bytes: int, size_bytes: int) -> int | None:
        """Observe native generation without pinning; not a residency guarantee."""
        handle = self._require_handle()
        if offset_bytes < 0 or size_bytes <= 0:
            raise ValueError("offset_bytes must be non-negative and size_bytes positive")
        fn = getattr(self._owner._library, "ddo_vmm_model_arena_peek_range_signature", None)
        if fn is None:
            return None  # Older DLLs preserve speculative transport behavior.
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                       ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int)]
        fn.restype = ctypes.c_int
        signature, resident = ctypes.c_uint64(), ctypes.c_int()
        if fn(handle, offset_bytes, size_bytes, ctypes.byref(signature), ctypes.byref(resident)):
            raise NativeVmmError(self._owner._last_error())
        return int(signature.value) if resident.value else None

    def prepare_view_range(self, offset_bytes: int, size_bytes: int) -> None:
        if self._handle is None:
            raise NativeVmmError("model arena is closed")
        if self._owner._library.ddo_vmm_model_arena_prepare_view_range(
            self._handle, offset_bytes, size_bytes
        ) != 0:
            raise NativeVmmError(self._owner._last_error())

    def release_view_range(self, offset_bytes: int, size_bytes: int) -> None:
        if self._handle is None:
            raise NativeVmmError("model arena is closed")
        if self._owner._library.ddo_vmm_model_arena_release_view_range(
            self._handle, offset_bytes, size_bytes
        ) != 0:
            raise NativeVmmError(self._owner._last_error())

    def unpin_range(self, offset_bytes: int, size_bytes: int) -> None:
        """Release one pin acquired by fault_range."""

        if self._handle is None:
            raise NativeVmmError("model arena is closed")
        if offset_bytes < 0 or size_bytes <= 0:
            raise ValueError("offset_bytes must be non-negative and size_bytes must be positive")
        if self._owner._library.ddo_vmm_model_arena_unpin_range(
            self._handle, offset_bytes, size_bytes
        ) != 0:
            raise NativeVmmError(self._owner._last_error())

    def copy_from_host(self, offset_bytes: int, source: Any) -> None:
        if self._handle is None:
            raise NativeVmmError("model arena is closed")
        size_bytes = int(source.numel() * source.element_size())
        if self._owner._library.ddo_vmm_model_arena_copy_from_host(
            self._handle, offset_bytes, size_bytes, ctypes.c_void_p(int(source.data_ptr()))
        ) != 0:
            raise NativeVmmError(self._owner._last_error())

    def copy_from_host_async(self, offset_bytes: int, source: Any, stream: Any = None) -> None:
        if self._handle is None:
            raise NativeVmmError("model arena is closed")
        size_bytes = int(source.numel() * source.element_size())
        stream_ptr = 0
        if stream is not None:
            stream_ptr = getattr(stream, "cuda_stream", stream)
        if self._owner._library.ddo_vmm_model_arena_copy_from_host_async(
            self._handle,
            offset_bytes,
            size_bytes,
            ctypes.c_void_p(int(source.data_ptr())),
            ctypes.c_void_p(int(stream_ptr)),
        ) != 0:
            raise NativeVmmError(self._owner._last_error())

    def close(self) -> None:
        if self._handle is not None:
            self._owner._library.ddo_vmm_model_arena_destroy(self._handle)
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
