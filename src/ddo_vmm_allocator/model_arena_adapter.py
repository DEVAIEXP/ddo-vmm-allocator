"""Model-address arena adapter for PyTorch dense linear weights.

This is the target native VMM route: one model-sized virtual address span,
range faults around actual weight use, range signatures to avoid redundant
copies, and range unpinning after the launched CUDA work.
"""

from __future__ import annotations

import os
import re
import sys
import time
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, replace
from itertools import islice, pairwise
from typing import Any

from .native import NativeExecutionPool, NativeHostRing, NativeModelArena, NativeVmmError, NativeVmmLibrary
from .source_cache import NativeSourceBatch, _measure_batch_scope, _run_cleanup_preserving_error
from .source import NativeSourceError
from .tensor_view import VmmTensorViewError, cuda_uint8_tensor_from_address


@contextmanager
def _measure_upload_stream(stream, stats):
    """Partition an existing stream context, without adding CUDA operations."""
    import torch

    start = time.perf_counter()
    entered = exiting = start
    body_entered = False
    try:
        with torch.cuda.stream(stream):
            entered = time.perf_counter()
            body_entered = True
            try:
                yield
            finally:
                exiting = time.perf_counter()
    finally:
        finished = time.perf_counter()
        if not body_entered:
            entered = exiting = finished
        for name, elapsed in (("enter", entered - start),
                              ("body", exiting - entered),
                              ("exit", finished - exiting)):
            key = "upload_stream_" + name + "_seconds"
            stats[key] = stats.get(key, 0.0) + elapsed
        stats["upload_stream_scope_calls"] = stats.get("upload_stream_scope_calls", 0) + 1

try:
    from diffusers_dynamic_offloader.vmm_weight_store import RegisteredHostWeightRegistry
except ImportError:
    RegisteredHostWeightRegistry = None


# Experimental execution-unit granularity (process-wide, set before attachment).
# "block" (default): the outermost ModuleList item fences all selected
# descendants. "block_child": each direct child of that item is its own unit
# (a Linear that is itself a direct child stays its own unit), so execution
# slots fit the largest child instead of the whole item. Structure only; no
# model or class names.
_EXECUTION_UNIT_LEVELS = ("block", "block_child")
_execution_unit_level = "block"


def configure_experimental_execution_unit_level(level: str) -> None:
    """Select the execution-unit granularity for later attachments."""
    global _execution_unit_level
    if level not in _EXECUTION_UNIT_LEVELS:
        raise ValueError(f"unsupported execution unit level {level!r}")
    _execution_unit_level = level


def experimental_execution_unit_level() -> str:
    return _execution_unit_level


def _has_sub_block_units(root, unit_names) -> bool:
    """Whether any execution unit is not itself a ModuleList item (by structure)."""
    import torch

    by_name = dict(root.named_modules())
    for name in unit_names:
        parent = name.rpartition(".")[0]
        if not isinstance(by_name.get(parent), torch.nn.ModuleList):
            return True
    return False


def _discover_linear_layout(root, module_pattern):
    """Snapshot selection and enclosing ModuleList units without model names.

    Keep enumeration order and parameter alignment identical to attachment.
    The snapshot retains modules, never detached loader tensors. The outermost
    list item fences all selected descendants, including nested lists; at the
    experimental ``block_child`` level its direct children are the units.
    """
    import torch

    if not isinstance(root, torch.nn.Module):
        raise TypeError("root must be a torch.nn.Module")
    pattern = re.compile(module_pattern)
    modules = tuple(root.named_modules())
    matched = {name for name, _ in modules if name and pattern.search(name)}
    list_items = {}
    for name, module in modules:
        if isinstance(module, torch.nn.ModuleList):
            for index, child in enumerate(module):
                list_items[f"{name}.{index}" if name else str(index)] = child
    selected = []
    units = {}
    payload = 0
    by_name = dict(modules) if _execution_unit_level == "block_child" else None
    for name, module in modules:
        if not isinstance(module, torch.nn.Linear):
            continue
        parts = name.split(".")
        ancestors = [".".join(parts[:index]) for index in range(1, len(parts) + 1)]
        if not (pattern.search(name) or any(parent in matched for parent in ancestors)):
            continue
        selected.append((name, module))
        unit = next((parent for parent in ancestors if parent in list_items), None)
        if unit is not None and by_name is not None and name != unit:
            # Direct child of the outermost item that contains this Linear.
            child = unit + "." + name[len(unit) + 1:].split(".")[0]
            units[name] = (child, by_name[child])
        elif unit is not None:
            units[name] = (unit, list_items[unit])
        for parameter_name in ("weight", "bias"):
            parameter = getattr(module, parameter_name, None)
            if isinstance(parameter, torch.nn.Parameter):
                payload = _align_up(payload, max(parameter.element_size(), 256))
                payload += parameter.numel() * parameter.element_size()
    return tuple(selected), units, payload


def inspect_model_linear_layout(root, module_pattern: str = ".*") -> dict[str, Any]:
    """Detached metadata using attachment's exact selector, without CUDA/DLL I/O.

    Counts are Parameter-identity unique, not storage-unique. A selected Linear
    alias does not cover a nonselected owner of the same Parameter. Selection
    also does not prove custom/quantized forward compatibility or execution.
    """
    import torch

    selected, units, payload = _discover_linear_layout(root, module_pattern)
    selected_names = {
        f"{name}.{key}" if name else key
        for name, module in selected
        for key in ("weight", "bias")
        if isinstance(getattr(module, key, None), torch.nn.Parameter)
    }
    groups = {}
    for name, parameter in root.named_parameters(remove_duplicate=False):
        group = groups.setdefault(id(parameter), {
            "names": [], "selected_names": [], "nonselected_names": [],
            "shape": list(parameter.shape), "dtype": str(parameter.dtype),
            "bytes": parameter.numel() * parameter.element_size(),
        })
        group["names"].append(name)
        group["selected_names" if name in selected_names else "nonselected_names"].append(name)
    parameters = list(groups.values())
    buffers = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                "bytes": value.numel() * value.element_size()}
               for name, value in root.named_buffers()]
    return {
        "schema_version": 1, "observation": "metadata_only",
        "module_pattern": module_pattern, "selected_module_count": len(selected),
        "selected_modules": [{"name": name, "class": type(module).__name__,
                              "execution_unit": units[name][0] if name in units else name}
                             for name, module in selected],
        "selected_aligned_payload_bytes": payload,
        "unique_parameter_bytes": sum(group["bytes"] for group in parameters),
        "selected_unique_parameter_bytes": sum(group["bytes"] for group in parameters
                                               if group["selected_names"]),
        "nonselected_unique_parameter_bytes": sum(group["bytes"] for group in parameters
                                                  if group["nonselected_names"]),
        "shared_selected_nonselected_bytes": sum(group["bytes"] for group in parameters
                                                 if group["selected_names"] and group["nonselected_names"]),
        "parameters": parameters, "buffers": buffers,
        "unique_buffer_bytes": sum(value["bytes"] for value in buffers),
    }


def estimate_execution_slot_bytes(root, module_pattern: str = ".*", *,
                                  granularity_bytes: int = 2 * 1024**2) -> int:
    """One execution slot for ``root`` at the configured unit level (CPU metadata only).

    Uses the attachment's selector, unit grouping and per-tensor 512-byte slot
    alignment, rounded up to ``granularity_bytes``; streaming attachment needs
    two such slots inside the residency ceiling. No CUDA, DLL or model change.
    """
    if granularity_bytes <= 0:
        raise ValueError("granularity_bytes must be positive")
    selected, units, _ = _discover_linear_layout(root, module_pattern)
    sizes: dict[str, int] = {}
    for name, module in selected:
        unit = units[name][0] if name in units else name
        for _key, parameter in _layout_parameter_items(module):
            size = sizes.get(unit, 0)
            sizes[unit] = ((size + 511) & ~511) + parameter.numel() * parameter.element_size()
    maximum = max(sizes.values(), default=0)
    return -(-maximum // granularity_bytes) * granularity_bytes


def _layout_parameter_items(module, auxiliary_groups=None):
    """Legacy Linear geometry, extended only by explicit auxiliary ownership."""
    import torch

    keys = ("weight", "bias") if isinstance(module, torch.nn.Linear) else tuple(module._parameters)
    for key in keys:
        parameter = module._parameters.get(key)
        if not isinstance(parameter, torch.nn.Parameter):
            continue
        if not isinstance(module, torch.nn.Linear) and (id(module), key) not in (auxiliary_groups or {}):
            continue
        yield key, parameter


def _layout_view_spans(selected, unit_by_module, auxiliary_groups=None):
    """Aligned contiguous setup spans, without keeping CPU source tensors."""
    import torch

    spans = []
    offset = 0
    previous_unit = None
    for name, module in selected:
        unit = unit_by_module[name][0] if name in unit_by_module else name
        for parameter_name, parameter in _layout_parameter_items(module, auxiliary_groups):
            offset = _align_up(offset, max(parameter.element_size(), 256))
            size = parameter.numel() * parameter.element_size()
            if size <= 0:
                continue
            if spans and previous_unit == unit:
                spans[-1] = (spans[-1][0], offset + size - spans[-1][0])
            else:
                spans.append((offset, size))
            previous_unit = unit
            offset += size
    return tuple(spans)


class ModelArenaError(RuntimeError):
    """Raised when the model-address arena cannot attach or execute safely."""


class DeviceTransferBuffer:
    """Reusable CUDA byte buffer for exceptional dtype/transfer conversions."""

    def __init__(self, device: int = 0) -> None:
        self.device = device
        self.tensor = None
        self.capacity_bytes = 0

    def ensure(self, size_bytes: int):
        import torch
        if size_bytes > self.capacity_bytes:
            self.tensor = torch.empty(size_bytes, dtype=torch.uint8, device=f"cuda:{self.device}")
            self.capacity_bytes = size_bytes
        return self.tensor[:size_bytes]

    def close(self) -> None:
        self.tensor = None
        self.capacity_bytes = 0


def _upload_execution_bindings(store, bindings, arena):
    """Upload mixed sources into existing views on one fenced upload stream.

    CPU sources remain owned by their bindings. Only checkpoint-backed bindings
    enter immutable file batches; no additional registration or staging is added.
    The enclosing execution buffer owns partial-failure and consumer fences.
    """
    host = [binding for binding in bindings if binding.file_slice is None]
    if not host:
        return store._native_shard_transport.upload_batch(bindings, arena, store._upload_stream)

    import torch

    for binding in host:
        if binding.source.device.type != "cpu":
            raise ModelArenaError("execution buffer host source must be a CPU tensor")
        if binding.source.dtype != binding.stable.dtype or binding.source.shape != binding.stable.shape:
            raise ModelArenaError("execution buffer host source must match its destination view")
    files = [binding for binding in bindings if binding.file_slice is not None]
    with torch.cuda.stream(store._upload_stream):
        if files:
            store._native_shard_transport.upload_batch(files, arena, store._upload_stream)
        for binding in host:
            binding.stable.copy_(binding.source, non_blocking=True)
        # Same-stream ordering covers the file batch and every CPU-source copy.
        ready = torch.cuda.Event(enable_timing=False)
        ready.record(store._upload_stream)
    return ready


class NativeExecutionBuffer:
    """Native mapped execution storage, permanently pinned against eviction.

    Capacity comes from unit geometry, carved out of the existing total GPU
    ceiling. CUDA compute fences and backing belong to the native arena; tensor
    objects are views only. Uploads reuse the existing immutable-source transport.
    """

    def __init__(self, store, capacity_bytes: int) -> None:
        self.store = store
        page = store.page_bytes
        self.capacity_bytes = ((capacity_bytes + page - 1) // page) * page
        self.native = store._library.create_model_arena(
            self.capacity_bytes, resident_budget_bytes=self.capacity_bytes,
            page_size_bytes=page, device_ordinal=store._device_ordinal,
            backing=store._backing,
        )
        self._materialized = False
        self._unsafe_fence = False
        self._views = []

    def upload(self, bindings):
        import torch

        self.native.wait_compute(self.store._upload_stream)
        if not self._materialized:
            self.native.fault_range(0, self.capacity_bytes)
            self._materialized = True  # Keep the single pin until destruction.
        copies = []
        views = []
        offset = 0
        for binding in bindings:
            offset = (offset + 511) & ~511
            if offset + binding.nbytes > self.capacity_bytes:
                raise ModelArenaError("execution unit exceeds native buffer capacity")
            raw, owner = cuda_uint8_tensor_from_address(
                self.native.info.address + offset, binding.nbytes, owner=self.native,
                backing=self.store._backing,
            )
            view = raw.view(binding.source.dtype).view(tuple(binding.source.shape))
            views.append((binding, view))
            copies.append(replace(binding, offset=offset, stable=view,
                                  signature=None, native_prefetch_ticket=None))
            offset += binding.nbytes
        start = time.perf_counter()
        try:
            with torch.cuda.stream(self.store._upload_stream):
                ready = _upload_execution_bindings(self.store, copies, self.native)
        except BaseException as error:
            # Also protects partially submitted uploads if the operation fails.
            _run_cleanup_preserving_error(
                lambda: self._record_fence(self.store._upload_stream),
                "execution buffer upload fence", error)
            raise
        else:
            self._record_fence(self.store._upload_stream)
        self._views = views
        self.store._runtime_stats["dma_enqueue_seconds"] += time.perf_counter() - start
        size = sum(binding.nbytes for binding in bindings)
        self.store._runtime_stats["dma_enqueue_bytes"] += size
        self.store._runtime_stats["upload_bytes"] += size
        self.store._runtime_stats["uploads"] += len(bindings)
        self.store._runtime_stats["execution_buffer_uploads"] += 1
        self.store._runtime_stats["execution_buffer_bytes"] += size
        return views, ready

    def finish_compute(self, stream) -> None:
        self._record_fence(stream)

    def _record_fence(self, stream) -> None:
        self._unsafe_fence = True
        self.native.record_compute(stream)
        self._unsafe_fence = False

    def close(self) -> None:
        if self._unsafe_fence:
            raise NativeVmmError("execution buffer completion is unproven; backing retained")
        self.native.drain_compute()
        self._views.clear()
        self.native.close()


@dataclass
class _ExecutionLease:
    slot: int
    arena: Any
    views: Any
    ready: Any


class _CompletionSet:
    """Retain both upload and compute completion for partial composite fallback."""

    def __init__(self, events):
        self.events = tuple(events)

    def query(self):
        return all(event.query() is True for event in self.events)

    def synchronize(self):
        for event in self.events:
            event.synchronize()


class NativeExecutionBuffers:
    """Two native slots, immutable unit layouts and native-owned rotation."""

    def __init__(self, store, capacity_bytes):
        self.store = store
        self.pool = NativeExecutionPool(store._library, store.native, capacity_bytes, slots=2)
        self.capacity_bytes = self.pool.capacity_bytes * self.pool.slots
        self._layouts = {}

    @property
    def resident_bytes(self):
        return self.pool.resident_bytes

    def upload(self, bindings):
        slot, arena = self.pool.acquire(self.store._upload_stream)
        key = (slot, tuple(binding.label for binding in bindings))
        try:
            layout = self._layouts.get(key)
            if layout is None:
                views, copies, offset = [], [], 0
                address = arena.info.address
                for binding in bindings:
                    offset = (offset + 511) & ~511
                    if offset + binding.nbytes > self.pool.capacity_bytes:
                        raise ModelArenaError("unit exceeds native execution slot")
                    raw, _owner = cuda_uint8_tensor_from_address(address + offset,
                        binding.nbytes, owner=arena, backing=self.store._backing)
                    view = raw.view(binding.source.dtype).view(tuple(binding.source.shape))
                    views.append((binding, view))
                    copies.append(replace(binding, offset=offset, stable=view,
                        signature=None, native_prefetch_ticket=None))
                    offset += binding.nbytes
                layout = self._layouts[key] = (views, copies)
            views, copies = layout
            start = time.perf_counter()
            ready = _upload_execution_bindings(self.store, copies, arena)
            size = sum(binding.nbytes for binding in bindings)
            for metric in ("upload_bytes", "dma_enqueue_bytes", "execution_buffer_bytes"):
                self.store._runtime_stats[metric] += size
            self.store._runtime_stats["dma_enqueue_seconds"] += time.perf_counter() - start
            self.store._runtime_stats["uploads"] += len(bindings)
            self.store._runtime_stats["execution_buffer_uploads"] += 1
            return _ExecutionLease(slot, arena, views, ready)
        except BaseException as error:
            # A partial upload is fenced on its own stream. Failure recording
            # that fence leaves the native lease owned and close refuses it.
            _run_cleanup_preserving_error(
                lambda: self.pool.release(slot, self.store._upload_stream),
                "execution lease upload fence", error)
            raise

    def finish_compute(self, lease, stream):
        self.pool.release(lease.slot, stream)

    def cancel(self, lease):
        self.pool.release(lease.slot, self.store._upload_stream)

    def close(self):
        self.pool.close()
        self._layouts.clear()


@dataclass
class _TensorBinding:
    owner: Any
    parameter_name: str
    source: Any
    file_slice: Any | None
    stable: Any
    offset: int
    nbytes: int
    signature: int | None = None
    ready_event: Any | None = None
    prefetched_source: Any | None = None
    prefetch_reads: list[Any] | None = None
    native_prefetch_ticket: _NativeReadTicket | None = None
    label: str = ""

    @property
    def virtual_address(self) -> int:
        return self.owner._parameters[self.parameter_name].data_ptr()


@dataclass(frozen=True)
class _FileTensorDescriptor:
    """Tensor metadata retained when immutable payload bytes stay in a shard."""

    shape: tuple[int, ...]
    dtype: Any
    nbytes: int
    element_size_bytes: int

    def numel(self) -> int:
        return self.nbytes // self.element_size_bytes

    def element_size(self) -> int:
        return self.element_size_bytes


@dataclass(frozen=True)
class _NativeSlotReady:
    ring: NativeHostRing
    slot_index: int


@dataclass
class _NativeReadWindow:
    slot_index: int
    path: str
    file_offset: int
    arena_offset: int
    size_bytes: int


@dataclass
class _NativeReadTicket:
    bindings: list[_TensorBinding]
    windows: list[_NativeReadWindow]
    started_at: float


@dataclass(eq=False)
class _NativeCacheReadTicket:
    bindings: list[_TensorBinding]
    cached: list[tuple[Any, _NativeReadWindow]]
    ring_ticket: _NativeReadTicket | None = None
    active: bool = True


class _NativeShardTransport:
    """Bounded native shard source backed by registered host slots."""

    _WINDOW_BYTES = 64 * 2**20

    def __init__(self, library: NativeVmmLibrary, *, slot_size_bytes: int, slots: int, device_ordinal: int,
                 cache_slot_bytes: int = 0, cache_slots: int = 0, cache_process_limit_bytes: int = 0,
                 cache_model_windows: bool = False, cache_registered_limit_bytes: int = 0) -> None:
        if sys.platform not in {"win32", "linux"}:
            raise ModelArenaError("native shard transport requires Windows or native Linux")
        # Preserve the configured total capacity, but give each internal
        # transfer window its own CUDA event.  A completed 64 MiB window can
        # feed DMA while later windows are still being read from a shard.
        self.window_size_bytes = min(self._WINDOW_BYTES, slot_size_bytes)
        window_count = (slot_size_bytes * slots + self.window_size_bytes - 1) // self.window_size_bytes
        self._library = library
        self.cache = None
        self._cache_sources: dict[str, Any] = {}
        self._cache_dma_tickets: list[Any] = []
        self._cache_read_tickets: list[_NativeCacheReadTicket] = []
        self._batches = {}
        if cache_slot_bytes:
            self.cache = library.create_source_cache(
                slot_bytes=cache_slot_bytes, slots=cache_slots, device=device_ordinal,
                process_limit_bytes=cache_process_limit_bytes,
                model_windows=cache_model_windows,
                **({"persistent_buffer": True, "registered_limit_bytes": cache_registered_limit_bytes}
                   if cache_model_windows else {}),
            )
        try:
            self.ring = library.create_host_ring(self.window_size_bytes, window_count, device_ordinal=device_ordinal)
        except BaseException:
            if self.cache is not None:
                self.cache.close()
            raise
        self.size_bytes = slot_size_bytes
        self.slots = window_count
        self._next_slot = 0
        self._pending: dict[int, _NativeReadTicket] = {}
        self.read_seconds = 0.0
        self.read_bytes = 0
        self.slot_wait_seconds = 0.0
        self.slot_waits = 0
        self._source_stat_origin = self.ring.source_stats

    @property
    def source_stats(self) -> dict[str, int]:
        current = self.ring.source_stats
        return {
            key: current[key] if key == "workers" else current[key] - self._source_stat_origin[key]
            for key in current
        }

    def reset_stats(self) -> None:
        self.read_seconds = 0.0
        self.read_bytes = 0
        self.slot_wait_seconds = 0.0
        self.slot_waits = 0
        self._source_stat_origin = self.ring.source_stats

    def _source(self, path: Any):
        key = str(path)
        source = self._cache_sources.get(key)
        if source is None:
            source = self._library.open_source_file(key)
            self._cache_sources[key] = source
        return source

    def reap_cache_dma(self) -> None:
        pending = []
        for ticket in self._cache_dma_tickets:
            if ticket.dma_ready():
                ticket.close()
            else:
                pending.append(ticket)
        self._cache_dma_tickets = pending

    @property
    def cache_stats(self) -> dict[str, int]:
        return {} if self.cache is None else self.cache.snapshot()

    def _acquire_slot(self) -> int:
        slot = self._next_slot
        self._next_slot = (slot + 1) % self.slots
        if slot in self._pending:
            # Look-ahead is bounded by the ring.  When its producer laps a
            # still-reading slot, wait for that *one* raw read and retire the
            # speculation.  This is local transport backpressure, never a
            # device-wide synchronization or an unsafe host-buffer reuse.
            start = time.perf_counter()
            self.discard_ticket(self._pending[slot])
            self.slot_wait_seconds += time.perf_counter() - start
            self.slot_waits += 1
        if not self.ring.slot_ready(slot):
            start = time.perf_counter()
            self.ring.wait_slot(slot)
            self.slot_wait_seconds += time.perf_counter() - start
            self.slot_waits += 1
        return slot

    def begin_read(self, bindings: list[_TensorBinding]) -> _NativeReadTicket:
        if not bindings:
            raise ValueError("native shard read batch must not be empty")
        if any(binding.file_slice is None for binding in bindings):
            raise ModelArenaError("native shard transport requires a file-backed binding")
        window_count = sum(
            (binding.nbytes + self.window_size_bytes - 1) // self.window_size_bytes for binding in bindings
        )
        if window_count > self.slots:
            # A read-only ticket cannot recycle its own slots before upload.
            # Demand uploads use bounded sub-tickets; speculative reads skip
            # a batch that cannot retain all its windows concurrently.
            raise ModelArenaError("native shard read ticket exceeds the ring window count")
        if self.cache is not None:
            return self._begin_cache_read(bindings)
        return self._begin_windows(bindings, list(self._window_ranges(bindings)))

    def _begin_cache_read(self, bindings):
        # Native cache admission happens BEFORE bypass reads, never in Python.
        # Keep unsubmitted source tickets separate from DMA polling tickets.
        ranges = list(islice(self._cache_ranges(bindings), 65))
        bypass_count = sum((window.size_bytes + self.window_size_bytes - 1) // self.window_size_bytes for window in ranges)
        if len(ranges) > 64 or bypass_count > self.slots:
            raise ModelArenaError("cache speculation exceeds bounded descriptor/ring window count")
        ticket = _NativeCacheReadTicket(list(bindings), [])
        self._cache_read_tickets.append(ticket)
        bypass = []
        try:
            self.reap_cache_dma()
            for window in ranges:
                source_ticket = self.cache.request(self._source(window.path), window.file_offset, window.size_bytes)
                if source_ticket is not None:
                    ticket.cached.append((source_ticket, window))
                else:
                    for start in range(0, window.size_bytes, self.window_size_bytes):
                        bypass.append(_NativeReadWindow(-1, window.path, window.file_offset + start,
                            window.arena_offset + start, min(self.window_size_bytes, window.size_bytes - start)))
            if bypass:
                ticket.ring_ticket = self._begin_windows(bindings, bypass)
            return ticket
        except BaseException:
            # Ring partial enqueue stays owned by _pending until read drainage.
            self.discard_ticket(ticket)
            raise

    def _upload_cache_ticket(self, ticket, arena, stream):
        import torch

        try:
            # Retire raw bypass reads before a registration-refusal fallback
            # needs a ring slot; otherwise fallback could cancel this ticket.
            if ticket.ring_ticket is not None:
                self.upload_ticket(ticket.ring_ticket, arena, stream)
            for source_ticket, window in ticket.cached:
                # Retain even a failed enqueue until native close has proved
                # that no DMA can still reference these registered bytes.
                self._cache_dma_tickets.append(source_ticket)
                self._copy_cached_source(source_ticket, window, arena, stream)
            ready = torch.cuda.Event(enable_timing=False)
            ready.record(stream)
            ticket.active = False
            self._cache_read_tickets.remove(ticket)
            return ready
        except BaseException:
            for source, _window in ticket.cached:
                if any(source is pending for pending in self._cache_dma_tickets):
                    try:
                        source.close()  # Local event/stream drainage on failure only.
                    except Exception:
                        continue  # Still owned; never expose potentially live bytes.
                    self._cache_dma_tickets.remove(source)
            # Submitted tickets are now owned by DMA cleanup. Unsubmitted
            # tickets remain owned by this read ticket for cancellation/close.
            ticket.cached = [(source, window) for source, window in ticket.cached
                             if not any(source is pending for pending in self._cache_dma_tickets)]
            raise

    def _window_ranges(self, bindings: list[_TensorBinding]):
        """Describe file/arena windows without reserving or recycling slots."""
        for binding in bindings:
            if binding.file_slice is None:
                raise ModelArenaError("native shard transport requires a file-backed binding")
            remaining = binding.nbytes
            source_offset = binding.file_slice.offset_bytes
            arena_offset = binding.offset
            while remaining:
                chunk = min(remaining, self.window_size_bytes)
                yield _NativeReadWindow(-1, str(binding.file_slice.path), source_offset, arena_offset, chunk)
                remaining -= chunk
                source_offset += chunk
                arena_offset += chunk

    def _begin_windows(
        self, bindings: list[_TensorBinding], windows: list[_NativeReadWindow]
    ) -> _NativeReadTicket:
        started_at = time.perf_counter()
        ticket = _NativeReadTicket(list(bindings), [], started_at)
        for window in windows:
            window.slot_index = self._acquire_slot()
            ticket.windows.append(window)
            self._pending[window.slot_index] = ticket
            self.ring.read_file_async(
                window.slot_index, window.path, window.file_offset, size_bytes=window.size_bytes
            )
        return ticket

    def _upload_ring_batch(self, bindings: list[_TensorBinding], arena: NativeModelArena, stream: Any) -> Any:
        return self._upload_ring_windows(bindings, self._window_ranges(bindings), arena, stream)

    def _upload_ring_windows(self, bindings, ranges, arena, stream):
        # Payload bytes alone do not bound slots: two tiny tensors still use
        # two windows. Submit at most one ring's windows before recycling any
        # slot; _acquire_slot then waits only for that slot's previous DMA.
        windows: list[_NativeReadWindow] = []
        ready = None
        for window in ranges:
            windows.append(window)
            if len(windows) == self.slots:
                ready = self.upload_ticket(self._begin_windows(bindings, windows), arena, stream)
                windows = []
        if windows:
            ready = self.upload_ticket(self._begin_windows(bindings, windows), arena, stream)
        # All tickets use the same upload stream. The last completion token
        # covers earlier sub-tickets without a device-wide synchronization.
        return ready

    def _cache_ranges(self, bindings):
        for binding in bindings:
            source = binding.file_slice
            if source is None:
                raise ModelArenaError("native source cache requires a file-backed binding")
            remaining, offset, target = binding.nbytes, source.offset_bytes, binding.offset
            while remaining:
                size = remaining
                if self.cache.model_windows:
                    size = min(size, self.cache.slot_bytes - offset % self.cache.slot_bytes)
                yield _NativeReadWindow(-1, str(source.path), offset, target, size)
                offset += size
                target += size
                remaining -= size

    def upload_ticket(self, ticket: _NativeReadTicket, arena: NativeModelArena, stream: Any) -> _NativeSlotReady:
        if not self.ticket_active(ticket):
            raise ModelArenaError("native shard read ticket is no longer active")
        if isinstance(ticket, _NativeCacheReadTicket):
            return self._upload_cache_ticket(ticket, arena, stream)
        try:
            self.ring.copy_ticket_to_arena_async(
                [(window.slot_index, window.arena_offset, window.size_bytes) for window in ticket.windows],
                arena,
                stream=stream,
            )
            ready = _NativeSlotReady(self.ring, ticket.windows[-1].slot_index)
            self.read_seconds += time.perf_counter() - ticket.started_at
            self.read_bytes += sum(window.size_bytes for window in ticket.windows)
            return ready
        finally:
            for window in ticket.windows:
                self._pending.pop(window.slot_index, None)

    def _immutable_batch(self, bindings):
        key = tuple((binding.label, binding.offset, binding.nbytes) for binding in bindings)
        batch = self._batches.get(key)
        if batch is None:
            with _measure_batch_scope(self, "batch_transport_create"):
                batch = self.cache.create_batch([
                    (self._source(binding.file_slice.path), binding.file_slice.offset_bytes,
                     binding.nbytes, binding.offset) for binding in bindings
                ], self.ring, ring_slots=self.slots, ring_slot_bytes=self.window_size_bytes)
            self._batches[key] = batch
        return batch

    def prepare_sources(self, units):
        """Provide unit descriptors once; the native dispatcher owns window I/O."""
        if self.cache is None or not getattr(self.cache, "source_dispatcher", False):
            return
        with _measure_batch_scope(self, "source_prepare"):
            for unit in units:
                bindings = [binding for binding in unit.tensors if binding.file_slice is not None]
                if bindings:
                    self._immutable_batch(bindings).prepare_source()

    def upload_batch(self, bindings: list[_TensorBinding], arena: NativeModelArena, stream: Any) -> Any:
        if self.cache is not None and getattr(self.cache, "persistent_buffer", False):
            # Unit metadata is immutable. Per-window scheduling/recycling no
            # longer crosses Python/ctypes on every model pass.
            with _measure_batch_scope(self, "batch_transport_total"):
                with _measure_batch_scope(self, "batch_transport_prepare"):
                    batch = self._immutable_batch(bindings)
                with _measure_batch_scope(self, "batch_transport_submit"):
                    batch.submit(arena, stream)
                return batch
        if self.cache is None:
            return self._upload_ring_batch(bindings, arena, stream)
        import torch

        self.reap_cache_dma()
        cached = 0
        ring_ready = None
        ranges = iter(self._cache_ranges(bindings))
        while batch := list(islice(ranges, 64)):
            pending = []
            bypass = []
            # Schedule a bounded batch's reads first. Waiting immediately
            # after each request serialised the native worker pool.
            for window in batch:
                ticket = self.cache.request(self._source(window.path), window.file_offset, window.size_bytes)
                if ticket is None:
                    # Ring windows remain bounded independently of cache size.
                    for start in range(0, window.size_bytes, self.window_size_bytes):
                        bypass.append(_NativeReadWindow(-1, window.path, window.file_offset + start,
                            window.arena_offset + start, min(self.window_size_bytes, window.size_bytes - start)))
                    continue
                self._cache_dma_tickets.append(ticket)
                pending.append((ticket, window))
            for ticket, window in pending:
                # copy_async itself waits for only this ticket's read. Later
                # reads are already running while its DMA/previous compute run.
                self._copy_cached_source(ticket, window, arena, stream)
                cached += 1
            if bypass:
                ring_ready = self._upload_ring_windows(bindings, bypass, arena, stream)
        if not cached:
            return ring_ready
        # All copies share this upload stream. One final event covers cache
        # tickets and any ring bypass, while each ticket retains its own event.
        ready = torch.cuda.Event(enable_timing=False)
        ready.record(stream)
        return ready

    def _copy_cached_source(self, ticket, window, arena, stream):
        try:
            ticket.copy_to_arena_async(arena, window.arena_offset, stream)
        except NativeSourceError as error:
            if error.status != 8:
                raise
            # C++ refused registration. Bytes stay in persistent storage;
            # reuse existing native staging, with no file reread or tensor.
            slot = self._acquire_slot()
            ticket.copy_via_ring_async(self.ring, slot, arena, window.arena_offset, stream)

    def discard_ticket(self, ticket: _NativeReadTicket) -> None:
        """Retire a speculative source read that was made unnecessary."""
        if isinstance(ticket, _NativeCacheReadTicket):
            if not ticket.active:
                return
            if ticket.ring_ticket is not None:
                self.discard_ticket(ticket.ring_ticket)
            for source_ticket, _window in ticket.cached:
                source_ticket.close()  # Native cache continues owning pending I/O.
            ticket.active = False
            self._cache_read_tickets.remove(ticket)
            return
        if not any(self._pending.get(window.slot_index) is ticket for window in ticket.windows):
            return
        for window in ticket.windows:
            # A read owns this registered address until the native worker has
            # completed.  Retiring a speculative ticket therefore waits for
            # its source operation before exposing the slot to another batch.
            try:
                self.ring.wait_file_read(window.slot_index)
            finally:
                self._pending.pop(window.slot_index, None)

    def ticket_active(self, ticket: _NativeReadTicket) -> bool:
        if isinstance(ticket, _NativeCacheReadTicket):
            return ticket.active and (ticket.ring_ticket is None or self.ticket_active(ticket.ring_ticket))
        return bool(ticket.windows) and all(self._pending.get(window.slot_index) is ticket for window in ticket.windows)

    def close(self) -> None:
        for cache_ticket in list(self._cache_read_tickets):
            self.discard_ticket(cache_ticket)
        tickets: list[_NativeReadTicket] = []
        for ticket in self._pending.values():
            if all(ticket is not known for known in tickets):
                tickets.append(ticket)
        for ticket in tickets:
            self.discard_ticket(ticket)
        self._pending.clear()
        for ticket in self._cache_dma_tickets:
            ticket.close()
        self._cache_dma_tickets.clear()
        for batch in self._batches.values():
            batch.close()
        self._batches.clear()
        if self.cache is not None:
            self.cache.close()
            self.cache = None
        for source in self._cache_sources.values():
            source.close()
        self._cache_sources.clear()
        self.ring.close()


@dataclass
class _ModuleBinding:
    module_name: str
    module: Any
    tensors: list[_TensorBinding]
    stable_data: dict[str, Any]
    temporary_views: list[tuple[_TensorBinding, Any]] | None = None
    prefetched: bool = False
    prefetch_next: "_ModuleBinding | None" = None
    active: bool = False
    execution_buffer_active: bool = False
    execution_buffer_lease: Any | None = None
    execution_buffer_prefetch: Any | None = None
    prefetch_cancelled_bindings: int = 0
    guard_reentry: bool = False


def _component_output_borrows_weights(output, bindings):
    """Reject observable weight-view escape in the opt-in component contract."""
    import torch
    from collections.abc import Mapping

    ranges = []
    for binding in bindings:
        tensor = binding.owner._parameters[binding.parameter_name]
        storage = tensor.untyped_storage()
        ranges.append((tensor.device, storage.data_ptr(), storage.nbytes()))
    pending, seen = [output], set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            start, size = storage.data_ptr(), storage.nbytes()
            if any(value.device == device and size and count and start < address + count and address < start + size
                   for device, address, count in ranges):
                return True
        elif isinstance(value, Mapping):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
    return False


@dataclass
class _ObservedExecutionQueue:
    """Validated block order for one transformer invocation.

    A queue becomes authoritative only while the module tree follows the
    order observed at attachment. Any dynamic/re-entrant deviation disables
    look-ahead for that invocation rather than preloading an unsafe range.
    """

    states: list[_ModuleBinding]
    cursor: int = 0
    valid: bool = True
    # Experimental sub-block units: enumeration order need not be call order.
    # With ``learn`` the first pass records the observed order without
    # look-ahead; the next pass adopts it only if every unit ran exactly once,
    # otherwise the enumeration order (with the existing validation) is kept.
    learn: bool = False
    recorded: list | None = None
    learn_failed: bool = False
    learned: bool = False

    def begin_pass(self) -> None:
        if self.learn:
            if self.recorded is not None and not self.learn_failed:
                complete = len(self.recorded) == len(self.states) and all(
                    any(r is s for s in self.states) for r in self.recorded)
                if complete:
                    self.states = list(self.recorded)
                    self.learned = True
                else:
                    self.learn_failed = True
                self.learn = False
                self.recorded = None
            elif self.recorded is not None:
                self.learn = False
                self.recorded = None
            else:
                self.recorded = []
        self.cursor = 0
        self.valid = True

    def consume(self, state: _ModuleBinding) -> _ModuleBinding | None:
        # Auxiliary modules may execute before/between transformer blocks.
        # They are not part of this queue and must not invalidate the observed
        # block sequence merely by having their own hook.
        if state not in self.states:
            return None
        if self.learn and self.recorded is not None:
            # Learning pass: record, never look ahead. A repeated unit means a
            # dynamic or re-entrant order that is not learned.
            if any(r is state for r in self.recorded):
                self.learn_failed = True
            else:
                self.recorded.append(state)
            return None
        if not self.valid or self.cursor >= len(self.states) or self.states[self.cursor] is not state:
            self.valid = False
            return None
        self.cursor += 1
        return self.states[self.cursor] if self.cursor < len(self.states) else None


@dataclass(frozen=True)
class ModelArenaAttachReport:
    pattern: str
    attached_modules: int
    attached_tensors: int
    execution_units: int
    payload_bytes: int
    virtual_bytes: int
    page_bytes: int
    adapter: "ModelArenaModuleAdapter"


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _execution_order_key(module_name: str) -> int:
    """Return the numeric suffix used by an observed execution unit."""
    return int(module_name.rsplit(".", 1)[-1])


class ModelArenaWeightStore:
    """Owns one native model arena plus immutable CPU source weights."""

    def __init__(
        self,
        library: NativeVmmLibrary,
        *,
        payload_bytes: int,
        virtual_size_multiplier: float = 1.0,
        resident_budget_bytes: int | None = None,
        page_size_bytes: int = 0,
        device_ordinal: int = 0,
        backing: str = "device",
        staging_pool: Any | None = None,
        file_slices: Any | None = None,
        host_cache_bytes: int = 0,
        native_transport_slot_size_bytes: int = 0,
        native_transport_slots: int = 2,
        native_source_cache_slot_bytes: int = 0,
        native_source_cache_slots: int = 0,
        native_source_cache_process_limit_bytes: int = 0,
        native_source_cache_model_windows: bool = False,
        native_source_cache_registered_limit_bytes: int = 0,
    ) -> None:
        if payload_bytes <= 0:
            raise ValueError("payload_bytes must be positive")
        if virtual_size_multiplier < 1:
            raise ValueError("virtual_size_multiplier must be >= 1")
        virtual_size = max(payload_bytes, int(payload_bytes * virtual_size_multiplier))
        self.native: NativeModelArena = library.create_model_arena(
            virtual_size,
            resident_budget_bytes=resident_budget_bytes,
            page_size_bytes=page_size_bytes,
            device_ordinal=device_ordinal,
            backing=backing,
        )
        self._library = library
        self._device_ordinal = device_ordinal
        self._backing = backing
        self._windows_memory_samples: list[dict[str, int | str]] = []
        self._debug_faults = os.environ.get("DDO_VMM_DEBUG_FAULTS", "").strip().lower() in {"1", "true", "yes", "on"}
        self._buffers: list[Any] = []
        self._host_registry = None
        self._staging_pool = staging_pool
        self._file_slices = file_slices
        self.transfer_buffer = DeviceTransferBuffer(device_ordinal)
        self.execution_buffer = None
        self._streaming_enabled = bool(native_source_cache_model_windows)
        import torch

        self._upload_stream = torch.cuda.Stream(device=device_ordinal)
        self._native_shard_transport = None
        if native_transport_slot_size_bytes:
            if staging_pool is not None:
                raise ValueError("native shard transport and PyTorch staging pool are mutually exclusive")
            self._native_shard_transport = _NativeShardTransport(
                library,
                slot_size_bytes=native_transport_slot_size_bytes,
                slots=native_transport_slots,
                device_ordinal=device_ordinal,
                cache_slot_bytes=native_source_cache_slot_bytes,
                cache_slots=native_source_cache_slots,
                cache_process_limit_bytes=native_source_cache_process_limit_bytes,
                cache_model_windows=native_source_cache_model_windows,
                cache_registered_limit_bytes=native_source_cache_registered_limit_bytes,
            )
        if host_cache_bytes < 0:
            raise ValueError("host_cache_bytes must be non-negative")
        # Direct registration of loader-owned CPU storage is an explicit
        # high-RAM transport mode.  The default remains the bounded file
        # staging path; do not implicitly register the complete model.
        if host_cache_bytes > 0 and RegisteredHostWeightRegistry is not None:
            self._host_registry = RegisteredHostWeightRegistry(
                device_ordinal,
                max_bytes=host_cache_bytes,
            )
        self._runtime_stats: dict[str, int] = {
            "faults": 0,
            "signature_hits": 0,
            "uploads": 0,
            "file_uploads": 0,
            "upload_bytes": 0,
            "fault_failures": 0,
            "guard_admission_failures": 0,
            "cuda_admission_failures": 0,
            "capacity_admission_failures": 0,
            "capacity_event_waits": 0,
            "temporary_tensors": 0,
            "temporary_bytes": 0,
            "execution_buffer_uploads": 0,
            "execution_buffer_bytes": 0,
            "dma_enqueue_seconds": 0,
            "dma_enqueue_bytes": 0,
            "compute_wait_events": 0,
            "prefetch_attempts": 0,
            "prefetch_hits": 0,
            "prefetch_skips": 0,
            "prefetch_resident_skips": 0,
            "gpu_prefetch_units": 0,
            "gpu_prefetch_hits": 0,
            "gpu_prefetch_capacity_skips": 0,
            "gpu_prefetch_cancelled_units": 0,
            "execution_plan_passes": 0,
            "execution_plan_mismatches": 0,
        }
        self._source_stats: dict[str, int] = {
            "file_backed_bindings": 0,
            "file_backed_bytes": 0,
            "cpu_source_bindings": 0,
            "cpu_source_bytes": 0,
        }
        self.capture_windows_memory_sample("arena_created")

    def capture_windows_memory_sample(self, phase: str) -> None:
        """Append a best-effort WDDM observation without influencing residency."""

        try:
            info = self._library.windows_memory_info(self._device_ordinal)
        except NativeVmmError:
            return
        arena_info = self.native.info
        self._windows_memory_samples.append(
            {
                "phase": phase,
                "local_budget_bytes": info.local_budget_bytes,
                "local_usage_bytes": info.local_usage_bytes,
                "nonlocal_budget_bytes": info.nonlocal_budget_bytes,
                "nonlocal_usage_bytes": info.nonlocal_usage_bytes,
                "available_physical_bytes": info.available_physical_bytes,
                "available_commit_bytes": info.available_commit_bytes,
                "arena_resident_bytes": arena_info.resident_size,
                "arena_virtual_bytes": arena_info.virtual_size,
                "arena_pressure_evictions": self.native.residency_counters["pressure_evictions"],
            }
        )

    @property
    def address(self) -> int:
        return self._immutable_arena_layout()[0]

    @property
    def page_bytes(self) -> int:
        return self._immutable_arena_layout()[1]

    @property
    def virtual_bytes(self) -> int:
        return self._immutable_arena_layout()[2]

    def _immutable_arena_layout(self):
        # Only fixed geometry is cached. Residency, signatures, pressure and
        # fault ceilings must continue to come from the live native runtime.
        layout = getattr(self, "_arena_layout", None)
        if layout is None:
            info = self.native.info
            layout = self._arena_layout = (info.address, info.page_size, info.virtual_size)
        return layout

    def reset_runtime_stats(self) -> None:
        for key in self._runtime_stats:
            self._runtime_stats[key] = 0
        if self._native_shard_transport is not None:
            self._native_shard_transport.reset_stats()

    def runtime_stats(self) -> dict[str, int]:
        info = self.native.info
        result = dict(self._runtime_stats)
        result.update(self._source_stats)
        result.update({f"torch_allocator_{key}": value for key, value in
                       getattr(self.native, "torch_allocator_stats", {}).items()})
        result["resident_bytes"] = info.resident_size
        buffer = getattr(self, "execution_buffer", None)
        result["execution_buffer_capacity_bytes"] = buffer.capacity_bytes if buffer is not None else 0
        if isinstance(buffer, NativeExecutionBuffers):
            result.update({f"execution_backing_{key}": value
                           for key, value in buffer.pool.reclaim_stats().items()})
        result["execution_buffer_resident_bytes"] = buffer.resident_bytes if isinstance(buffer, NativeExecutionBuffers) else buffer.native.info.resident_size if buffer is not None else 0
        result["execution_buffer_slots"] = buffer.pool.slots if isinstance(buffer, NativeExecutionBuffers) else 1 if buffer is not None else 0
        result["execution_buffer_slot_bytes"] = buffer.pool.capacity_bytes if isinstance(buffer, NativeExecutionBuffers) else buffer.capacity_bytes if buffer is not None else 0
        result["total_weight_resident_bytes"] = result["resident_bytes"] + result["execution_buffer_resident_bytes"]
        lifetime = getattr(self, "_component_lifetime_buffer", None)
        if lifetime is not None:
            result["component_lifetime_capacity_bytes"] = lifetime.capacity_bytes
            result["component_lifetime_resident_bytes"] = lifetime.native.info.resident_size
            result["total_weight_resident_bytes"] += result["component_lifetime_resident_bytes"]
        result["fault_ceiling_bytes"] = info.fault_ceiling
        result["virtual_bytes"] = info.virtual_size
        result["page_bytes"] = info.page_size
        result.update({f"arena_{key}": value for key, value in self.native.residency_counters.items()})
        if self._staging_pool is not None:
            staging = self._staging_pool.transport_stats
            result.update(
                {
                    "staging_slot_wait_seconds": staging.slot_reuse_wait_seconds,
                    "staging_slot_waits": staging.slot_reuse_waits,
                    "staging_source_copy_seconds": staging.source_copy_seconds,
                    "staging_source_copy_bytes": staging.source_copy_bytes,
                    "shard_read_seconds": staging.shard_read_seconds,
                    "shard_read_bytes": staging.shard_read_bytes,
                }
            )
        if self._native_shard_transport is not None:
            result.update({f"native_transport_{name}": value for name, value in
                           getattr(self._native_shard_transport, "_batch_timings", {}).items()})
            native_source = self._native_shard_transport.source_stats
            result.update(
                {
                    "native_transport_slot_wait_seconds": self._native_shard_transport.slot_wait_seconds,
                    "native_transport_slot_waits": self._native_shard_transport.slot_waits,
                    "native_transport_read_seconds": self._native_shard_transport.read_seconds,
                    "native_transport_read_bytes": self._native_shard_transport.read_bytes,
                    "native_transport_source_workers": native_source["workers"],
                    "native_transport_source_read_chunks": native_source["read_chunks"],
                    "native_transport_source_read_bytes": native_source["read_bytes"],
                    "native_transport_source_read_failures": native_source["read_failures"],
                }
            )
            if self._native_shard_transport.cache is not None:
                result.update({f"source_cache_{name}": value for name, value in self._native_shard_transport.cache_stats.items()})
        return result

    def metrics_snapshot(
        self,
        *,
        configuration: dict[str, Any] | None = None,
        setup: dict[str, Any] | None = None,
    ) -> Any:
        """Return the versioned DDO report consumed by host runners."""

        # Keep this optional at import time so the focused native package can
        # still run standalone contracts without installing the DDO host.
        from diffusers_dynamic_offloader.metrics import DynamicOffloadMetrics

        stats = self.runtime_stats()
        return DynamicOffloadMetrics(
            route="vmm_fast_runtime",
            configuration=configuration or {},
            setup=setup or {},
            runtime={
                "vmm": {f"fast_{key}": value for key, value in stats.items()},
                "wddm": {"samples": list(self._windows_memory_samples)},
            },
        )

    @contextmanager
    def view_setup(self, spans):
        """Coalesce transient view mappings during single-threaded attach only.

        Never used for compute or residency. Native prepare aliases its one
        bootstrap allocation; all transient mappings close before hooks run.
        Release errors propagate rather than allowing an unsafe runtime.
        """
        if getattr(self, "_view_setup_state", None) is not None:
            raise ModelArenaError("view setup cannot be nested")
        state = {"spans": iter(spans), "active": None, "current": None}
        self._view_setup_state = state
        try:
            yield
        finally:
            try:
                if state["active"] is not None:
                    self._release_setup_view(*state["active"])
            finally:
                self._view_setup_state = None

    def _prepare_setup_view(self, offset, size):
        self._runtime_stats["view_prepare_calls"] = self._runtime_stats.get("view_prepare_calls", 0) + 1
        start = time.perf_counter()
        try:
            self.native.prepare_view_range(offset, size)
        finally:
            self._runtime_stats["view_prepare_seconds"] = (
                self._runtime_stats.get("view_prepare_seconds", 0.0) + time.perf_counter() - start
            )

    def _release_setup_view(self, offset, size):
        self._runtime_stats["view_release_calls"] = self._runtime_stats.get("view_release_calls", 0) + 1
        start = time.perf_counter()
        try:
            self.native.release_view_range(offset, size)
        finally:
            self._runtime_stats["view_release_seconds"] = (
                self._runtime_stats.get("view_release_seconds", 0.0) + time.perf_counter() - start
            )

    def _setup_view_is_prepared(self, offset, size):
        if size <= 0:
            return False
        state = getattr(self, "_view_setup_state", None)
        if state is None:
            return False
        current = state["current"]
        if current is None or offset >= current[0] + current[1]:
            if state["active"] is not None:
                self._release_setup_view(*state["active"])
                state["active"] = None
            current = next(state["spans"], None)
            if current is None or not (current[0] <= offset and offset + size <= sum(current)):
                raise ModelArenaError("view setup layout does not match tensor offsets")
            state["current"] = current
            try:
                self._prepare_setup_view(*current)
            except NativeVmmError:
                # Retain the individual-tensor diagnostic/fallback contract.
                # A refused span does not force all its tensors to CPU.
                return False
            state["active"] = current
        if not (current[0] <= offset and offset + size <= sum(current)):
            raise ModelArenaError("tensor crosses its view setup unit")
        return state["active"] is not None

    def stable_tensor(self, *, offset: int, source: Any) -> tuple[Any, int]:
        size = source.numel() * source.element_size()
        prepared = self._setup_view_is_prepared(offset, size)
        try:
            if not prepared:
                self._prepare_setup_view(offset, size)
        except NativeVmmError:
            # If a CUDA view cannot be represented, the module hook retains
            # the source and takes its temporary CUDA fallback.
            return source, None
        try:
            start = time.perf_counter()
            try:
                byte_tensor, buffer = cuda_uint8_tensor_from_address(
                    self.address + offset,
                    size,
                    owner=self.native,
                    backing=self._backing,
                )
            finally:
                self._runtime_stats["view_create_calls"] = self._runtime_stats.get("view_create_calls", 0) + 1
                self._runtime_stats["view_create_seconds"] = (
                    self._runtime_stats.get("view_create_seconds", 0.0) + time.perf_counter() - start
                )
        finally:
            if not prepared:
                self._release_setup_view(offset, size)
        self._buffers.append(buffer)
        stable = byte_tensor.view(source.dtype).view(tuple(source.shape))
        # This only creates a persistent virtual view. Runtime faults own
        # physical residency and establish the first content signature.
        return stable, None

    def _source_for_upload(self, binding: _TensorBinding) -> Any:
        if binding.file_slice is None:
            return binding.source
        if self._staging_pool is None:
            raise ModelArenaError("file-backed weight requires a staging pool")
        try:
            self._staging_pool.begin_upload()
            source = self._staging_pool.stage_file_slice(binding.source, binding.file_slice)
            self._runtime_stats["file_uploads"] += 1
            return source
        except (OSError, RuntimeError, ValueError) as error:
            raise ModelArenaError("unable to stage file-backed weight") from error

    def fault_and_copy_if_needed(self, binding: _TensorBinding) -> Any | None:
        self._runtime_stats["faults"] += 1
        if self._debug_faults:
            print(f"[ddo-vmm-fast] fault begin tensor={binding.label} bytes={binding.nbytes}", flush=True)
        signature = self.native.fault_range(binding.offset, binding.nbytes)
        if self._debug_faults:
            print(f"[ddo-vmm-fast] fault mapped tensor={binding.label} signature={signature}", flush=True)
        if binding.signature == signature:
            self._runtime_stats["signature_hits"] += 1
            return binding.ready_event
        if binding.file_slice is not None and self._native_shard_transport is not None:
            import torch

            start = time.perf_counter()
            with torch.cuda.stream(self._upload_stream):
                ready_event = self._native_shard_transport.upload_batch([binding], self.native, self._upload_stream)
            self._runtime_stats["dma_enqueue_seconds"] += time.perf_counter() - start
            self._runtime_stats["dma_enqueue_bytes"] += binding.nbytes
            self._runtime_stats["file_uploads"] += 1
            self._runtime_stats["uploads"] += 1
            self._runtime_stats["upload_bytes"] += binding.nbytes
            binding.signature = signature
            binding.ready_event = ready_event
            return ready_event
        source = self._source_for_upload(binding)
        if self._debug_faults:
            print(f"[ddo-vmm-fast] fault staged tensor={binding.label}", flush=True)
        import torch

        start = time.perf_counter()
        with torch.cuda.stream(self._upload_stream):
            if (
                binding.file_slice is None
                and self._host_registry is not None
                and self._host_registry.register(source)
            ):
                self.native.copy_from_host_async(binding.offset, source, stream=self._upload_stream)
            else:
                if source.dtype == binding.stable.dtype:
                    binding.stable.copy_(source, non_blocking=True)
                else:
                    converted = source.to(dtype=binding.stable.dtype, device=binding.stable.device)
                    binding.stable.copy_(converted, non_blocking=True)
            ready_event = torch.cuda.Event(enable_timing=False)
            ready_event.record(self._upload_stream)
        self._runtime_stats["dma_enqueue_seconds"] += time.perf_counter() - start
        self._runtime_stats["dma_enqueue_bytes"] += binding.nbytes
        if self._staging_pool is not None and self._staging_pool.has_active_upload:
            self._staging_pool.mark_upload(ready_event)
        binding.signature = signature
        binding.ready_event = ready_event
        if self._debug_faults:
            print(f"[ddo-vmm-fast] fault queued tensor={binding.label}", flush=True)
        self._runtime_stats["uploads"] += 1
        self._runtime_stats["upload_bytes"] += binding.nbytes
        return ready_event

    def fault_and_copy_group_if_needed(self, bindings: list[_TensorBinding]) -> list[Any | None]:
        """Materialize one execution unit with one file-staging slot/event."""

        if not bindings:
            return []
        if (
            (self._staging_pool is None and self._native_shard_transport is None)
            or any(binding.file_slice is None for binding in bindings)
        ):
            return [self.fault_and_copy_if_needed(binding) for binding in bindings]
        self._runtime_stats["faults"] += len(bindings)
        signatures: list[int] = []
        faulted: list[_TensorBinding] = []
        try:
            for binding in bindings:
                signatures.append(self.native.fault_range(binding.offset, binding.nbytes))
                faulted.append(binding)
        except NativeVmmError:
            for binding in reversed(faulted):
                with suppress(NativeVmmError):
                    self.unpin(binding)
            raise
        try:
            return self._upload_faulted_group(bindings, signatures)
        except Exception:
            # A later batch may fail after earlier DMAs were submitted. Drain
            # only our upload stream before releasing any destination pins.
            # If draining fails, retain the pins and propagate the CUDA error.
            self._upload_stream.synchronize()
            for binding in bindings:
                binding.signature = None
                binding.ready_event = None
            for binding in reversed(bindings):
                self.unpin(binding)
            raise

    def _upload_faulted_group(self, bindings: list[_TensorBinding], signatures: list[int]) -> list[Any | None]:
        """Upload a successfully pinned unit; its caller owns failure cleanup."""
        upload_indexes = [
            index for index, (binding, signature) in enumerate(zip(bindings, signatures, strict=True))
            if binding.signature != signature
        ]
        if not upload_indexes:
            self._discard_unused_native_prefetches(bindings)
            self._runtime_stats["signature_hits"] += len(bindings)
            return [binding.ready_event for binding in bindings]
        import torch

        preloaded_indexes = [
            index for index in upload_indexes if bindings[index].prefetched_source is not None
        ]
        if self._staging_pool is None:
            preloaded_indexes = []
        if preloaded_indexes and not self._staging_pool.has_active_upload:
            # A re-entrant model path may have consumed the ring slot before
            # this block reaches its hook. Discard the speculative view and
            # reload normally rather than associating it with another upload.
            for index in preloaded_indexes:
                bindings[index].prefetched_source = None
                bindings[index].prefetch_reads = None
            preloaded_indexes = []
        if preloaded_indexes:
            # A speculative file read owns the active staging slot until its
            # consumer queues this DMA. It is intentionally one subbatch.
            for index in preloaded_indexes:
                for future in bindings[index].prefetch_reads or []:
                    future.result()
            start = time.perf_counter()
            with torch.cuda.stream(self._upload_stream):
                for index in preloaded_indexes:
                    bindings[index].stable.copy_(bindings[index].prefetched_source, non_blocking=True)
                ready_event = torch.cuda.Event(enable_timing=False)
                ready_event.record(self._upload_stream)
            self._runtime_stats["dma_enqueue_seconds"] += time.perf_counter() - start
            self._runtime_stats["dma_enqueue_bytes"] += sum(bindings[index].nbytes for index in preloaded_indexes)
            self._staging_pool.mark_upload(ready_event)
            for index in preloaded_indexes:
                bindings[index].signature = signatures[index]
                bindings[index].ready_event = ready_event
                bindings[index].prefetched_source = None
                bindings[index].prefetch_reads = None
            self._runtime_stats["prefetch_hits"] += len(preloaded_indexes)
            upload_indexes = [index for index in upload_indexes if index not in preloaded_indexes]
        # A transformer execution unit can be larger than a bounded staging
        # slot. Keep its native ranges pinned as one unit, but enqueue its
        # immutable source in slot-sized batches. This is transport batching,
        # not a Python residency policy.
        batches: list[list[int]] = []
        batch: list[int] = []
        batch_bytes = 0
        for index in upload_indexes:
            binding_bytes = bindings[index].nbytes
            slot_size = (
                self._staging_pool.size_bytes
                if self._staging_pool is not None
                else self._native_shard_transport.size_bytes
            )
            if binding_bytes > slot_size:
                if self._native_shard_transport is None:
                    raise ModelArenaError(
                        f"file-backed tensor {bindings[index].label} exceeds the bounded staging slot"
                    )
                if batch:
                    batches.append(batch)
                    batch = []
                    batch_bytes = 0
                # The native transport splits one immutable file range across
                # event-protected windows; its logical batch boundary remains
                # unchanged for the adapter and runner.
                batches.append([index])
                continue
            if batch and batch_bytes + binding_bytes > slot_size:
                batches.append(batch)
                batch = []
                batch_bytes = 0
            batch.append(index)
            batch_bytes += binding_bytes
        if batch:
            batches.append(batch)

        for batch in batches:
            if self._native_shard_transport is not None:
                ticket = batch and bindings[batch[0]].native_prefetch_ticket
                if ticket is not None and not self._native_shard_transport.ticket_active(ticket):
                    self._native_shard_transport.discard_ticket(ticket)
                    for candidate in bindings:
                        if candidate.native_prefetch_ticket is ticket:
                            candidate.native_prefetch_ticket = None
                    ticket = None
                if ticket is not None and all(
                    bindings[index].native_prefetch_ticket is ticket for index in batch
                ):
                    for index in batch:
                        bindings[index].native_prefetch_ticket = None
                    self._runtime_stats["prefetch_hits"] += len(batch)
                else:
                    # A signature change can make the real consumer batch
                    # differ from the speculative one.  Finish and retire its
                    # raw read before reusing a ring slot; it must never leak
                    # into a later unit or force a generic fallback.
                    if ticket is not None:
                        for candidate in bindings:
                            if candidate.native_prefetch_ticket is ticket:
                                candidate.native_prefetch_ticket = None
                        self._native_shard_transport.discard_ticket(ticket)
                    ticket = None
                start = time.perf_counter()
                with _measure_upload_stream(self._upload_stream, self._runtime_stats):
                    ready_token = (
                        self._native_shard_transport.upload_ticket(ticket, self.native, self._upload_stream)
                        if ticket is not None
                        else self._native_shard_transport.upload_batch(
                            [bindings[index] for index in batch], self.native, self._upload_stream
                        )
                    )
                self._runtime_stats["dma_enqueue_seconds"] += time.perf_counter() - start
                self._runtime_stats["dma_enqueue_bytes"] += sum(bindings[index].nbytes for index in batch)
                for index in batch:
                    bindings[index].signature = signatures[index]
                    bindings[index].ready_event = ready_token
                continue
            self._staging_pool.begin_upload()
            try:
                sources = self._staging_pool.stage_file_slices(
                    [(bindings[index].source, bindings[index].file_slice) for index in batch]
                )
            except (OSError, RuntimeError, ValueError) as error:
                raise ModelArenaError("unable to stage file-backed execution unit") from error
            start = time.perf_counter()
            with torch.cuda.stream(self._upload_stream):
                for index, source in zip(batch, sources, strict=True):
                    bindings[index].stable.copy_(source, non_blocking=True)
                ready_event = torch.cuda.Event(enable_timing=False)
                ready_event.record(self._upload_stream)
            self._runtime_stats["dma_enqueue_seconds"] += time.perf_counter() - start
            self._runtime_stats["dma_enqueue_bytes"] += sum(bindings[index].nbytes for index in batch)
            self._staging_pool.mark_upload(ready_event)
            for index in batch:
                bindings[index].signature = signatures[index]
                bindings[index].ready_event = ready_event
        self._runtime_stats["file_uploads"] += len(upload_indexes)
        self._runtime_stats["uploads"] += len(upload_indexes)
        self._runtime_stats["upload_bytes"] += sum(bindings[index].nbytes for index in upload_indexes)
        self._discard_unused_native_prefetches(bindings)
        return [binding.ready_event for binding in bindings]

    def _discard_unused_native_prefetches(self, bindings: list[_TensorBinding]) -> None:
        if self._native_shard_transport is None:
            return
        tickets: list[_NativeReadTicket] = []
        for binding in bindings:
            ticket = binding.native_prefetch_ticket
            binding.native_prefetch_ticket = None
            if ticket is not None and all(ticket is not known for known in tickets):
                tickets.append(ticket)
        for ticket in tickets:
            self._native_shard_transport.discard_ticket(ticket)

    def wait_for_compute(self, ready_event: Any | None) -> None:
        """Queue, but never host-synchronize, the dependency on an upload."""
        if isinstance(ready_event, NativeSourceBatch):
            import torch

            ready_event.wait_on_stream(torch.cuda.current_stream(self._device_ordinal))
            self._runtime_stats["compute_wait_events"] += 1
            return
        if ready_event is None:
            return
        if isinstance(ready_event, _NativeSlotReady):
            import torch

            ready_event.ring.wait_slot_async(
                ready_event.slot_index,
                stream=torch.cuda.current_stream(self._device_ordinal),
            )
            self._runtime_stats["compute_wait_events"] += 1
            return
        import torch

        torch.cuda.current_stream(self._device_ordinal).wait_event(ready_event)
        self._runtime_stats["compute_wait_events"] += 1

    def temporary_source(self, binding: _TensorBinding) -> Any:
        """Materialize a source only for the short-lived fallback view."""
        if binding.file_slice is not None and self._native_shard_transport is not None and self._staging_pool is None:
            import torch

            raw = bytearray(binding.nbytes)
            self._native_shard_transport._source(binding.file_slice.path).read_into(
                raw, binding.file_slice.offset_bytes
            )
            return torch.frombuffer(raw, dtype=torch.uint8).view(binding.source.dtype).view(tuple(binding.source.shape))
        return self._source_for_upload(binding)

    def temporary_cuda_source(self, binding: _TensorBinding) -> tuple[Any, Any]:
        """Transfer a fallback source through the same protected upload slot."""
        import torch

        source = self.temporary_source(binding)
        start = time.perf_counter()
        with torch.cuda.stream(self._upload_stream):
            # A transient pageable file source must finish its H2D copy before
            # its bytearray can leave this function. Normal pinned sources keep
            # their asynchronous transport behavior.
            pageable_file_source = binding.file_slice is not None and self._native_shard_transport is not None and self._staging_pool is None
            temporary = source.to(device=f"cuda:{self._device_ordinal}", non_blocking=not pageable_file_source)
            ready_event = torch.cuda.Event(enable_timing=False)
            ready_event.record(self._upload_stream)
        self._runtime_stats["dma_enqueue_seconds"] += time.perf_counter() - start
        self._runtime_stats["dma_enqueue_bytes"] += binding.nbytes
        if self._staging_pool is not None and self._staging_pool.has_active_upload:
            self._staging_pool.mark_upload(ready_event)
        return temporary, ready_event

    def unpin(self, binding: _TensorBinding) -> None:
        self.native.unpin_range(binding.offset, binding.nbytes)

    def record_fault_failure(self, error: BaseException | None = None) -> None:
        self._runtime_stats["fault_failures"] += 1
        message = str(error)
        if "memory guard" in message:
            self._runtime_stats["guard_admission_failures"] += 1
        elif "CUDA allocation pressure" in message:
            self._runtime_stats["cuda_admission_failures"] += 1
        elif "resident capacity" in message:
            self._runtime_stats["capacity_admission_failures"] += 1

    def record_temporary(self, nbytes: int) -> None:
        self._runtime_stats["temporary_tensors"] += 1
        self._runtime_stats["temporary_bytes"] += nbytes

    def record_capacity_event_wait(self) -> None:
        self._runtime_stats["capacity_event_waits"] += 1

    def collect_completed_events(self, *, wait: bool = False) -> None:
        # The model-address arena currently unpins at post-forward after PyTorch
        # schedules the module work. Keep this method for the runner wrapper's
        # existing cleanup interface.
        del wait

    def close(self) -> None:
        # Also protects destinations retained after a failed upload cleanup.
        if self._upload_stream is not None:
            self._upload_stream.synchronize()
        if self.execution_buffer is not None:
            self.execution_buffer.close()
            self.execution_buffer = None
        self._buffers.clear()
        self.transfer_buffer.close()
        self._upload_stream = None
        if self._host_registry is not None:
            self._host_registry.close()
            self._host_registry = None
        if self._native_shard_transport is not None:
            self._native_shard_transport.close()
            self._native_shard_transport = None
        self.native.close()


class ModelArenaModuleAdapter:
    """Attach dense linear parameters to one model-address arena."""

    def __init__(self, store: ModelArenaWeightStore) -> None:
        self.store = store
        self._handles: list[Any] = []
        self._bindings: list[_ModuleBinding] = []
        self._temporary_lifetimes: list[tuple[Any, list[tuple[_TensorBinding, Any]]]] = []
        self._pending_unpins: list[tuple[Any, list[_TensorBinding]]] = []
        self._address_bindings: dict[int, _TensorBinding] = {}
        self._block_bindings: dict[str, list[_TensorBinding]] = {}
        self._execution_units: list[_ModuleBinding] = []
        self._execution_queue: _ObservedExecutionQueue | None = None

    def cpu_source_intervals(self) -> dict[str, Any]:
        """Snapshot retained CPU payload ranges without copying or releasing them.

        Caller must own an exclusive pre-execution boundary. Returned integers
        do not retain tensors and do not establish DMA completion or alias safety.
        """
        import torch

        intervals = []
        for module in self._bindings:
            for binding in module.tensors:
                if binding.file_slice is not None:
                    continue
                source = binding.source
                if (type(source) not in (torch.Tensor, torch.nn.Parameter)
                        or source.device.type != "cpu" or not source.is_contiguous()
                        or source.layout != torch.strided
                        or source.numel() * source.element_size() != binding.nbytes):
                    raise ModelArenaError("CPU source interval observation is unsupported")
                if binding.nbytes:
                    intervals.append((source.data_ptr(), binding.nbytes))
                else:
                    raise ModelArenaError("Empty CPU source interval is unsupported")
        return {"schema_version": 1, "binding_count": len(intervals), "intervals": intervals}

    def _collect_temporary_lifetimes(self, *, wait: bool = False) -> None:
        pending: list[tuple[Any, list[tuple[_TensorBinding, Any]]]] = []
        for event, views in self._temporary_lifetimes:
            if wait and not event.query():
                event.synchronize()
            if not event.query():
                pending.append((event, views))
        self._temporary_lifetimes = pending
        index = 0
        while index < len(self._pending_unpins):
            event, bindings = self._pending_unpins[index]
            if wait and not event.query():
                event.synchronize()
            if event.query():
                self._release_pending_unpin(index)
            else:
                index += 1

    def _release_pending_unpin(self, index: int) -> None:
        """Commit each successful release; failures retain remaining owners."""
        event, bindings = self._pending_unpins[index]
        for position, binding in enumerate(bindings):
            remaining = bindings[position + 1:]
            self.store.unpin(binding)
            self._pending_unpins[index] = (event, remaining)
        del self._pending_unpins[index]

    def _retire_oldest_pending_unpin(self) -> bool:
        """Release one completed execution range when capacity is exhausted.

        This is deliberately narrower than a device-wide synchronization: it
        waits only for the oldest module event that protects arena pages, then
        lets the native LRU choose the reusable physical pages.
        """

        if not self._pending_unpins:
            return False
        event, bindings = self._pending_unpins[0]
        if self.store._debug_faults:
            print(
                "[ddo-vmm-fast] capacity wait begin "
                f"tensor={bindings[0].label if bindings else '<empty>'} ranges={len(bindings)}",
                flush=True,
            )
        if not event.query():
            event.synchronize()
        self._release_pending_unpin(0)
        self.store.record_capacity_event_wait()
        if self.store._debug_faults:
            print(
                "[ddo-vmm-fast] capacity wait released "
                f"tensor={bindings[0].label if bindings else '<empty>'}",
                flush=True,
            )
        return True

    def _fault_binding_with_capacity_retry(self, binding: _TensorBinding) -> Any | None:
        while True:
            try:
                return self.store.fault_and_copy_if_needed(binding)
            except Exception as error:
                if "resident capacity" not in str(error):
                    raise
                if self.store._debug_faults:
                    print(
                        "[ddo-vmm-fast] capacity retry "
                        f"tensor={binding.label} reason={error!r}",
                        flush=True,
                    )
                if not self._retire_oldest_pending_unpin():
                    raise

    def _fault_block(self, bindings: list[_TensorBinding]) -> None:
        """Prepare all weights belonging to one execution block."""
        pinned = []
        try:
            for binding in bindings:
                self._fault_binding_with_capacity_retry(binding)
                pinned.append(binding)
        except Exception:
            for binding in reversed(pinned):
                self.store.unpin(binding)
            raise

    def _fault_execution_unit_with_capacity_retry(self, bindings: list[_TensorBinding]) -> list[Any | None]:
        """Fault a coalesced unit, waiting only for its oldest live peer."""

        while True:
            try:
                return self.store.fault_and_copy_group_if_needed(bindings)
            except Exception as error:
                if "resident capacity" not in str(error):
                    raise
                if self.store._debug_faults:
                    print(
                        "[ddo-vmm-fast] capacity retry "
                        f"unit={bindings[0].label if bindings else '<empty>'} reason={error!r}",
                        flush=True,
                    )
                if not self._retire_oldest_pending_unpin():
                    raise

    def _cancel_gpu_prefetches(self) -> None:
        """Release abandoned look-ahead only after its upload has completed."""
        pending = [state for state in self._execution_units if state.prefetched]
        if not pending:
            return
        # Ring tokens refer to reusable slot events, so the owned stream is
        # the unambiguous fence when cancelling a whole prefetched unit.
        if any(state.execution_buffer_prefetch is None for state in pending):
            self.store._upload_stream.synchronize()
        for state in pending:
            if state.execution_buffer_prefetch is not None:
                self.store.execution_buffer.cancel(state.execution_buffer_prefetch)
                state.execution_buffer_prefetch = None
            else:
                for position in range(state.prefetch_cancelled_bindings, len(state.tensors)):
                    self.store.unpin(state.tensors[position])
                    state.prefetch_cancelled_bindings = position + 1
            state.prefetch_cancelled_bindings = 0
            state.prefetched = False
            self.store._runtime_stats["gpu_prefetch_cancelled_units"] += 1

    def _prefetch_next_unit(self, state: _ModuleBinding, *, materialize: bool = False) -> None:
        """Queue the following block's immutable transfer without waiting for it."""

        if self._execution_queue is not None and state not in self._execution_queue.states:
            return
        target = self._execution_queue.consume(state) if self._execution_queue is not None else state.prefetch_next
        if self._execution_queue is not None and not self._execution_queue.valid:
            self.store._runtime_stats["execution_plan_mismatches"] += 1
        if target is None or target.prefetched:
            return
        if getattr(self.store, "execution_buffer", None) is not None and any(
            binding.offset + binding.nbytes > self.store.native.info.fault_ceiling
            for binding in target.tensors
        ):
            if materialize and isinstance(self.store.execution_buffer, NativeExecutionBuffers):
                # Native acquire rotates to another leased slot. The current
                # compute slot has its own last-use fence, never overwritten.
                target.execution_buffer_prefetch = self.store.execution_buffer.upload(target.tensors)
                target.prefetched = True
                self.store._runtime_stats["gpu_prefetch_units"] += 1
                self.store._runtime_stats["prefetch_attempts"] += 1
            else:
                self.store._runtime_stats["prefetch_skips"] += 1
            return
        self.store._runtime_stats["prefetch_attempts"] += 1
        if materialize:
            # Called after the current forward has enqueued its computation.
            # The native arena alone decides whether the next unit fits while
            # the current unit remains protected by its completion event.
            self._collect_temporary_lifetimes()
            try:
                self.store.fault_and_copy_group_if_needed(target.tensors)
            except (NativeVmmError, ModelArenaError) as error:
                if "resident capacity" not in str(error):
                    raise
                # Fault-group rollback has released partial pins. Speculation
                # must not wait for compute or create a temporary CUDA tensor.
                self.store._runtime_stats["gpu_prefetch_capacity_skips"] += 1
                self.store._runtime_stats["prefetch_skips"] += 1
                return
            target.prefetched = True
            self.store._runtime_stats["gpu_prefetch_units"] += 1
            return
        if self.store._native_shard_transport is not None:
            batch: list[_TensorBinding] = []
            batch_bytes = 0
            for candidate in target.tensors:
                if candidate.file_slice is None or candidate.native_prefetch_ticket is not None:
                    continue
                if candidate.signature is not None and self.store.native.peek_range_signature(candidate.offset, candidate.nbytes) == candidate.signature:
                    # Observe C++ state only; the consumer still faults and
                    # checks generation. Eviction after observation is safe:
                    # demand transport remains available, no range is pinned.
                    self.store._runtime_stats["prefetch_resident_skips"] += 1
                    continue
                if candidate.nbytes > self.store._native_shard_transport.size_bytes:
                    continue
                if batch and batch_bytes + candidate.nbytes > self.store._native_shard_transport.size_bytes:
                    break
                batch.append(candidate)
                batch_bytes += candidate.nbytes
            if not batch:
                self.store._runtime_stats["prefetch_skips"] += 1
                return
            try:
                ticket = self.store._native_shard_transport.begin_read(batch)
            except (RuntimeError, ValueError, ModelArenaError):
                self.store._runtime_stats["prefetch_skips"] += 1
                return
            for candidate in batch:
                candidate.native_prefetch_ticket = ticket
            return
        if self.store._staging_pool is None:
            self.store._runtime_stats["prefetch_skips"] += 1
            return
        # A speculative read must consume one immediately available transport
        # slot. For larger blocks preload only their first eligible tensor;
        # faulting and DMA remain the responsibility of its consumer hook.
        if not self.store._staging_pool.next_slot_ready:
            self.store._runtime_stats["prefetch_skips"] += 1
            return
        binding = next(
            (
                candidate
                for candidate in target.tensors
                if candidate.file_slice is not None
                and candidate.nbytes <= self.store._staging_pool.size_bytes
                and candidate.prefetched_source is None
            ),
            None,
        )
        if binding is None:
            # Prefetch is opportunistic: the target's own pre-hook retains the
            # normal fault/fallback contract if this look-ahead cannot admit.
            self.store._runtime_stats["prefetch_skips"] += 1
            return
        self.store._staging_pool.begin_upload()
        try:
            sources, reads = self.store._staging_pool.stage_file_slices_async(
                [(binding.source, binding.file_slice)]
            )
        except (OSError, RuntimeError, ValueError):
            self.store._runtime_stats["prefetch_skips"] += 1
            return
        binding.prefetched_source = sources[0]
        binding.prefetch_reads = reads
        target.prefetched = False

    def _register_execution_unit(self, state: _ModuleBinding) -> None:
        """Attach one hook pair for an execution unit.

        Transformer blocks are execution units, rather than their individual
        ``Linear`` children.  That lets the immutable shard source coalesce
        neighboring ranges for the block while preserving the native arena's
        pin lifetime through the enclosing forward call.  Non-block modules
        remain individual units.
        """

        def before(_module: Any, _args: tuple[Any, ...]) -> None:
            import torch

            self._collect_temporary_lifetimes()
            state.active = False
            if self.store._native_shard_transport is not None and not state.prefetched:
                # A changed execution order must not leave a skipped unit
                # pinned while the actual consumer tries to obtain capacity.
                self._cancel_gpu_prefetches()
            state.temporary_views = None
            state.execution_buffer_active = False
            pinned: list[_TensorBinding] = []
            if state.execution_buffer_prefetch is not None or (getattr(self.store, "execution_buffer", None) is not None and any(
                binding.offset + binding.nbytes > self.store.native.info.fault_ceiling
                for binding in state.tensors
            )):
                if state.prefetched and state.execution_buffer_prefetch is None:
                    # Pressure may have reduced admission after look-ahead.
                    # Retire its queued upload before releasing destination pins.
                    self.store._upload_stream.synchronize()
                    for binding in state.tensors:
                        self.store.unpin(binding)
                    state.prefetched = False
                self.store._discard_unused_native_prefetches(state.tensors)
                lease = state.execution_buffer_prefetch
                if lease is None:
                    lease = self.store.execution_buffer.upload(state.tensors)
                else:
                    self.store._runtime_stats["gpu_prefetch_hits"] += 1
                    self.store._runtime_stats["prefetch_hits"] += 1
                state.execution_buffer_prefetch = None
                state.prefetched = False
                # Publish ownership before dependency/parameter operations so
                # an always-call post hook can fence cleanup if one fails.
                state.execution_buffer_lease = lease
                state.temporary_views = lease.views
                state.execution_buffer_active = True
                state.active = True
                self.store.wait_for_compute(lease.ready)
                for binding, view in lease.views:
                    binding.owner._parameters[binding.parameter_name].data = view
                return
            try:
                if state.prefetched:
                    ready_events = [binding.ready_event for binding in state.tensors]
                    state.prefetched = False
                    self.store._runtime_stats["prefetch_hits"] += 1
                    self.store._runtime_stats["gpu_prefetch_hits"] += 1
                else:
                    ready_events = self._fault_execution_unit_with_capacity_retry(state.tensors)
                pinned = list(state.tensors)
                for binding, ready_event in zip(state.tensors, ready_events, strict=True):
                    self.store.wait_for_compute(ready_event)
                    if binding.stable.device.type == "cpu":
                        view, buffer = cuda_uint8_tensor_from_address(
                            self.store.address + binding.offset,
                            binding.nbytes,
                            owner=self.store.native,
                            backing=self.store._backing,
                        )
                        self.store._buffers.append(buffer)
                        binding.stable = view.view(binding.source.dtype).view(tuple(binding.source.shape))
                        binding.owner._parameters[binding.parameter_name] = torch.nn.Parameter(
                            binding.stable,
                            requires_grad=False,
                        )
                if self.store._native_shard_transport is None:
                    self._prefetch_next_unit(state)
            except (ModelArenaError, NativeVmmError, VmmTensorViewError) as error:
                self.store.record_fault_failure(error)
                if self.store._debug_faults:
                    print(
                        "[ddo-vmm-fast] fault fallback "
                        f"module={state.module_name} reason={error!r}",
                        flush=True,
                    )
                for binding in reversed(pinned):
                    with suppress(NativeVmmError):
                        self.store.unpin(binding)
                temporary_views: list[tuple[_TensorBinding, Any]] = []
                composite_ownership = getattr(state, "composite_payload_ownership", False)
                if composite_ownership:
                    # Own partial fallback uploads before any wait/publication
                    # can fail. The component boundary then fences their cleanup.
                    state.temporary_views = temporary_views
                    state.temporary_ready_events = []
                    state.active = True
                for binding in state.tensors:
                    if self.store._debug_faults:
                        print(
                            "[ddo-vmm-fast] temporary begin "
                            f"tensor={binding.label} bytes={binding.nbytes}",
                            flush=True,
                        )
                    temporary, ready_event = self.store.temporary_cuda_source(binding)
                    if composite_ownership:
                        temporary_views.append((binding, temporary))
                        state.temporary_ready_events.append(ready_event)
                    self.store.wait_for_compute(ready_event)
                    if not composite_ownership:
                        temporary_views.append((binding, temporary))
                    binding.owner._parameters[binding.parameter_name].data = temporary
                    self.store.record_temporary(binding.nbytes)
                if self.store._debug_faults:
                    print(
                        "[ddo-vmm-fast] temporary ready "
                        f"module={state.module_name}",
                        flush=True,
                    )
                state.temporary_views = temporary_views
            state.active = True

        def after(_module: Any, _args: tuple[Any, ...], _output: Any) -> Any:
            import torch

            if not state.active:
                return _output
            state.active = False
            composite_ownership = getattr(state, "composite_payload_ownership", False)
            if composite_ownership:
                # A failed fence/restoration keeps the owning unit active;
                # close must not free its backing while completion is unknown.
                state.active = True
            if state.execution_buffer_active:
                self.store.execution_buffer.finish_compute(
                    state.execution_buffer_lease,
                    torch.cuda.current_stream(self.store._device_ordinal)
                )
                state.execution_buffer_lease = None
                for binding, _view in state.temporary_views:
                    binding.owner._parameters[binding.parameter_name].data = binding.stable
                state.temporary_views = None
                state.execution_buffer_active = False
            elif state.temporary_views is not None:
                event = torch.cuda.Event(enable_timing=False)
                event.record(torch.cuda.current_stream(self.store._device_ordinal))
                if composite_ownership:
                    event = _CompletionSet((event, *state.temporary_ready_events))
                self._temporary_lifetimes.append((event, state.temporary_views))
                for binding, _temporary in state.temporary_views:
                    binding.owner._parameters[binding.parameter_name].data = binding.stable
                state.temporary_views = None
            else:
                # Keep pages pinned until the stream has completed the
                # execution unit's work; this permits asynchronous CUDA
                # execution without allowing native eviction to unmap live
                # data.
                event = torch.cuda.Event(enable_timing=False)
                event.record(torch.cuda.current_stream(self.store._device_ordinal))
                self._pending_unpins.append((event, list(state.tensors)))
            if composite_ownership:
                state.active = False
            if self.store._native_shard_transport is not None and _output is not None:
                self._prefetch_next_unit(state, materialize=True)
            return _output

        # These markers never schedule transfers or own weight lifetimes.
        tokens = []
        unit_id = (len(getattr(self.store, "_timeline_units", {})) + 1
                   if state.guard_reentry else len(self._handles) // 2 + 1)
        if not hasattr(self.store, "_timeline_units"):
            self.store._timeline_units = {}
        self.store._timeline_units[unit_id] = state.module_name
        def observed_before(module, args):
            if state.guard_reentry:
                # Sentinel is consumed by always_call after-hooks even when a
                # pre-hook fails; never retire the enclosing active invocation.
                tokens.append((None, 0, False))
                if state.active:
                    raise ModelArenaError("reentrant component execution unit is unsupported")
                # A partially successful preamble can already own native work.
                tokens[-1] = (None, 0, True)
            before(module, args)
            timeline = getattr(self.store, "_timeline", None)
            token = 0
            if timeline is not None and state.active:
                import torch
                token = timeline.begin_unit(unit_id, torch.cuda.current_stream(self.store._device_ordinal).cuda_stream)
            if state.guard_reentry:
                tokens[-1] = (timeline, token, True)
            else:
                tokens.append((timeline, token))

        def observed_after(module, args, output):
            if tokens:
                entry = tokens.pop()
                if state.guard_reentry and not entry[2]:
                    return output
                timeline, token = entry[:2]
                if timeline is not None and token:
                    import torch
                    timeline.end_unit(token, torch.cuda.current_stream(self.store._device_ordinal).cuda_stream)
            if state.guard_reentry and _component_output_borrows_weights(output, state.tensors):
                error = ModelArenaError("component execution units must not return managed weight views")
                _run_cleanup_preserving_error(lambda: after(module, args, None),
                                             "borrowed component output cleanup", error)
                raise error
            return after(module, args, output)

        if state.guard_reentry:
            # Torch's always_call hooks catch Exception, not BaseException.
            # Scope the opt-in component lifetime around the unchanged forward,
            # including partial preamble failures and cooperative interrupts.
            module = state.module
            original_forward = module.forward
            had_local_forward = "forward" in module.__dict__
            local_forward = module.__dict__.get("forward")

            def component_forward(*args, **kwargs):
                depth = len(tokens)
                try:
                    observed_before(module, args)
                    output = original_forward(*args, **kwargs)
                    return observed_after(module, args, output)
                except BaseException as error:
                    def cleanup():
                        if len(tokens) > depth:
                            observed_after(module, args, None)
                        elif state.active:
                            after(module, args, None)
                    _run_cleanup_preserving_error(cleanup, "component execution cleanup", error)
                    raise

            class ForwardBoundaryHandle:
                removed = False

                def remove(self):
                    if self.removed:
                        return
                    if module.forward is not component_forward:
                        raise ModelArenaError("component forward changed before boundary removal")
                    if had_local_forward:
                        module.forward = local_forward
                    else:
                        del module.forward
                    self.removed = True

            module.forward = component_forward
            self._handles.append(ForwardBoundaryHandle())
        else:
            self._handles.append(state.module.register_forward_pre_hook(observed_before))
            self._handles.append(state.module.register_forward_hook(observed_after, always_call=True))

    def attach_linear_modules(self, root: Any, module_pattern: str, *, _layout=None) -> ModelArenaAttachReport:
        layout = _layout or _discover_linear_layout(root, module_pattern)
        # Some focused contracts use a minimal store double. Production stores
        # always expose the scoped setup operation; no runner policy is added.
        setup = getattr(self.store, "view_setup", None)
        scope = setup(_layout_view_spans(layout[0], layout[1], layout[3] if len(layout) > 3 else None)) if setup is not None else nullcontext()
        with scope:
            report = self._attach_linear_modules(root, module_pattern, _layout=layout)
        # Temporary view mappings must be released before source startup or
        # startup-error cleanup can close the arena.
        prepare_sources = getattr(getattr(self.store, "_native_shard_transport", None), "prepare_sources", None)
        if prepare_sources is not None:
            try:
                prepare_sources(self._execution_units)
            except BaseException:
                self.close()
                self.store.close()
                raise
        return report

    def _attach_linear_modules(self, root: Any, module_pattern: str, *, _layout) -> ModelArenaAttachReport:
        import torch

        if not isinstance(root, torch.nn.Module):
            raise TypeError("root must be a torch.nn.Module")
        layout = _layout or _discover_linear_layout(root, module_pattern)
        selected, unit_by_module, _payload = layout[:3]
        auxiliary_groups = layout[3] if len(layout) > 3 else {}
        virtual_bytes = self.store.virtual_bytes
        unit_modules = {}
        offset = 0
        attached_modules = 0
        attached_tensors = 0

        for module_name, module in selected:
            tensor_bindings: list[_TensorBinding] = []
            for parameter_name, parameter in _layout_parameter_items(module, auxiliary_groups):
                # Keep the loader-owned CPU storage intact.  The safetensor
                # slice registry is keyed by that storage address; cloning
                # here would silently disable direct shard reads.
                source = parameter.detach()
                if source.device.type != "cpu" or not source.is_contiguous():
                    source = source.to(device="cpu").contiguous()
                file_slice = (
                    self.store._file_slices.lookup(source)
                    if self.store._file_slices is not None
                    and (self.store._staging_pool is not None or self.store._native_shard_transport is not None)
                    else None
                )
                # Register loader-owned host storage once at attachment time.
                # The registry retains the registration until store.close(),
                # so faults do not repeat host registration work.
                if self.store._host_registry is not None:
                    self.store._host_registry.register(source)
                nbytes = source.numel() * source.element_size()
                offset = _align_up(offset, max(source.element_size(), 256))
                if offset + nbytes > virtual_bytes:
                    raise ModelArenaError("model arena virtual span is smaller than attached payload")
                stable, initial_signature = self.store.stable_tensor(offset=offset, source=source)
                group = auxiliary_groups.get((id(module), parameter_name))
                if group is not None:
                    self._publish_auxiliary_view(group, stable, file_slice)
                else:
                    module.register_parameter(parameter_name, torch.nn.Parameter(stable, requires_grad=False))
                if file_slice is not None and (stable.device.type != "cpu" or group is not None):
                    retained_source: Any = _FileTensorDescriptor(
                        shape=tuple(source.shape),
                        dtype=source.dtype,
                        nbytes=nbytes,
                        element_size_bytes=source.element_size(),
                    )
                    self.store._source_stats["file_backed_bindings"] += 1
                    self.store._source_stats["file_backed_bytes"] += nbytes
                else:
                    retained_source = source
                    self.store._source_stats["cpu_source_bindings"] += 1
                    self.store._source_stats["cpu_source_bytes"] += nbytes
                tensor_bindings.append(
                    _TensorBinding(
                        owner=module,
                        parameter_name=parameter_name,
                        source=retained_source,
                        file_slice=file_slice if retained_source is not source else None,
                        stable=stable,
                        offset=offset,
                        nbytes=nbytes,
                        signature=initial_signature,
                        label=f"{module_name}.{parameter_name}",
                    )
                )
                self._address_bindings[stable.data_ptr()] = tensor_bindings[-1]
                offset += nbytes
                attached_tensors += 1

            if not tensor_bindings:
                continue
            attached_modules += 1
            module_binding = _ModuleBinding(
                module_name=module_name,
                module=module,
                tensors=tensor_bindings,
                stable_data={binding.parameter_name: binding.stable for binding in tensor_bindings},
            )
            self._bindings.append(module_binding)
            if module_name in unit_by_module:
                block_name, block_module = unit_by_module[module_name]
                unit_modules[block_name] = block_module
                self._block_bindings.setdefault(block_name, []).extend(tensor_bindings)
        # Discovery preserves ModuleList enumeration (2 before 10), without
        # imposing a model-specific naming convention or lexical ordering.
        for block_name, block_module in unit_modules.items():
            self._execution_units.append(
                _ModuleBinding(
                    module_name=block_name,
                    module=block_module,
                    tensors=self._block_bindings[block_name],
                    stable_data={},
                )
            )
        for module_binding in self._bindings:
            if module_binding.module_name in unit_by_module:
                continue
            self._execution_units.append(module_binding)
        block_units = self._execution_units[: len(unit_modules)]
        for current, following in pairwise(block_units):
            current.prefetch_next = following
        # Learn the call order only when some unit is a sub-block. When every
        # unit is a whole list item, enumeration order is the block order, so
        # block_child keeps block behavior (look-ahead from the first pass).
        learn = _execution_unit_level == "block_child" and _has_sub_block_units(root, unit_modules)
        self._execution_queue = _ObservedExecutionQueue(block_units, learn=learn)

        if self.store._streaming_enabled and self.store._native_shard_transport is not None:
            # Geometry, not calibrated residency: two native execution slots
            # fit the largest unit, including per-tensor alignment, inside the
            # existing aggregate ceiling. Native code owns admission/rotation.
            maximum = 0
            for unit in self._execution_units:
                size = 0
                for binding in unit.tensors:
                    size = ((size + 511) & ~511) + binding.nbytes
                maximum = max(maximum, size)
            if maximum:
                buffer = NativeExecutionBuffers(self.store, maximum)
                self.store.execution_buffer = buffer

        def begin_execution_pass(_module: Any, _args: tuple[Any, ...]) -> None:
            assert self._execution_queue is not None
            self._cancel_gpu_prefetches()
            self._execution_queue.begin_pass()
            self.store._runtime_stats["execution_plan_passes"] += 1
            self.store.capture_windows_memory_sample(
                f"denoise_pass_{self.store._runtime_stats['execution_plan_passes']}_before"
            )

        def end_execution_pass(_module: Any, _args: tuple[Any, ...], _output: Any) -> None:
            self._cancel_gpu_prefetches()
            self.store.capture_windows_memory_sample(
                f"denoise_pass_{self.store._runtime_stats['execution_plan_passes']}_after"
            )

        self._handles.append(root.register_forward_pre_hook(begin_execution_pass))
        self._handles.append(root.register_forward_hook(end_execution_pass, always_call=True))
        for execution_unit in self._execution_units:
            self._register_execution_unit(execution_unit)

        if attached_modules == 0:
            raise ValueError(f"no dense linear modules matched {module_pattern!r}")
        return ModelArenaAttachReport(
            pattern=module_pattern,
            attached_modules=attached_modules,
            attached_tensors=attached_tensors,
            execution_units=len(self._execution_units),
            payload_bytes=offset,
            virtual_bytes=virtual_bytes,
            page_bytes=self.store.page_bytes,
            adapter=self,
        )

    def close(self) -> None:
        self._collect_temporary_lifetimes(wait=True)
        self._collect_temporary_lifetimes(wait=True)
        self._cancel_gpu_prefetches()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._bindings.clear()
        self._execution_units.clear()
        self._address_bindings.clear()
        self._block_bindings.clear()


def attach_model_linear_modules(
    root: Any,
    library: NativeVmmLibrary,
    *,
    module_pattern: str,
    resident_budget_bytes: int | None,
    virtual_size_multiplier: float = 1.0,
    page_size_bytes: int = 0,
    device_ordinal: int = 0,
    backing: str = "device",
    staging_pool: Any | None = None,
    file_slices: Any | None = None,
    host_cache_bytes: int = 0,
    native_transport_slot_size_bytes: int = 0,
    native_transport_slots: int = 2,
    native_source_cache_slot_bytes: int = 0,
    native_source_cache_slots: int = 0,
    native_source_cache_process_limit_bytes: int = 0,
    native_source_cache_model_windows: bool = False,
    native_source_cache_registered_limit_bytes: int = 0,
    component_tensor_ownership: bool = False,
    execution_auxiliary_ownership: bool = False,
    execution_auxiliary_small_tensor_threshold_bytes: int = 16 * 1024,
    component_buffer_limit_bytes: int = 1024**2,
) -> tuple[ModelArenaAttachReport, ModelArenaWeightStore]:
    """Create one model arena and attach matching dense linear weights."""

    import torch

    if component_tensor_ownership and execution_auxiliary_ownership:
        raise ValueError("component and execution auxiliary ownership are mutually exclusive")
    if component_tensor_ownership:
        if module_pattern != ".*":
            raise ValueError("component ownership requires full-component selection (module_pattern='.*')")
        from .component_arena_adapter import attach_component_parameters

        return attach_component_parameters(
            root, library, component_buffer_limit_bytes=component_buffer_limit_bytes,
            virtual_size_multiplier=virtual_size_multiplier, resident_budget_bytes=resident_budget_bytes,
            page_size_bytes=page_size_bytes, device_ordinal=device_ordinal, backing=backing,
            staging_pool=staging_pool, file_slices=file_slices, host_cache_bytes=host_cache_bytes,
            native_transport_slot_size_bytes=native_transport_slot_size_bytes,
            native_transport_slots=native_transport_slots,
            native_source_cache_slot_bytes=native_source_cache_slot_bytes,
            native_source_cache_slots=native_source_cache_slots,
            native_source_cache_process_limit_bytes=native_source_cache_process_limit_bytes,
            native_source_cache_model_windows=native_source_cache_model_windows,
            native_source_cache_registered_limit_bytes=native_source_cache_registered_limit_bytes,
        )

    if execution_auxiliary_ownership:
        if not native_transport_slot_size_bytes:
            raise ValueError("execution auxiliary ownership requires native immutable file transport")
        from .execution_auxiliary import discover_execution_auxiliary_layout

        layout = discover_execution_auxiliary_layout(
            root, module_pattern, file_slices,
            small_tensor_threshold_bytes=execution_auxiliary_small_tensor_threshold_bytes)
    else:
        layout = _discover_linear_layout(root, module_pattern)
    payload_bytes = layout[2]
    if payload_bytes <= 0:
        raise ValueError(f"no dense linear parameters matched {module_pattern!r}")

    store = ModelArenaWeightStore(
        library,
        payload_bytes=payload_bytes,
        virtual_size_multiplier=virtual_size_multiplier,
        resident_budget_bytes=resident_budget_bytes,
        page_size_bytes=page_size_bytes,
        device_ordinal=device_ordinal,
        backing=backing,
        staging_pool=staging_pool,
        file_slices=file_slices,
        host_cache_bytes=host_cache_bytes,
        native_transport_slot_size_bytes=native_transport_slot_size_bytes,
        native_transport_slots=native_transport_slots,
        native_source_cache_slot_bytes=native_source_cache_slot_bytes,
        native_source_cache_slots=native_source_cache_slots,
        native_source_cache_process_limit_bytes=native_source_cache_process_limit_bytes,
        native_source_cache_model_windows=native_source_cache_model_windows,
        native_source_cache_registered_limit_bytes=native_source_cache_registered_limit_bytes,
    )
    if execution_auxiliary_ownership:
        from .execution_auxiliary import ExecutionAuxiliaryAdapter

        adapter = ExecutionAuxiliaryAdapter(store, tuple(layout[3].values()),
                                           execution_auxiliary_small_tensor_threshold_bytes)
        try:
            report = adapter.attach_linear_modules(root, module_pattern, _layout=layout)
        except BaseException as error:
            def cleanup():
                adapter.close()
                store.close()
            _run_cleanup_preserving_error(cleanup, "auxiliary attachment cleanup", error)
            error.auxiliary_adapter = adapter
            raise
    else:
        adapter = ModelArenaModuleAdapter(store)
        report = adapter.attach_linear_modules(root, module_pattern, _layout=layout)
    return report, store
