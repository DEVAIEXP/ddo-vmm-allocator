"""Explicit residency policy for stable-address VMM weight groups.

The arena has no model-framework dependency.  A caller pins a group before a
layer can read it and releases the pin only after the caller knows its CUDA
work is complete.  This keeps memory residency policy separate from weight
transport and tensor-framework integration.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from .native import NativeVmmLibrary, NativeVmmRegion


class WeightArenaError(RuntimeError):
    """Raised for invalid group lifecycle or an unsatisfiable resident budget."""


@dataclass(frozen=True)
class WeightGroupView:
    name: str
    address: int
    requested_size: int
    mapped_size: int
    materialized: bool
    pin_count: int


@dataclass
class _WeightGroup:
    name: str
    region: NativeVmmRegion
    pin_count: int = 0
    source_tensor: Any | None = None
    tensor_view: Any | None = None
    payload_loaded: bool = False
    pending_events: list[Any] | None = None
    source_pinned: bool = False
    upload_event: Any | None = None
    packed_sources: list[tuple[str, Any, int]] = field(default_factory=list)
    packed_views: dict[str, Any] = field(default_factory=dict)
    packed_payload_bytes: int = 0

class VmmWeightArena:
    """A bounded resident set of model-weight groups with stable virtual VA."""

    def __init__(
        self,
        library: NativeVmmLibrary,
        *,
        resident_budget_bytes: int | None,
        device_ordinal: int = 0,
        backing: str = "device",
        pin_source: bool = False,
        pin_source_budget_bytes: int | None = None,
    ) -> None:
        if resident_budget_bytes is not None and resident_budget_bytes <= 0:
            raise ValueError("resident_budget_bytes must be positive when specified")
        self._library = library
        self._resident_budget_bytes = resident_budget_bytes
        self._device_ordinal = device_ordinal
        self._backing = backing
        self._pin_source = pin_source
        self._pin_source_budget_bytes = pin_source_budget_bytes
        self._pinned_source_bytes = 0
        self._groups: dict[str, _WeightGroup] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._file_slices: Any | None = None
        self._staging_pool: Any | None = None
        self._managed_source: Any | None = None
        self._runtime_stats: dict[str, int] = {
            "materializations": 0,
            "lost_residency": 0,
            "uploads": 0,
            "upload_bytes": 0,
            "file_uploads": 0,
            "file_upload_bytes": 0,
            "cpu_uploads": 0,
            "cpu_upload_bytes": 0,
            "managed_uploads": 0,
            "managed_upload_bytes": 0,
            "evictions": 0,
        }
        self._closed = False

    def reset_runtime_stats(self) -> None:
        """Reset counters after model attachment and before a measured denoise."""

        for key in self._runtime_stats:
            self._runtime_stats[key] = 0

    def runtime_stats(self) -> dict[str, int]:
        """Return transfer/residency counters for the current measured interval."""

        result = dict(self._runtime_stats)
        result["resident_bytes"] = self.resident_bytes
        result["resident_groups"] = sum(
            group.region.materialized for group in self._groups.values()
        )
        return result
    def configure_file_transport(self, file_slices: Any, staging_pool: Any) -> None:
        """Configure optional checkpoint-slice transport for packed group uploads.

        ``file_slices`` and ``staging_pool`` follow DDO's public-in-practice
        transport protocol: lookup(source), begin_upload(), stage_file_slices(),
        packed_bytes and mark_upload(event). Keeping the dependency duck-typed
        lets this allocator remain usable without Diffusers or DDO installed.
        """
        if self._groups:
            raise WeightArenaError("configure file transport before reserving weight groups")
        self._file_slices = file_slices
        self._staging_pool = staging_pool
    def configure_managed_source(self, managed_source: Any) -> None:
        """Configure a persistent managed-memory source for packed groups.

        The source object implements ``populate_group(name, weights)`` and
        ``group_address(name)``. It stays separate from local VMM residency:
        kernels still read local pages, while Windows can place the immutable
        source model in its managed/shared budget.
        """
        self._managed_source = managed_source

    def preload_managed_sources(self, names: list[str] | None = None) -> int:
        """Populate managed sources incrementally and release duplicated CPU data."""
        if self._managed_source is None:
            raise WeightArenaError("no managed source is configured")
        populated = 0
        for name in names or list(self._groups):
            group = self._group(name)
            if not group.packed_sources:
                continue
            weights = [(weight_name, source) for weight_name, source, _ in group.packed_sources]
            if not self._managed_source.populate_group(name, weights):
                raise WeightArenaError(f"managed source budget cannot hold group {name!r}")
            group.packed_payload_bytes = sum(source.numel() * source.element_size() for _, source in weights)
            group.packed_sources.clear()
            populated += 1
        return populated
    def packed_source_groups(self) -> list[tuple[str, list[tuple[str, Any]]]]:
        """Return the immutable CPU source layout for each packed group."""
        return [
            (name, [(weight_name, source) for weight_name, source, _ in group.packed_sources])
            for name, group in self._groups.items()
        ]
    @property
    def resident_budget_bytes(self) -> int | None:
        return self._resident_budget_bytes

    def has_group(self, name: str) -> bool:
        """Return whether a named weight group has already been reserved."""
        return name in self._groups

    @property
    def resident_bytes(self) -> int:
        return sum(
            group.region.info.mapped_size
            for group in self._groups.values()
            if group.region.materialized
        )

    def reserve(self, name: str, size_bytes: int) -> WeightGroupView:
        self._require_open()
        if not name:
            raise ValueError("weight-group name must be non-empty")
        if name in self._groups:
            raise WeightArenaError(f"weight group already exists: {name}")
        region = self._library.create_region(size_bytes, self._device_ordinal, backing=self._backing)
        group = _WeightGroup(name=name, region=region)
        self._groups[name] = group
        self._lru[name] = None
        self._make_room_for(group, protected_name=name)
        if group.region.materialized:
            self._evict(group)
        return self.view(name)

    def bind_cpu_tensor(self, name: str, source_tensor: Any) -> WeightGroupView:
        """Associate CPU weight bytes with one persistent CUDA VMM tensor view.

        Binding creates the tensor object once while backing is present, then
        evicts the backing immediately.  Later ``pin_loaded_tensor`` calls
        materialize the same address and copy the CPU source again.
        """

        group = self._group(name)
        try:
            import torch

            from .tensor_view import PersistentVmmTensorView
        except ImportError as error:
            raise WeightArenaError("binding tensor weights requires PyTorch and cuda-python") from error
        if not isinstance(source_tensor, torch.Tensor):
            raise TypeError("source_tensor must be a torch.Tensor")
        if source_tensor.device.type != "cpu":
            raise ValueError("source_tensor must remain CPU-backed for explicit VMM transport")
        source_tensor = source_tensor.detach().contiguous()
        source_bytes = source_tensor.numel() * source_tensor.element_size()
        can_pin = self._pin_source and (
            self._pin_source_budget_bytes is None
            or self._pinned_source_bytes + source_bytes <= self._pin_source_budget_bytes
        )
        source_pinned = False
        if can_pin and source_tensor.device.type == "cpu" and not source_tensor.is_pinned():
            source_tensor = source_tensor.pin_memory()
            source_pinned = True
        if source_tensor.numel() * source_tensor.element_size() > group.region.info.mapped_size:
            raise WeightArenaError(f"CPU tensor does not fit in VMM group {name!r}")

        self.materialize(name)
        group.tensor_view = PersistentVmmTensorView.create(group.region)
        group.source_tensor = source_tensor
        group.source_pinned = source_pinned or source_tensor.is_pinned()
        if group.source_pinned:
            self._pinned_source_bytes += source_bytes
        group.payload_loaded = False
        if group.pin_count == 0:
            self._evict(group)
        return self.view(name)

    def bind_cpu_tensor_group(self, name: str, sources: list[tuple[str, Any]]) -> dict[str, Any]:
        """Bind several CPU tensors as one execution group's stable CUDA views.

        A group owns one VMM mapping. Its parameter views are byte offsets into
        that mapping, so materialization and eviction occur once per execution
        block rather than once per individual parameter.
        """
        group = self._group(name)
        if group.source_tensor is not None or group.packed_sources:
            raise WeightArenaError(f"weight group already has bound sources: {name}")
        if not sources:
            raise ValueError("packed weight group requires at least one source")
        try:
            import torch

            from .tensor_view import PersistentVmmTensorView
        except ImportError as error:
            raise WeightArenaError("binding tensor weights requires PyTorch and cuda-python") from error
        normalized: list[tuple[str, Any, int]] = []
        offset = 0
        for weight_name, source in sources:
            if not weight_name:
                raise ValueError("packed weight name must be non-empty")
            if not isinstance(source, torch.Tensor) or source.device.type != "cpu":
                raise TypeError("packed VMM sources must be CPU torch tensors")
            source = source.detach().contiguous()
            size_bytes = source.numel() * source.element_size()
            if self._pin_source and not source.is_pinned():
                if self._pin_source_budget_bytes is None or self._pinned_source_bytes + size_bytes <= self._pin_source_budget_bytes:
                    source = source.pin_memory()
                    self._pinned_source_bytes += size_bytes
            normalized.append((weight_name, source, offset))
            offset += size_bytes
        if offset > group.region.info.mapped_size:
            raise WeightArenaError(f"packed CPU tensors do not fit in VMM group {name!r}")
        self.materialize(name)
        group.tensor_view = PersistentVmmTensorView.create(group.region)
        group.packed_views = {}
        for weight_name, source, offset in normalized:
            byte_count = source.numel() * source.element_size()
            group.packed_views[weight_name] = group.tensor_view.tensor.narrow(0, offset, byte_count).view(source.dtype).reshape(source.shape)
        group.packed_sources = normalized
        group.packed_payload_bytes = offset
        group.payload_loaded = False
        if group.pin_count == 0:
            self._evict(group)
        return dict(group.packed_views)

    def _upload_packed_group(self, group: _WeightGroup, stream: Any) -> None:
        """Upload one packed execution group, preferring checkpoint slices."""
        import torch

        if group.payload_loaded:
            return
        if self._managed_source is not None:
            weights = [(weight_name, source) for weight_name, source, _ in group.packed_sources]
            if self._managed_source.populate_group(group.name, weights):
                try:
                    from cuda.bindings import driver
                except ImportError as error:
                    raise WeightArenaError("managed source transfer requires cuda-python") from error
                payload_bytes = group.packed_payload_bytes
                if payload_bytes <= 0:
                    raise WeightArenaError(f"managed source has no payload geometry for group {group.name!r}")
                result = driver.cuMemcpyDtoDAsync(
                    driver.CUdeviceptr(group.tensor_view.tensor.data_ptr()),
                    driver.CUdeviceptr(self._managed_source.group_address(group.name)),
                    payload_bytes,
                    driver.CUstream(stream.cuda_stream),
                )
                if result[0] != driver.CUresult.CUDA_SUCCESS:
                    raise WeightArenaError(f"managed source DMA failed for group {group.name!r}: {result[0]}")
                import torch

                event = torch.cuda.Event(enable_timing=False)
                event.record(stream)
                group.upload_event = event
                group.payload_loaded = True
                self._runtime_stats["uploads"] += 1
                self._runtime_stats["upload_bytes"] += payload_bytes
                self._runtime_stats["managed_uploads"] += 1
                self._runtime_stats["managed_upload_bytes"] += payload_bytes
                return
        if self._file_slices is not None and self._staging_pool is not None:
            entries = [(source, self._file_slices.lookup(source)) for _, source, _ in group.packed_sources]
            if entries and all(file_slice is not None for _, file_slice in entries):
                self._staging_pool.begin_upload()
                staged_bytes = self._staging_pool.stage_file_slices(entries)
                expected_bytes = sum(source.numel() * source.element_size() for _, source, _ in group.packed_sources)
                if staged_bytes != expected_bytes:
                    raise WeightArenaError(f"checkpoint slice payload mismatch for VMM group {group.name!r}")
                with torch.cuda.stream(stream):
                    group.tensor_view.tensor.narrow(0, 0, staged_bytes).copy_(
                        self._staging_pool.packed_bytes,
                        non_blocking=True,
                    )
                    event = torch.cuda.Event(enable_timing=False)
                    event.record(stream)
                self._staging_pool.mark_upload(event)
                group.upload_event = event
                group.payload_loaded = True
                self._runtime_stats["uploads"] += 1
                self._runtime_stats["upload_bytes"] += staged_bytes
                self._runtime_stats["file_uploads"] += 1
                self._runtime_stats["file_upload_bytes"] += staged_bytes
                return
        with torch.cuda.stream(stream):
            for _weight_name, source, offset in group.packed_sources:
                byte_count = source.numel() * source.element_size()
                destination = group.tensor_view.tensor.narrow(0, offset, byte_count).view(source.dtype).reshape(source.shape)
                destination.copy_(source, non_blocking=source.is_pinned())
            event = torch.cuda.Event(enable_timing=False)
            event.record(stream)
        group.upload_event = event
        group.payload_loaded = True
        self._runtime_stats["uploads"] += 1
        self._runtime_stats["upload_bytes"] += group.packed_payload_bytes
        self._runtime_stats["cpu_uploads"] += 1
        self._runtime_stats["cpu_upload_bytes"] += group.packed_payload_bytes

    def pin_loaded_group(self, name: str) -> dict[str, Any]:
        """Materialize and populate a complete execution group on the current stream."""
        group = self._group(name)
        if not group.packed_sources or group.tensor_view is None:
            raise WeightArenaError(f"weight group has no packed CPU sources: {name}")
        self.pin(name)
        if not group.payload_loaded:
            import torch

            self._upload_packed_group(group, torch.cuda.current_stream(self._device_ordinal))
        if group.upload_event is not None:
            import torch

            torch.cuda.current_stream(self._device_ordinal).wait_event(group.upload_event)
            group.upload_event = None
        return dict(group.packed_views)

    def prefetch_loaded_group(self, name: str, stream: Any) -> None:
        """Enqueue a whole execution group's upload on a copy stream."""
        group = self._group(name)
        if not group.packed_sources or group.tensor_view is None:
            raise WeightArenaError(f"weight group has no packed CPU sources: {name}")
        if group.payload_loaded:
            # A previous forward may have released the group after its CUDA
            # event completed.  Retain it again before this forward uses the
            # stable view.  Do not increment an existing look-ahead pin: that
            # pin already belongs to this forthcoming consumer.
            if group.pin_count == 0:
                self.pin(name)
            return
        self.pin(name)
        self._upload_packed_group(group, stream)
    def wait_for_prefetch_group(self, name: str) -> dict[str, Any]:
        group = self._group(name)
        if group.upload_event is not None:
            import torch

            torch.cuda.current_stream(self._device_ordinal).wait_event(group.upload_event)
            group.upload_event = None
        if not group.packed_sources:
            raise WeightArenaError(f"weight group has no packed CPU sources: {name}")
        return dict(group.packed_views)
    def pin_loaded_tensor(self, name: str) -> Any:
        """Materialize, pin, and populate a bound CUDA tensor on the current stream."""

        group = self._group(name)
        if group.source_tensor is None or group.tensor_view is None:
            raise WeightArenaError(f"weight group has no bound CPU tensor: {name}")
        self.pin(name)
        if group.upload_event is not None:
            torch.cuda.current_stream(self._device_ordinal).wait_event(group.upload_event)
            group.upload_event = None
        if not group.payload_loaded:
            source = group.source_tensor
            payload_bytes = source.numel() * source.element_size()
            destination = group.tensor_view.tensor.narrow(0, 0, payload_bytes).view(source.dtype).reshape(source.shape)
            destination.copy_(source, non_blocking=group.source_pinned)
            group.payload_loaded = True
        return group.tensor_view.tensor.narrow(0, 0, group.source_tensor.numel() * group.source_tensor.element_size()).view(
            group.source_tensor.dtype
        ).reshape(group.source_tensor.shape)

    def prefetch_loaded_tensor(self, name: str, stream: Any) -> None:
        """Populate a group on ``stream`` so its consumer can overlap compute."""
        group = self._group(name)
        if group.source_tensor is None or group.tensor_view is None:
            raise WeightArenaError(f"weight group has no bound CPU tensor: {name}")
        import torch

        self.pin(name)
        if group.payload_loaded:
            return
        source = group.source_tensor
        payload_bytes = source.numel() * source.element_size()
        destination = group.tensor_view.tensor.narrow(0, 0, payload_bytes).view(source.dtype).reshape(source.shape)
        with torch.cuda.stream(stream):
            destination.copy_(source, non_blocking=group.source_pinned)
            event = torch.cuda.Event(enable_timing=False)
            event.record(stream)
        group.payload_loaded = True
        group.upload_event = event

    def wait_for_prefetch(self, name: str) -> Any:
        """Wait for an asynchronous prefetch and return its typed CUDA view."""
        group = self._group(name)
        if group.upload_event is not None:
            import torch

            torch.cuda.current_stream(self._device_ordinal).wait_event(group.upload_event)
            group.upload_event = None
        if group.source_tensor is None or group.tensor_view is None:
            raise WeightArenaError(f"weight group has no bound CPU tensor: {name}")
        source = group.source_tensor
        payload_bytes = source.numel() * source.element_size()
        return group.tensor_view.tensor.narrow(0, 0, payload_bytes).view(source.dtype).reshape(source.shape)

    def release_after_current_stream(self, name: str) -> None:
        """Keep a pinned group resident until the current CUDA stream completes."""

        group = self._group(name)
        if group.pin_count == 0:
            raise WeightArenaError(f"weight group is not pinned: {name}")
        try:
            import torch
        except ImportError as error:
            raise WeightArenaError("event-based release requires PyTorch") from error
        event = torch.cuda.Event(enable_timing=False)
        event.record(torch.cuda.current_stream(self._device_ordinal))
        if group.pending_events is None:
            group.pending_events = []
        group.pending_events.append(event)

    def collect_completed_events(self, *, wait: bool = False) -> int:
        """Unpin groups whose previously-recorded CUDA events have completed."""

        released = 0
        for group in self._groups.values():
            if not group.pending_events:
                continue
            pending = []
            for event in group.pending_events:
                if wait and not event.query():
                    event.synchronize()
                if event.query():
                    self.unpin(group.name)
                    released += 1
                else:
                    pending.append(event)
            group.pending_events = pending
        return released

    def persistent_tensor(self, name: str, *, dtype: Any, shape: tuple[int, ...]) -> Any:
        """Return a typed view over a bound group's stable CUDA address.

        The returned tensor object remains valid across eviction/rematerialization,
        but its bytes are only valid while the group is pinned and populated.
        """
        group = self._group(name)
        if group.tensor_view is None or group.source_tensor is None:
            raise WeightArenaError(f"weight group has no bound CPU tensor: {name}")
        import torch

        expected = 1
        for dimension in shape:
            expected *= dimension
        if expected * torch.empty((), dtype=dtype).element_size() > group.region.info.mapped_size:
            raise WeightArenaError(f"typed tensor does not fit in VMM group {name!r}")
        self.materialize(name)
        typed = group.tensor_view.tensor.narrow(0, 0, expected * torch.empty((), dtype=dtype).element_size())
        typed = typed.view(dtype).reshape(shape)
        if group.pin_count == 0:
            self._evict(group)
        return typed

    def view(self, name: str) -> WeightGroupView:
        group = self._group(name)
        info = group.region.info
        return WeightGroupView(
            name=group.name,
            address=info.address,
            requested_size=info.requested_size,
            mapped_size=info.mapped_size,
            materialized=group.region.materialized,
            pin_count=group.pin_count,
        )

    def materialize(self, name: str) -> WeightGroupView:
        group = self._group(name)
        if not group.region.materialized:
            # Native allocator pressure may have released this stable range
            # since its last Python-side use.  Its data must be uploaded again.
            group.payload_loaded = False
            group.upload_event = None
            self._runtime_stats["materializations"] += 1
            self._runtime_stats["lost_residency"] += 1
        self._make_room_for(group, protected_name=name)
        group.region.materialize()
        self._touch(name)
        return self.view(name)

    def pin(self, name: str) -> WeightGroupView:
        group = self._group(name)
        self.materialize(name)
        group.pin_count += 1
        group.region.set_pressure_pinned(True)
        self._touch(name)
        return self.view(name)

    def unpin(self, name: str) -> WeightGroupView:
        group = self._group(name)
        if group.pin_count == 0:
            raise WeightArenaError(f"weight group is not pinned: {name}")
        group.pin_count -= 1
        if group.pin_count == 0:
            group.region.set_pressure_pinned(False)
        self._touch(name)
        return self.view(name)

    @contextmanager
    def use(self, name: str) -> Iterator[WeightGroupView]:
        """Pin a group for synchronous use, then make it eviction-eligible.

        The caller must synchronize its CUDA work before this context exits.
        An asynchronous event-based release API will be added with the tensor
        integration; evicting a live group before its kernels complete is invalid.
        """

        view = self.pin(name)
        try:
            yield view
        finally:
            self.unpin(name)

    def evict_unpinned(self, *, keep_bytes: int = 0) -> int:
        """Evict least-recently-used unpinned groups until ``keep_bytes`` remains."""

        if keep_bytes < 0:
            raise ValueError("keep_bytes cannot be negative")
        evicted_bytes = 0
        for name in list(self._lru):
            if self.resident_bytes <= keep_bytes:
                break
            group = self._groups[name]
            if group.pin_count != 0 or not group.region.materialized:
                continue
            evicted_bytes += group.region.info.mapped_size
            self._evict(group)
        return evicted_bytes

    def close(self) -> None:
        if self._closed:
            return
        for group in self._groups.values():
            if group.pin_count != 0:
                raise WeightArenaError("cannot close a VMM weight arena with pinned groups")
            group.region.close()
        self._groups.clear()
        self._lru.clear()
        self._closed = True

    def _make_room_for(self, group: _WeightGroup, *, protected_name: str) -> None:
        if group.region.materialized:
            return
        if self._resident_budget_bytes is None:
            # The native bridge applies its own LRU eviction only after CUDA
            # reports pressure.  No Python-side limit is imposed here.
            return
        required_bytes = group.region.info.mapped_size
        if required_bytes > self._resident_budget_bytes:
            raise WeightArenaError(
                f"weight group {group.name!r} requires {required_bytes} bytes, exceeding resident budget"
            )
        while self.resident_bytes + required_bytes > self._resident_budget_bytes:
            evicted = False
            for name in list(self._lru):
                if name == protected_name:
                    continue
                candidate = self._groups[name]
                if candidate.pin_count != 0 or not candidate.region.materialized:
                    continue
                self._evict(candidate)
                evicted = True
                break
            if not evicted:
                raise WeightArenaError(
                    "resident budget is exhausted by pinned VMM weight groups; no safe eviction exists"
                )

    def _touch(self, name: str) -> None:
        self._lru.move_to_end(name)
        self._groups[name].region.touch_pressure()

    def _evict(self, group: _WeightGroup) -> None:
        group.region.evict()
        group.payload_loaded = False
        self._runtime_stats["evictions"] += 1

    def _group(self, name: str) -> _WeightGroup:
        self._require_open()
        try:
            return self._groups[name]
        except KeyError as error:
            raise WeightArenaError(f"unknown weight group: {name}") from error

    def _require_open(self) -> None:
        if self._closed:
            raise WeightArenaError("VMM weight arena is closed")

    def __enter__(self) -> "VmmWeightArena":
        return self

    def __exit__(self, _exception_type: object, _exception: object, _traceback: object) -> None:
        self.close()
