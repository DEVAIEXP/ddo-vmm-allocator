"""Explicit residency policy for stable-address VMM weight groups.

The arena has no model-framework dependency.  A caller pins a group before a
layer can read it and releases the pin only after the caller knows its CUDA
work is complete.  This keeps memory residency policy separate from weight
transport and tensor-framework integration.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
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


class VmmWeightArena:
    """A bounded resident set of model-weight groups with stable virtual VA."""

    def __init__(
        self,
        library: NativeVmmLibrary,
        *,
        resident_budget_bytes: int,
        device_ordinal: int = 0,
    ) -> None:
        if resident_budget_bytes <= 0:
            raise ValueError("resident_budget_bytes must be positive")
        self._library = library
        self._resident_budget_bytes = resident_budget_bytes
        self._device_ordinal = device_ordinal
        self._groups: dict[str, _WeightGroup] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._closed = False

    @property
    def resident_budget_bytes(self) -> int:
        return self._resident_budget_bytes

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
        region = self._library.create_region(size_bytes, self._device_ordinal)
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
        if source_tensor.numel() * source_tensor.element_size() > group.region.info.mapped_size:
            raise WeightArenaError(f"CPU tensor does not fit in VMM group {name!r}")

        self.materialize(name)
        group.tensor_view = PersistentVmmTensorView.create(group.region)
        group.source_tensor = source_tensor
        group.payload_loaded = False
        if group.pin_count == 0:
            self._evict(group)
        return self.view(name)

    def pin_loaded_tensor(self, name: str) -> Any:
        """Materialize, pin, and populate a bound CUDA tensor on the current stream."""

        group = self._group(name)
        if group.source_tensor is None or group.tensor_view is None:
            raise WeightArenaError(f"weight group has no bound CPU tensor: {name}")
        self.pin(name)
        if not group.payload_loaded:
            source = group.source_tensor
            payload_bytes = source.numel() * source.element_size()
            destination = group.tensor_view.tensor.narrow(0, 0, payload_bytes).view(source.dtype).reshape(source.shape)
            destination.copy_(source, non_blocking=False)
            group.payload_loaded = True
        return group.tensor_view.tensor.narrow(0, 0, group.source_tensor.numel() * group.source_tensor.element_size()).view(
            group.source_tensor.dtype
        ).reshape(group.source_tensor.shape)

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

    def collect_completed_events(self) -> int:
        """Unpin groups whose previously-recorded CUDA events have completed."""

        released = 0
        for group in self._groups.values():
            if not group.pending_events:
                continue
            pending = []
            for event in group.pending_events:
                if event.query():
                    self.unpin(group.name)
                    released += 1
                else:
                    pending.append(event)
            group.pending_events = pending
        return released

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
        self._make_room_for(group, protected_name=name)
        group.region.materialize()
        self._touch(name)
        return self.view(name)

    def pin(self, name: str) -> WeightGroupView:
        group = self._group(name)
        self.materialize(name)
        group.pin_count += 1
        self._touch(name)
        return self.view(name)

    def unpin(self, name: str) -> WeightGroupView:
        group = self._group(name)
        if group.pin_count == 0:
            raise WeightArenaError(f"weight group is not pinned: {name}")
        group.pin_count -= 1
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

    @staticmethod
    def _evict(group: _WeightGroup) -> None:
        group.region.evict()
        group.payload_loaded = False

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
