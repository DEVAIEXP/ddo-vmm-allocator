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
from typing import Iterator

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
            group.region.evict()
        return self.view(name)

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
            group.region.evict()
        return evicted_bytes

    def close(self) -> None:
        if self._closed:
            return
        for group in self._groups.values():
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
                candidate.region.evict()
                evicted = True
                break
            if not evicted:
                raise WeightArenaError(
                    "resident budget is exhausted by pinned VMM weight groups; no safe eviction exists"
                )

    def _touch(self, name: str) -> None:
        self._lru.move_to_end(name)

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
