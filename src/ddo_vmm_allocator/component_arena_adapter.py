"""Opt-in component ownership using the existing native fault/unit machinery.

No architecture names, parameter CPU cache or Python residency policy. Initial
scope: disjoint execution units, immutable file-backed parameters, bounded
device buffers and one compute stream. Legacy Linear attachment is unchanged.
"""

from dataclasses import dataclass

from .model_arena_adapter import (
    ModelArenaError, ModelArenaAttachReport, ModelArenaModuleAdapter,
    ModelArenaWeightStore, NativeExecutionBuffers, _ModuleBinding,
    _TensorBinding, _FileTensorDescriptor, _align_up,
)
from .tensor_owners import capture_model_tensor_owners


@dataclass
class ComponentLayout:
    groups: tuple
    parameters: tuple  # group, offset, descriptor
    units: tuple  # name, module, parameter group indexes
    payload_bytes: int
    buffer_bytes: int
    component_lifetime_parameters: tuple = ()  # group, immutable descriptor

    def snapshot(self):
        result = {"schema_version": 1, "parameter_groups": len(self.parameters),
                "buffer_groups": sum(group.kind == "buffer" for group in self.groups),
                "payload_bytes": self.payload_bytes, "buffer_bytes": self.buffer_bytes,
                "execution_units": len(self.units), "lookahead": False}
        if self.component_lifetime_parameters:
            result.update(component_lifetime_parameter_groups=len(self.component_lifetime_parameters),
                          component_lifetime_parameter_bytes=sum(g.nbytes for g, _ in self.component_lifetime_parameters))
        return result

    def capacity_snapshot(self):
        """Detached loaded-source geometry; no native allocation/admission."""
        units = []
        for name, _, indices in self.units:
            payload = 0
            for index in indices:
                payload = _align_up(payload, 512) + self.parameters[index][0].nbytes
            units.append({"name": name, "parameter_groups": len(indices),
                          "execution_slot_payload_bytes": payload})
        largest = max(unit["execution_slot_payload_bytes"] for unit in units)
        return {**self.snapshot(), "observation": "loaded_source_preflight",
                "native_admission_validated": False,
                "unique_parameter_bytes": (sum(group.nbytes for group, _, _ in self.parameters)
                                           + sum(g.nbytes for g, _ in self.component_lifetime_parameters)),
                "execution_slot_count": 2, "execution_slot_payload_bytes": largest,
                "execution_slot_total_payload_bytes": 2 * largest,
                "units": units}


def discover_component_layout(root, file_slices, *, buffer_limit_bytes, _prepared_groups=None,
                              _component_lifetime_root_limit_bytes=None):
    """Preflight before any allocator operation or module mutation."""
    if type(buffer_limit_bytes) is not int or buffer_limit_bytes < 0:
        raise ValueError("component buffer limit must be a non-negative byte integer")
    if (_component_lifetime_root_limit_bytes is not None
            and (type(_component_lifetime_root_limit_bytes) is not int
                 or _component_lifetime_root_limit_bytes < 0)):
        raise ValueError("component-lifetime root limit must be a non-negative byte integer")
    groups = capture_model_tensor_owners(root) if _prepared_groups is None else _prepared_groups
    modules = dict(root.named_modules())
    list_items = {}
    import torch

    if any(module.training for module in modules.values()):
        raise ValueError("component ownership requires eval mode for immutable inference")
    for path, module in modules.items():
        if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)) and module.max_norm is not None:
            raise ValueError("in-place embedding renormalization is incompatible with immutable native sources")
        if isinstance(module, torch.nn.ModuleList):
            for index, child in enumerate(module):
                list_items[f"{path}.{index}" if path else str(index)] = child
    parameters, units, offset = [], {}, 0
    buffers = 0
    lifetime, lifetime_bytes = [], 0
    for group in groups:
        source = group.loader_tensor
        if source.device.type != "cpu" or not source.is_contiguous() or not source.numel():
            raise ValueError("component ownership requires materialized contiguous nonempty CPU tensors")
        if group.kind == "buffer":
            buffers += group.nbytes
            continue
        descriptor = file_slices.lookup(source) if file_slices is not None else None
        if descriptor is None:
            raise ValueError(f"component parameter has no immutable file source: {group.snapshot()['names']}")
        if _component_lifetime_root_limit_bytes is not None and any(owner is root for owner, _, _ in group.owners):
            if not all(owner is root for owner, _, _ in group.owners) or type(source) is not torch.nn.Parameter:
                raise ValueError("component-lifetime root state requires plain parameters with root-only owners")
            lifetime.append((group, descriptor))
            lifetime_bytes += group.nbytes
            continue
        offset = _align_up(offset, max(source.element_size(), 256))
        position = len(parameters)
        parameters.append((group, offset, descriptor))
        offset += group.nbytes
        for owner, key, names in group.owners:
            path = names[0].rsplit(".", 1)[0] if "." in names[0] else ""
            ancestors = [".".join(path.split(".")[:index]) for index in range(1, len(path.split(".")) + 1)]
            unit = next((parent for parent in ancestors if parent in list_items), path)
            execution_owner = getattr(owner, "_ddo_execution_module", owner)
            entry = units.setdefault(unit, [list_items.get(unit, execution_owner), []])
            if position not in entry[1]:
                entry[1].append(position)
    if not parameters:
        raise ValueError("component has no file-backed parameters")
    if buffers > buffer_limit_bytes:
        raise ValueError("component buffers exceed the explicit device-buffer limit")
    if (_component_lifetime_root_limit_bytes is not None
            and buffers + lifetime_bytes > _component_lifetime_root_limit_bytes):
        raise ValueError("component-lifetime root state and buffers exceed the explicit component limit")
    names = list(units)
    if "" in units and len(units) > 1:
        raise ValueError("overlapping direct-parameter execution units are not supported")
    # A parameter-owning parent consumes its parameter across child calls.
    # Enclose descendant units in that parent rather than swapping native
    # storage underneath an active invocation. This is execution geometry,
    # not an architecture-specific residency or cache policy.
    merged = {}
    for name, (unit_module, indices) in units.items():
        ancestors = [parent for parent in names if parent and name.startswith(parent + ".")]
        enclosing = min(ancestors, key=lambda path: len(path.split("."))) if ancestors else name
        entry = merged.setdefault(enclosing, [units[enclosing][0], []])
        for index in indices:
            if index not in entry[1]:
                entry[1].append(index)
    units = merged
    return ComponentLayout(groups, tuple(parameters),
                           tuple((name, module, tuple(indices)) for name, (module, indices) in units.items()),
                           offset, buffers, tuple(lifetime))


class ComponentArenaAdapter(ModelArenaModuleAdapter):
    def __init__(self, store):
        super().__init__(store)
        self._ownership_groups = ()
        self._retirement_event = None
        self.component_report = {}

    def _move_buffer(self, source):
        return source.to(f"cuda:{self.store._device_ordinal}")

    def _native_view(self, group, source, offset):
        stable, signature = self.store.stable_tensor(offset=offset, source=source)
        if stable.device.type != "cuda":
            raise ModelArenaError("component publication requires a CUDA native view, not a retained CPU source")
        return stable, signature

    def _completion_event(self):
        import torch

        event = torch.cuda.Event(enable_timing=False)
        event.record(torch.cuda.current_stream(self.store._device_ordinal))
        return event

    def _publish_component_lifetime_parameters(self, parameters):
        if parameters:
            raise ModelArenaError("component-lifetime root state requires an explicit dedicated adapter")

    def attach_component(self, layout):
        self._ownership_groups = layout.groups
        self.component_report = layout.snapshot()
        bindings = []
        spans = tuple((offset, group.nbytes) for group, offset, _ in layout.parameters)
        with self.store.view_setup(spans):
            for group, offset, descriptor in layout.parameters:
                source = group.loader_tensor
                stable, signature = self._native_view(group, source, offset)
                metadata = _FileTensorDescriptor(group.shape, group.dtype, group.nbytes, source.element_size())
                group.publish(stable)
                owner, key, names = group.owners[0]
                binding = _TensorBinding(owner, key, metadata, descriptor, stable, offset,
                                         group.nbytes, signature=signature, label=names[0])
                bindings.append(binding)
                self._address_bindings[stable.data_ptr()] = binding
                self.store._source_stats["file_backed_bindings"] += 1
                self.store._source_stats["file_backed_bytes"] += group.nbytes
                del source
        for group in layout.groups:
            if group.kind == "buffer":
                group.publish(self._move_buffer(group.loader_tensor))
        self._publish_component_lifetime_parameters(layout.component_lifetime_parameters)
        for name, module, indices in layout.units:
            state = _ModuleBinding(name, module, [bindings[index] for index in indices], {},
                                   guard_reentry=True)
            self._bindings.append(state)
            self._execution_units.append(state)
        # Geometry only: native slots/admission retain the existing GPU ceiling.
        # No speculative lookahead in the initial component ownership contract.
        if self.store._streaming_enabled and self.store._native_shard_transport is not None:
            maximum = 0
            for unit in self._execution_units:
                size = 0
                for binding in unit.tensors:
                    size = _align_up(size, 512) + binding.nbytes
                maximum = max(maximum, size)
            self.store.execution_buffer = NativeExecutionBuffers(self.store, maximum)
        for unit in self._execution_units:
            self._register_execution_unit(unit)
        self.store._native_shard_transport.prepare_sources(self._execution_units)
        return ModelArenaAttachReport("component_parameters", len(layout.units), len(bindings),
                                      len(layout.units), layout.payload_bytes,
                                      self.store.virtual_bytes, self.store.page_bytes, self)

    def close(self):
        if any(unit.active for unit in self._execution_units):
            raise ModelArenaError("cannot close a component with an active execution unit")
        super().close()  # Last upload/compute users and pins must retire first.
        published = [group for group in self._ownership_groups if group.state in ("published", "retiring")]
        if published:
            if self._retirement_event is None:
                self._retirement_event = self._completion_event()
            self._retirement_event.synchronize()
            for group in published:
                if not group.retire(self._retirement_event):
                    raise ModelArenaError("component final-consumer completion is unproven")
        for group in self._ownership_groups:
            if group.state == "publication_failed":
                group.rollback_failed_publication()
        self._ownership_groups = ()
        self._retirement_event = None


def attach_component_parameters(root, library, *, component_buffer_limit_bytes, **store_kwargs):
    if not store_kwargs.get("native_transport_slot_size_bytes"):
        raise ValueError("component ownership requires native immutable file transport")
    layout = discover_component_layout(root, store_kwargs.get("file_slices"),
                                       buffer_limit_bytes=component_buffer_limit_bytes)
    store = ModelArenaWeightStore(library, payload_bytes=layout.payload_bytes, **store_kwargs)
    adapter = ComponentArenaAdapter(store)
    try:
        return adapter.attach_component(layout), store
    except BaseException as error:
        # Never close backing if alias/event cleanup cannot prove release.
        from .source_cache import _run_cleanup_preserving_error

        def cleanup():
            adapter.close()
            store.close()
        _run_cleanup_preserving_error(cleanup, "component attachment cleanup", error)
        # Retain the owning adapter for diagnostics/retry after cleanup failure.
        error.component_adapter = adapter
        raise
