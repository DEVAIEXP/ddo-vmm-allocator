"""Experimental FP8 bindings on the existing native component arena route.

No new residency planner, transfers, dequantization or forward implementation.
Initial scope: complete immutable CPU sources, per-row E4M3/BF16, unique weight
owners and a single native compute stream. Requires a separately approved CUDA
gate before model use. Do not combine with execution-auxiliary ownership.
"""

from collections.abc import MutableMapping
from dataclasses import dataclass, replace

import torch

from .component_arena_adapter import ComponentArenaAdapter, discover_component_layout
from .model_arena_adapter import (ModelArenaError, ModelArenaWeightStore, NativeExecutionBuffer,
                                  _TensorBinding, _FileTensorDescriptor, _align_up)
from .quantized_owners import _capture_float8_component_owners_cpu
from .quantized_payload import _supported_backend
from .source_cache import _run_cleanup_preserving_error
from .tensor_owners import TensorOwnerGroup


class _LivePayloadTable(MutableMapping):
    """Bindings resolve directly to the unchanged backend wrapper's payloads.

    Existing slot/fallback hooks assign `.data` on these plain physical tensors.
    The table itself owns no loader or resident cache. Metadata never changes.
    """

    def __init__(self, module, key, names):
        self.module, self.key, self.names = module, key, tuple(names)

    def __getitem__(self, key):
        if key not in self.names:
            raise KeyError(key)
        return getattr(self.module._parameters[self.key], key)

    def __setitem__(self, key, value):
        old = self[key]
        if value.shape != old.shape or value.dtype != old.dtype or value.device != old.device:
            raise ValueError("Physical payload replacement must preserve geometry and device")
        setattr(self.module._parameters[self.key], key, value.detach())

    def __delitem__(self, key):
        raise RuntimeError("Managed payloads cannot be deleted")

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)


@dataclass
class _PreparedComposite:
    logical_group: object
    holder: object
    physical_groups: tuple


def discover_float8_component_layout(root, file_slices, *, buffer_limit_bytes=1024**2):
    """Expand physical geometry before CUDA operations; no model mutation."""
    cls = _supported_backend()
    if not isinstance(root, torch.nn.Module) or any(module.training for module in root.modules()):
        raise ValueError("FP8 native attachment requires an eval component")
    owner_groups = _capture_float8_component_owners_cpu(root)
    groups, composites, seen = [], [], set()
    for path, module in root.named_modules(remove_duplicate=False):
        if torch.nn.utils.parametrize.is_parametrized(module):
            raise ValueError("Parametrized native owners are unsupported")
        for key, tensor in module._parameters.items():
            if tensor is None or id(tensor) in seen:
                continue
            if type(tensor) is cls:
                logical = owner_groups[id(tensor)]
                if len(logical.owners) != 1:
                    raise ValueError("Initial native FP8 gate requires a unique logical weight owner")
                owner, owner_key, owner_names = logical.owners[0]
                owner_path = owner_names[0].rsplit(".", 1)[0] if "." in owner_names[0] else ""
                holder = torch.nn.Module()
                # Do not register the real execution module as a holder child.
                object.__setattr__(holder, "_ddo_execution_module", owner)
                physical = []
                for item in logical.weight_spec.payloads:
                    value = getattr(tensor, item.name)
                    holder.register_parameter(item.name, torch.nn.Parameter(value, requires_grad=False))
                    label = f"{owner_path}.{owner_key}_{item.name}" if owner_path else f"{owner_key}_{item.name}"
                    group = TensorOwnerGroup(holder._parameters[item.name], "parameter",
                                             ((holder, item.name, (label,)),))
                    if file_slices is None or file_slices.lookup(value) is None:
                        raise ValueError("Every FP8 payload requires an immutable source descriptor")
                    physical.append(group)
                    groups.append(group)
                composites.append(_PreparedComposite(logical, holder, tuple(physical)))
            elif type(tensor) is torch.nn.Parameter:
                groups.append(owner_groups[id(tensor)])
            else:
                raise ValueError("Unsupported native tensor subclass")
            seen.add(id(tensor))
    for path, module in root.named_modules(remove_duplicate=False):
        for key, tensor in module._buffers.items():
            if tensor is None or id(tensor) in seen:
                continue
            if type(tensor) is not torch.Tensor:
                raise ValueError("Unsupported native buffer subclass")
            groups.append(owner_groups[id(tensor)])
            seen.add(id(tensor))
    if not composites:
        raise ValueError("Explicit FP8 attachment requires at least one supported FP8 weight")
    layout = discover_component_layout(root, file_slices, buffer_limit_bytes=buffer_limit_bytes,
                                       _prepared_groups=tuple(groups),
                                       _component_lifetime_root_limit_bytes=buffer_limit_bytes)
    return layout, tuple(composites)


class Float8ComponentArenaAdapter(ComponentArenaAdapter):
    """Reuse native fault, slots, fallback, and component completion fencing."""

    def __init__(self, store, composites):
        super().__init__(store)
        self._composites = composites
        self._composites_published = False
        self._component_lifetime_buffer = None

    def _make_component_lifetime_buffer(self, size):
        return NativeExecutionBuffer(self.store, size)

    def _publish_component_lifetime_parameters(self, parameters):
        if not parameters:
            return
        size = 0
        bindings = []
        for group, descriptor in parameters:
            source = group.loader_tensor
            owner, key, names = group.owners[0]
            metadata = _FileTensorDescriptor(group.shape, group.dtype, group.nbytes, source.element_size())
            bindings.append(_TensorBinding(owner, key, metadata, descriptor, None, 0,
                                           group.nbytes, label=names[0]))
            size = _align_up(size, 512) + group.nbytes
            del source
        self._component_lifetime_buffer = self._make_component_lifetime_buffer(size)
        self.store._component_lifetime_buffer = self._component_lifetime_buffer
        views, ready = self._component_lifetime_buffer.upload(bindings)
        self.store.wait_for_compute(ready)
        for (group, _), (_, view) in zip(parameters, views, strict=True):
            group.publish(view)
            self.store._source_stats["file_backed_bindings"] += 1
            self.store._source_stats["file_backed_bytes"] += group.nbytes
        self.component_report.update(
            component_lifetime_native_capacity_bytes=self._component_lifetime_buffer.capacity_bytes,
            component_lifetime_uploads=1)

    def _fence_component_lifetime(self):
        self._component_lifetime_buffer.finish_compute(torch.cuda.current_stream(self.store._device_ordinal))

    def _finish_composite_publication(self):
        if self._composites_published:
            return
        for item in self._composites:
            logical = item.logical_group
            # Keep rollback/event ownership visible if publication fails midway.
            if logical not in self._ownership_groups:
                self._ownership_groups += (logical,)
            payloads = {spec.name: item.holder._parameters[spec.name].detach()
                        for spec in logical.weight_spec.payloads}
            logical.publish_payload_views(payloads)
            owner, key, _ = logical.owners[0]
            item.holder._parameters = _LivePayloadTable(owner, key, payloads)
            self._ownership_groups = tuple(group for group in self._ownership_groups
                                           if group not in item.physical_groups)
            # The logical wrapper and native bindings now own both physical
            # views. Do not retain a second physical-group publication owner.
            for group in item.physical_groups:
                group._published = None
                group.state = "delegated_to_composite"
        self._composites_published = True

    def _register_execution_unit(self, state):
        self._finish_composite_publication()
        state.guard_reentry = True
        state.composite_payload_ownership = True
        return super()._register_execution_unit(state)

    def close(self):
        if self._component_lifetime_buffer is not None:
            self._fence_component_lifetime()
        super().close()
        if self._component_lifetime_buffer is not None:
            self._component_lifetime_buffer.close()
            self._component_lifetime_buffer = None
            self.store._component_lifetime_buffer = None
        self._composites = ()


def _component_lifetime_store_kwargs(layout, library, store_kwargs):
    """Charge page-rounded persistent backing against the existing ceiling."""
    if not layout.component_lifetime_parameters:
        return dict(store_kwargs), 0
    size = 0
    for group, _ in layout.component_lifetime_parameters:
        size = _align_up(size, 512) + group.nbytes
    granularity = library.device_granularity(store_kwargs.get("device_ordinal", 0))
    requested_page = store_kwargs.get("page_size_bytes", 0)
    if (type(granularity) is not int or granularity <= 0 or type(requested_page) is not int
            or requested_page < 0):
        raise ValueError("component-lifetime root state requires a valid native page size")
    page = _align_up(requested_page or granularity, granularity)
    charge = _align_up(size, page)
    ceiling = store_kwargs.get("resident_budget_bytes")
    if type(ceiling) is not int or ceiling <= charge:
        raise ValueError("component-lifetime root state exceeds the existing weight ceiling")
    return {**store_kwargs, "resident_budget_bytes": ceiling - charge}, charge


def attach_float8_component_parameters(root, library, *, component_buffer_limit_bytes=1024**2, **store_kwargs):
    """Explicit experimental native API; no automatic default-route selection."""
    if not store_kwargs.get("native_transport_slot_size_bytes"):
        raise ValueError("FP8 ownership requires existing native immutable file transport")
    layout, composites = discover_float8_component_layout(
        root, store_kwargs.get("file_slices"), buffer_limit_bytes=component_buffer_limit_bytes)
    adjusted_kwargs, lifetime_charge = _component_lifetime_store_kwargs(layout, library, store_kwargs)
    store = ModelArenaWeightStore(library, payload_bytes=layout.payload_bytes, **adjusted_kwargs)
    adapter = Float8ComponentArenaAdapter(store, composites)
    try:
        report = adapter.attach_component(layout)
        if lifetime_charge:
            if adapter._component_lifetime_buffer.capacity_bytes != lifetime_charge:
                raise ModelArenaError("component-lifetime capacity differs from its reserved budget charge")
            adapter.component_report.update(component_lifetime_budget_charge_bytes=lifetime_charge,
                                            declared_total_weight_ceiling_bytes=store_kwargs["resident_budget_bytes"],
                                            arena_weight_ceiling_bytes=adjusted_kwargs["resident_budget_bytes"])
            report = replace(report, attached_tensors=report.attached_tensors + len(layout.component_lifetime_parameters))
        return report, store
    except BaseException as error:
        def cleanup():
            adapter.close()
            store.close()
        _run_cleanup_preserving_error(cleanup, "FP8 component attachment cleanup", error)
        error.quantized_adapter = adapter
        raise
