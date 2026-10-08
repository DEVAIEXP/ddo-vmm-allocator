"""Opt-in auxiliary ownership within existing dense execution units.

Discovery is CPU-only. Bytes join the existing arena, immutable source batches
and event-fenced execution slots; no separate residency cache is introduced.
"""

from .tensor_owners import capture_model_tensor_owners
from .component_arena_adapter import ComponentArenaAdapter
from .model_arena_adapter import ModelArenaError


class ExecutionAuxiliaryAdapter(ComponentArenaAdapter):
    """Dense scheduling with existing alias retirement and native event fences."""

    def __init__(self, store, groups, small_tensor_threshold_bytes=16 * 1024):
        super().__init__(store)
        self._ownership_groups = groups
        self.auxiliary_report = {
            "schema_version": 1, "parameter_groups": len(groups),
            "parameter_bytes": sum(group.nbytes for group in groups),
            "budget_policy": "existing_native_ceiling", "residency_cache": False,
            "preserved_small_tensor_threshold_bytes": small_tensor_threshold_bytes,
        }

    def _register_execution_unit(self, state):
        # Reuse the proven component boundary for cooperative BaseException
        # cleanup and output-view/reentry guards, without changing scheduling.
        state.guard_reentry = True
        return super()._register_execution_unit(state)

    def _publish_auxiliary_view(self, group, stable, file_slice):
        if stable.device.type != "cuda" or file_slice is None:
            raise ModelArenaError("auxiliary ownership requires a file-backed CUDA native view")
        group.publish(stable)


def discover_execution_auxiliary_layout(root, pattern, file_slices, *, small_tensor_threshold_bytes=16 * 1024):
    """Include residual direct parameters in selected units, once per alias.

Aliases crossing unit boundaries, selected Linear owners or distinct overlapping
storage objects are refused before mutation. Parameters within the existing
small-tensor contract, buffers and out-of-unit parameters retain their route.
This deliberately does not infer arbitrary graphs.
"""
    import torch
    from .model_arena_adapter import _discover_linear_layout, _align_up

    if type(small_tensor_threshold_bytes) is not int or small_tensor_threshold_bytes < 0:
        raise ValueError("small tensor threshold must be a non-negative integer byte count")

    selected, units, payload = _discover_linear_layout(root, pattern)
    unit_modules = {name: module for name, module in units.values()}
    groups = capture_model_tensor_owners(root)
    auxiliary = {}
    added = []
    seen_modules = {id(module) for _, module in selected}
    for group in groups:
        if group.kind != "parameter" or group.nbytes <= small_tensor_threshold_bytes:
            continue
        locations = []
        for owner, key, names in group.owners:
            for name in names:
                path = name.rsplit(".", 1)[0] if "." in name else ""
                unit = next((name for name in unit_modules
                             if path == name or path.startswith(name + ".")), None)
                locations.append((owner, key, path, unit))
        candidates = [item for item in locations
                      if item[3] is not None and not isinstance(item[0], torch.nn.Linear)]
        if not candidates:
            continue
        if any(item[3] != candidates[0][3] or isinstance(item[0], torch.nn.Linear)
               for item in locations):
            raise ValueError("auxiliary aliases must stay within one non-Linear execution unit")
        source = group.loader_tensor
        if any(owner.training for owner, _, _, _ in locations):
            raise ValueError("auxiliary ownership requires eval mode")
        if source.device.type != "cpu" or not source.numel() or not source.is_contiguous():
            raise ValueError("auxiliary sources must be nonempty contiguous CPU parameters")
        if file_slices is None or file_slices.lookup(source) is None:
            raise ValueError("auxiliary parameter requires an immutable file source")
        start, end = source.data_ptr(), source.data_ptr() + group.nbytes
        for other in groups:
            if other is group:
                continue
            value = other.loader_tensor
            if value.device.type == "cpu" and value.numel():
                storage = value.untyped_storage()
                low = storage.data_ptr()
                high = low + storage.nbytes()
                if start < high and low < end:
                    raise ValueError("distinct auxiliary storage aliases are unsupported")
        owner, key, path, unit = candidates[0]
        auxiliary[(id(owner), key)] = group
        if id(owner) not in seen_modules:
            added.append((path, owner))
            seen_modules.add(id(owner))
            units[path] = (unit, unit_modules[unit])
    expanded = tuple(selected) + tuple(added)
    # Match attachment order and alignment exactly, including unique aliases.
    payload = 0
    from .model_arena_adapter import _layout_parameter_items
    for _, module in expanded:
        for _, parameter in _layout_parameter_items(module, auxiliary):
            payload = _align_up(payload, max(parameter.element_size(), 256))
            payload += parameter.numel() * parameter.element_size()
    return expanded, units, payload, auxiliary
