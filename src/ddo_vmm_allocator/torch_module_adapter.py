"""Small PyTorch module adapter for explicit VMM weight residency.

This adapter deliberately operates at the module boundary: a pre-forward hook
populates one stable-address parameter, and a post-forward hook records a CUDA
event before the group becomes eviction-eligible.  It does not replace
PyTorch's global allocator and does not depend on any external implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .weight_arena import VmmWeightArena, WeightArenaError


@dataclass
class _Binding:
    module: Any
    parameter_name: str
    group_name: str


class VmmModuleAdapter:
    """Attach one CPU-backed parameter to a VMM arena with safe hooks."""

    def __init__(self, arena: VmmWeightArena) -> None:
        self.arena = arena
        self._bindings: list[_Binding] = []
        self._handles: list[Any] = []

    def attach_parameter(self, module: Any, parameter_name: str, group_name: str, source: Any) -> Any:
        """Bind ``source`` and replace the module parameter with a stable view."""
        import torch

        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        parameter = getattr(module, parameter_name, None)
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(f"{parameter_name!r} is not a torch.nn.Parameter")
        if not self.arena.has_group(group_name):
            self.arena.reserve(group_name, source.numel() * source.element_size())
        self.arena.bind_cpu_tensor(group_name, source)
        stable = self.arena.persistent_tensor(
            group_name, dtype=source.dtype, shape=tuple(source.shape)
        )
        setattr(module, parameter_name, torch.nn.Parameter(stable, requires_grad=False))
        binding = _Binding(module, parameter_name, group_name)
        self._bindings.append(binding)

        def before(_module: Any, _args: tuple[Any, ...]) -> None:
            self.arena.collect_completed_events()
            try:
                loaded = self.arena.pin_loaded_tensor(group_name)
            except WeightArenaError:
                # A previous module may still be executing on the stream.  Waiting
                # is the safe bounded-budget fallback; normal runs use query().
                self.arena.collect_completed_events(wait=True)
                loaded = self.arena.pin_loaded_tensor(group_name)
            setattr(_module, parameter_name, torch.nn.Parameter(loaded, requires_grad=False))

        def after(_module: Any, _args: tuple[Any, ...], _output: Any) -> Any:
            self.arena.release_after_current_stream(group_name)
            return _output

        self._handles.append(module.register_forward_pre_hook(before))
        self._handles.append(module.register_forward_hook(after))
        return stable

    def attach_module_parameters(self, module: Any, group_prefix: str) -> list[str]:
        """Attach all direct parameters of a module as one hook-managed block.

        Each parameter has its own stable VMM view, while the pre/post hooks
        treat the complete module as one residency unit.  The caller therefore
        sizes the arena for the full block, not only for its largest matrix.
        """
        import torch

        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        bindings: list[_Binding] = []
        for parameter_name, parameter in module.named_parameters(recurse=False):
            source = parameter.detach().to(device="cpu").contiguous()
            group_name = f"{group_prefix}.{parameter_name}"
            if not self.arena.has_group(group_name):
                self.arena.reserve(group_name, source.numel() * source.element_size())
            self.arena.bind_cpu_tensor(group_name, source)
            stable = self.arena.persistent_tensor(group_name, dtype=source.dtype, shape=tuple(source.shape))
            module.register_parameter(parameter_name, torch.nn.Parameter(stable, requires_grad=False))
            bindings.append(_Binding(module, parameter_name, group_name))
        if not bindings:
            raise ValueError("module has no direct parameters")
        self._bindings.extend(bindings)

        def before(_module: Any, _args: tuple[Any, ...]) -> None:
            self.arena.collect_completed_events()
            try:
                for binding in bindings:
                    loaded = self.arena.pin_loaded_tensor(binding.group_name)
                    _module.register_parameter(
                        binding.parameter_name, torch.nn.Parameter(loaded, requires_grad=False)
                    )
            except WeightArenaError:
                self.arena.collect_completed_events(wait=True)
                for binding in bindings:
                    loaded = self.arena.pin_loaded_tensor(binding.group_name)
                    _module.register_parameter(
                        binding.parameter_name, torch.nn.Parameter(loaded, requires_grad=False)
                    )

        def after(_module: Any, _args: tuple[Any, ...], _output: Any) -> Any:
            for binding in bindings:
                self.arena.release_after_current_stream(binding.group_name)
            return _output

        self._handles.append(module.register_forward_pre_hook(before))
        self._handles.append(module.register_forward_hook(after))
        return [binding.group_name for binding in bindings]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._bindings.clear()
