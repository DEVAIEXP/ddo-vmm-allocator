"""Small PyTorch module adapter for explicit VMM weight residency.

This adapter deliberately operates at the module boundary: a pre-forward hook
populates one stable-address parameter, and a post-forward hook records a CUDA
event before the group becomes eviction-eligible.  It does not replace
PyTorch's global allocator and does not depend on any external implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
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

    def attach_module_tree(self, root: Any, module_pattern: str) -> list[str]:
        """Attach descendant parameters and hooks for matching transformer blocks.

        A block may contain nested attention/MLP linears, so ``recurse=False``
        is not sufficient.  Parameters are still stored in separate VMM groups,
        but the matching block's hook pins and releases all of them together.
        Shared parameters are registered only once.
        """
        import torch

        if not isinstance(root, torch.nn.Module):
            raise TypeError("root must be a torch.nn.Module")
        pattern = re.compile(module_pattern)
        attached_ids: set[int] = set()
        block_count = 0
        names: list[str] = []
        for block_name, block in root.named_modules():
            if not block_name or not pattern.search(block_name):
                continue
            bindings: list[_Binding] = []
            for relative_name, parameter in block.named_parameters(recurse=True):
                if id(parameter) in attached_ids:
                    continue
                owner = block
                pieces = relative_name.split(".")
                for piece in pieces[:-1]:
                    owner = getattr(owner, piece)
                parameter_name = pieces[-1]
                source = parameter.detach().to(device="cpu").contiguous()
                group_name = f"{block_name}.{relative_name}"
                if not self.arena.has_group(group_name):
                    self.arena.reserve(group_name, source.numel() * source.element_size())
                self.arena.bind_cpu_tensor(group_name, source)
                stable = self.arena.persistent_tensor(group_name, dtype=source.dtype, shape=tuple(source.shape))
                owner.register_parameter(parameter_name, torch.nn.Parameter(stable, requires_grad=False))
                binding = _Binding(owner, parameter_name, group_name)
                bindings.append(binding)
                attached_ids.add(id(parameter))
                names.append(group_name)
            if not bindings:
                continue
            block_count += 1

            def before(_module: Any, _args: tuple[Any, ...], bindings=bindings) -> None:
                self.arena.collect_completed_events()
                try:
                    for binding in bindings:
                        loaded = self.arena.pin_loaded_tensor(binding.group_name)
                        binding.module.register_parameter(
                            binding.parameter_name, torch.nn.Parameter(loaded, requires_grad=False)
                        )
                except WeightArenaError:
                    self.arena.collect_completed_events(wait=True)
                    for binding in bindings:
                        loaded = self.arena.pin_loaded_tensor(binding.group_name)
                        binding.module.register_parameter(
                            binding.parameter_name, torch.nn.Parameter(loaded, requires_grad=False)
                        )

            def after(_module: Any, _args: tuple[Any, ...], _output: Any, bindings=bindings) -> Any:
                for binding in bindings:
                    self.arena.release_after_current_stream(binding.group_name)
                return _output

            self._handles.append(block.register_forward_pre_hook(before))
            self._handles.append(block.register_forward_hook(after))
        if block_count == 0:
            raise ValueError(f"no modules matched {module_pattern!r}")
        self._bindings.extend(
            _Binding(root, "", name) for name in names
        )
        return names

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._bindings.clear()
