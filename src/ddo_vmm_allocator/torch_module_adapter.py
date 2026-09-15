"""Small PyTorch module adapter for explicit VMM weight residency.

This adapter deliberately operates at the module boundary: a pre-forward hook
populates one stable-address parameter, and a post-forward hook records a CUDA
event before the group becomes eviction-eligible.  It does not replace
PyTorch's global allocator and does not depend on any external implementation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import re
import threading
import time
from typing import Any

from .native import NativeVmmError
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
        self._temporary_lifetimes: list[tuple[Any, dict[str, Any]]] = []
        self._read_executor: ThreadPoolExecutor | None = None
        self._prefetch_futures: dict[str, Any] = {}
        self._prefetch_lock = threading.RLock()

    def _collect_temporary_lifetimes(self, *, wait: bool = False) -> None:
        import torch

        pending: list[tuple[Any, dict[str, Any]]] = []
        for event, views in self._temporary_lifetimes:
            if wait and not event.query():
                event.synchronize()
            if not event.query():
                pending.append((event, views))
        self._temporary_lifetimes = pending

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

    def attach_module_tree(
        self,
        root: Any,
        module_pattern: str,
        *,
        recursive: bool = True,
        background_prefetch: bool = False,
        prefetch_distance: int = 1,
    ) -> list[str]:
        """Attach matching blocks as packed VMM execution groups.

        Each matching block gets one stable virtual region containing all of its
        parameters. The parameter tensors are persistent typed subviews, so
        pre/post hooks manage residency only once per execution block.
        """
        import torch

        if not isinstance(root, torch.nn.Module):
            raise TypeError("root must be a torch.nn.Module")
        pattern = re.compile(module_pattern)
        if prefetch_distance <= 0:
            raise ValueError("prefetch_distance must be positive")
        attached_ids: set[int] = set()
        group_names: list[str] = []
        all_block_names: list[str] = []
        copy_stream = torch.cuda.Stream()
        if background_prefetch:
            if self._read_executor is not None:
                raise RuntimeError("background prefetch is already configured")
            self._read_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ddo-vmm-stage")
        for block_name, block in root.named_modules():
            if not block_name or not pattern.search(block_name):
                continue
            entries: list[tuple[str, Any, str, Any]] = []
            packed_sources: list[tuple[str, Any]] = []
            for relative_name, parameter in block.named_parameters(recurse=recursive):
                if id(parameter) in attached_ids:
                    continue
                owner = block
                pieces = relative_name.split(".")
                for piece in pieces[:-1]:
                    owner = getattr(owner, piece)
                parameter_name = pieces[-1]
                source = parameter.detach().to(device="cpu").contiguous()
                entries.append((relative_name, owner, parameter_name, source))
                packed_sources.append((relative_name, source))
                attached_ids.add(id(parameter))
            if not entries:
                continue
            payload_bytes = sum(source.numel() * source.element_size() for _, source in packed_sources)
            if not self.arena.has_group(block_name):
                self.arena.reserve(block_name, payload_bytes)
            stable_views = self.arena.bind_cpu_tensor_group(block_name, packed_sources)
            for relative_name, owner, parameter_name, _source in entries:
                owner.register_parameter(
                    parameter_name,
                    torch.nn.Parameter(stable_views[relative_name], requires_grad=False),
                )
            stable_data = {
                relative_name: owner._parameters[parameter_name].data
                for relative_name, owner, parameter_name, _source in entries
            }
            # Keep per-block hook state in an object captured as a default
            # argument.  Loop-created closures must never share the temporary
            # storage or parameter list of the last attached block.
            hook_state: dict[str, Any] = {"temporary_views": None}
            block_index = len(all_block_names)
            all_block_names.append(block_name)
            group_names.append(block_name)

            def stage_group(index: int, *, all_blocks=all_block_names) -> None:
                if index < len(all_blocks):
                    self.arena.prefetch_loaded_group(all_blocks[index], copy_stream)

            def submit_stage(index: int, *, all_blocks=all_block_names) -> None:
                if index >= len(all_blocks):
                    return
                name = all_blocks[index]
                with self._prefetch_lock:
                    if name in self._prefetch_futures:
                        return

                    def worker() -> None:
                        torch.cuda.set_device(self.arena._device_ordinal)
                        stage_group(index)

                    self._prefetch_futures[name] = self._read_executor.submit(worker)

            def before(
                _module: Any,
                _args: tuple[Any, ...],
                *,
                group_name=block_name,
                group_index=block_index,
                module_entries=entries,
                block_state=hook_state,
                all_blocks=all_block_names,
            ) -> None:
                self._collect_temporary_lifetimes()
                self.arena.collect_completed_events()
                block_state["temporary_views"] = None
                try:
                    if self._read_executor is None:
                        stage_group(group_index)
                    else:
                        with self._prefetch_lock:
                            future = self._prefetch_futures.pop(group_name, None)
                        if future is None:
                            stage_group(group_index)
                        else:
                            future_wait_started = time.perf_counter()
                            future.result()
                            self.arena.record_prefetch_future_wait(
                                time.perf_counter() - future_wait_started
                            )
                except (WeightArenaError, NativeVmmError):
                    # The current group is mandatory.  If a previously launched
                    # kernel still pins the available budget, wait rather than
                    # allocating outside the arena's bound.
                    self.arena.collect_completed_events(wait=True)
                    try:
                        self.arena.pin_loaded_group(group_name)
                    except (WeightArenaError, NativeVmmError):
                        # VMM local backing is full.  Keep the stable virtual
                        # view intact and execute this block from one ordinary
                        # CUDA temporary, then release it after the hook.
                        temporary_views = self.arena.temporary_loaded_group(group_name)
                        block_state["temporary_views"] = temporary_views
                        for relative_name, owner, parameter_name, _source in module_entries:
                            owner._parameters[parameter_name].data = temporary_views[relative_name]
                else:
                    next_indices = range(group_index + 1, group_index + 1 + prefetch_distance)
                    for next_index in next_indices:
                        if next_index >= len(all_blocks):
                            break
                        try:
                            if self._read_executor is None:
                                stage_group(next_index)
                            else:
                                submit_stage(next_index)
                        except (WeightArenaError, NativeVmmError):
                            # Look-ahead is optional.  A sub-page resident
                            # budget may fit one rounded block but not two.
                            continue
                    self.arena.wait_for_prefetch_group(group_name)

            def after(
                _module: Any,
                _args: tuple[Any, ...],
                _output: Any,
                *,
                group_name=block_name,
                module_entries=entries,
                block_state=hook_state,
                block_stable_data=stable_data,
            ) -> Any:
                temporary_views = block_state["temporary_views"]
                if temporary_views is not None:
                    # A temporary tensor is not protected by the VMM arena.
                    # Synchronize before restoring the stable parameter view so
                    # no queued kernel can observe its storage being recycled.
                    torch.cuda.current_stream(self.arena._device_ordinal).synchronize()
                    event = torch.cuda.Event(enable_timing=False)
                    event.record(torch.cuda.current_stream(self.arena._device_ordinal))
                    self._temporary_lifetimes.append((event, temporary_views))
                    for relative_name, owner, parameter_name, _source in module_entries:
                        owner._parameters[parameter_name].data = block_stable_data[relative_name]
                    block_state["temporary_views"] = None
                else:
                    self.arena.release_after_current_stream(group_name)
                return _output

            self._handles.append(block.register_forward_pre_hook(before))
            self._handles.append(block.register_forward_hook(after))
        if not group_names:
            raise ValueError(f"no modules matched {module_pattern!r}")
        self._bindings.extend(_Binding(root, "", name) for name in group_names)
        return group_names

    def close(self) -> None:
        self._collect_temporary_lifetimes(wait=True)
        self._temporary_lifetimes.clear()
        executor = self._read_executor
        if executor is not None:
            with self._prefetch_lock:
                futures = list(self._prefetch_futures.values())
                self._prefetch_futures.clear()
            for future in futures:
                future.result()
            executor.shutdown(wait=True)
            self._read_executor = None
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._bindings.clear()
