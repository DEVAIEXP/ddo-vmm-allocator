"""Model-address arena adapter for PyTorch dense linear weights.

This is the target native VMM route: one model-sized virtual address span,
range faults around actual weight use, range signatures to avoid redundant
copies, and range unpinning after the launched CUDA work.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .native import NativeModelArena, NativeVmmError, NativeVmmLibrary
from .tensor_view import VmmTensorViewError, cuda_uint8_tensor_from_address
try:
    from diffusers_dynamic_offloader.vmm_weight_store import RegisteredHostWeightRegistry
except ImportError:
    RegisteredHostWeightRegistry = None


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


@dataclass
class _ModuleBinding:
    module_name: str
    module: Any
    tensors: list[_TensorBinding]
    stable_data: dict[str, Any]
    temporary_views: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelArenaAttachReport:
    pattern: str
    attached_modules: int
    attached_tensors: int
    payload_bytes: int
    virtual_bytes: int
    page_bytes: int
    adapter: "ModelArenaModuleAdapter"


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


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
        self._device_ordinal = device_ordinal
        self._backing = backing
        self._buffers: list[Any] = []
        self._host_registry = None
        self._staging_pool = staging_pool
        self._file_slices = file_slices
        self.transfer_buffer = DeviceTransferBuffer(device_ordinal)
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
            "temporary_tensors": 0,
            "temporary_bytes": 0,
        }
        self._source_stats: dict[str, int] = {
            "file_backed_bindings": 0,
            "file_backed_bytes": 0,
            "cpu_source_bindings": 0,
            "cpu_source_bytes": 0,
        }

    @property
    def address(self) -> int:
        return self.native.info.address

    @property
    def page_bytes(self) -> int:
        return self.native.info.page_size

    @property
    def virtual_bytes(self) -> int:
        return self.native.info.virtual_size

    def reset_runtime_stats(self) -> None:
        for key in self._runtime_stats:
            self._runtime_stats[key] = 0

    def runtime_stats(self) -> dict[str, int]:
        info = self.native.info
        result = dict(self._runtime_stats)
        result.update(self._source_stats)
        result["resident_bytes"] = info.resident_size
        result["fault_ceiling_bytes"] = info.fault_ceiling
        result["virtual_bytes"] = info.virtual_size
        result["page_bytes"] = info.page_size
        return result

    def stable_tensor(self, *, offset: int, source: Any) -> tuple[Any, int]:
        try:
            self.native.prepare_view_range(offset, source.numel() * source.element_size())
        except NativeVmmError:
            # If a CUDA view cannot be represented, the module hook retains
            # the source and takes its temporary CUDA fallback.
            return source, None
        try:
            byte_tensor, buffer = cuda_uint8_tensor_from_address(
                self.address + offset,
                source.numel() * source.element_size(),
                owner=self.native,
                backing=self._backing,
            )
        finally:
            self.native.release_view_range(offset, source.numel() * source.element_size())
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

    def fault_and_copy_if_needed(self, binding: _TensorBinding) -> None:
        self._runtime_stats["faults"] += 1
        signature = self.native.fault_range(binding.offset, binding.nbytes)
        if binding.signature == signature:
            self._runtime_stats["signature_hits"] += 1
            return
        source = self._source_for_upload(binding)
        if (
            binding.file_slice is None
            and self._host_registry is not None
            and self._host_registry.register(source)
        ):
            self.native.copy_from_host(binding.offset, source)
        else:
            if source.dtype == binding.stable.dtype:
                binding.stable.copy_(source, non_blocking=True)
            else:
                converted = source.to(dtype=binding.stable.dtype, device=binding.stable.device)
                binding.stable.copy_(converted, non_blocking=True)
        binding.signature = signature
        self._runtime_stats["uploads"] += 1
        self._runtime_stats["upload_bytes"] += binding.nbytes

    def temporary_source(self, binding: _TensorBinding) -> Any:
        """Materialize a source only for the short-lived fallback view."""
        return self._source_for_upload(binding)

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

    def collect_completed_events(self, *, wait: bool = False) -> None:
        # The model-address arena currently unpins at post-forward after PyTorch
        # schedules the module work. Keep this method for the runner wrapper's
        # existing cleanup interface.
        del wait

    def close(self) -> None:
        self._buffers.clear()
        self.transfer_buffer.close()
        if self._host_registry is not None:
            self._host_registry.close()
            self._host_registry = None
        self.native.close()


class ModelArenaModuleAdapter:
    """Attach dense linear parameters to one model-address arena."""

    def __init__(self, store: ModelArenaWeightStore) -> None:
        self.store = store
        self._handles: list[Any] = []
        self._bindings: list[_ModuleBinding] = []
        self._temporary_lifetimes: list[tuple[Any, dict[str, Any]]] = []
        self._pending_unpins: list[tuple[Any, list[_TensorBinding]]] = []
        self._address_bindings: dict[int, _TensorBinding] = {}
        self._block_bindings: dict[str, list[_TensorBinding]] = {}

    def _collect_temporary_lifetimes(self, *, wait: bool = False) -> None:
        import torch

        pending: list[tuple[Any, dict[str, Any]]] = []
        for event, views in self._temporary_lifetimes:
            if wait and not event.query():
                event.synchronize()
            if not event.query():
                pending.append((event, views))
        self._temporary_lifetimes = pending
        pending_unpins: list[tuple[Any, list[_TensorBinding]]] = []
        for event, bindings in self._pending_unpins:
            if wait and not event.query():
                event.synchronize()
            if event.query():
                for binding in bindings:
                    try:
                        self.store.unpin(binding)
                    except NativeVmmError:
                        pass
            else:
                pending_unpins.append((event, bindings))
        self._pending_unpins = pending_unpins

    def _fault_block(self, bindings: list[_TensorBinding]) -> None:
        """Prepare all weights belonging to one execution block."""
        pinned = []
        try:
            for binding in bindings:
                self.store.fault_and_copy_if_needed(binding)
                pinned.append(binding)
        except Exception:
            for binding in reversed(pinned):
                self.store.unpin(binding)
            raise

    def attach_linear_modules(self, root: Any, module_pattern: str) -> ModelArenaAttachReport:
        import torch

        if not isinstance(root, torch.nn.Module):
            raise TypeError("root must be a torch.nn.Module")
        pattern = re.compile(module_pattern)
        module_names = {name for name, _module in root.named_modules() if name and pattern.search(name)}
        offset = 0
        attached_modules = 0
        attached_tensors = 0

        for module_name, module in root.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not (
                pattern.search(module_name)
                or any(module_name == name or module_name.startswith(name + ".") for name in module_names)
            ):
                continue

            tensor_bindings: list[_TensorBinding] = []
            for parameter_name in ("weight", "bias"):
                parameter = getattr(module, parameter_name, None)
                if not isinstance(parameter, torch.nn.Parameter):
                    continue
                # Keep the loader-owned CPU storage intact.  The safetensor
                # slice registry is keyed by that storage address; cloning
                # here would silently disable direct shard reads.
                source = parameter.detach()
                if source.device.type != "cpu" or not source.is_contiguous():
                    source = source.to(device="cpu").contiguous()
                file_slice = (
                    self.store._file_slices.lookup(source)
                    if self.store._file_slices is not None and self.store._staging_pool is not None
                    else None
                )
                # Register loader-owned host storage once at attachment time.
                # The registry retains the registration until store.close(),
                # so faults do not repeat host registration work.
                if self.store._host_registry is not None:
                    self.store._host_registry.register(source)
                nbytes = source.numel() * source.element_size()
                offset = _align_up(offset, max(source.element_size(), 256))
                if offset + nbytes > self.store.virtual_bytes:
                    raise ModelArenaError("model arena virtual span is smaller than attached payload")
                stable, initial_signature = self.store.stable_tensor(offset=offset, source=source)
                module.register_parameter(parameter_name, torch.nn.Parameter(stable, requires_grad=False))
                if file_slice is not None and stable.device.type != "cpu":
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
            block_match = re.search(r"(transformer_blocks\.\d+)", module_name)
            block_name = block_match.group(1) if block_match else module_name.rsplit(".", 1)[0]
            self._block_bindings.setdefault(block_name, []).extend(tensor_bindings)

            def before(_module: Any, _args: tuple[Any, ...], *, state=module_binding) -> None:
                self._collect_temporary_lifetimes()
                state.temporary_views = None
                pinned: list[_TensorBinding] = []
                try:
                    for binding in state.tensors:
                        self.store.fault_and_copy_if_needed(binding)
                        if binding.stable.device.type == "cpu":
                            view, buffer = cuda_uint8_tensor_from_address(
                                self.store.address + binding.offset,
                                binding.nbytes,
                                owner=self.store.native,
                                backing=self.store._backing,
                            )
                            self.store._buffers.append(buffer)
                            binding.stable = view.view(binding.source.dtype).view(tuple(binding.source.shape))
                            state.stable_data[binding.parameter_name] = binding.stable
                            binding.owner._parameters[binding.parameter_name] = torch.nn.Parameter(binding.stable, requires_grad=False)
                        pinned.append(binding)
                except (ModelArenaError, NativeVmmError, VmmTensorViewError) as error:
                    self.store.record_fault_failure(error)
                    for binding in reversed(pinned):
                        try:
                            self.store.unpin(binding)
                        except NativeVmmError:
                            pass
                    temporary_views: dict[str, Any] = {}
                    for binding in state.tensors:
                        temporary = self.store.temporary_source(binding).to(
                            device=f"cuda:{self.store._device_ordinal}",
                            non_blocking=True,
                        )
                        temporary_views[binding.parameter_name] = temporary
                        binding.owner._parameters[binding.parameter_name].data = temporary
                        self.store.record_temporary(binding.nbytes)
                    state.temporary_views = temporary_views

            def after(_module: Any, _args: tuple[Any, ...], _output: Any, *, state=module_binding) -> Any:
                import torch

                if state.temporary_views is not None:
                    event = torch.cuda.Event(enable_timing=False)
                    event.record(torch.cuda.current_stream(self.store._device_ordinal))
                    self._temporary_lifetimes.append((event, state.temporary_views))
                    for binding in state.tensors:
                        binding.owner._parameters[binding.parameter_name].data = state.stable_data[binding.parameter_name]
                    state.temporary_views = None
                else:
                    # Keep pages pinned until the stream has completed the
                    # module's work; this permits asynchronous CUDA execution
                    # without allowing native eviction to unmap live data.
                    event = torch.cuda.Event(enable_timing=False)
                    event.record(torch.cuda.current_stream(self.store._device_ordinal))
                    self._pending_unpins.append((event, list(state.tensors)))
                return _output

            self._handles.append(module.register_forward_pre_hook(before))
            self._handles.append(module.register_forward_hook(after))


        if attached_modules == 0:
            raise ValueError(f"no dense linear modules matched {module_pattern!r}")
        return ModelArenaAttachReport(
            pattern=module_pattern,
            attached_modules=attached_modules,
            attached_tensors=attached_tensors,
            payload_bytes=offset,
            virtual_bytes=self.store.virtual_bytes,
            page_bytes=self.store.page_bytes,
            adapter=self,
        )

    def close(self) -> None:
        self._collect_temporary_lifetimes(wait=True)
        self._collect_temporary_lifetimes(wait=True)
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._bindings.clear()
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
) -> tuple[ModelArenaAttachReport, ModelArenaWeightStore]:
    """Create one model arena and attach matching dense linear weights."""

    import torch

    pattern = re.compile(module_pattern)
    module_names = {name for name, module in root.named_modules() if name and pattern.search(name)}
    payload_bytes = 0
    for module_name, module in root.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not (
            pattern.search(module_name)
            or any(module_name == name or module_name.startswith(name + ".") for name in module_names)
        ):
            continue
        for parameter_name in ("weight", "bias"):
            parameter = getattr(module, parameter_name, None)
            if isinstance(parameter, torch.nn.Parameter):
                payload_bytes = _align_up(payload_bytes, max(parameter.element_size(), 256))
                payload_bytes += parameter.numel() * parameter.element_size()
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
    )
    adapter = ModelArenaModuleAdapter(store)
    report = adapter.attach_linear_modules(root, module_pattern)
    return report, store
