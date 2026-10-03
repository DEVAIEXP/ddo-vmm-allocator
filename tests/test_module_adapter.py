"""Hardware-backed regression tests for the DDO VMM module adapter."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_model_arena_adapter_retires_only_the_oldest_protected_range() -> None:
    """Capacity recovery waits for one module event, not the whole device."""

    from ddo_vmm_allocator.model_arena_adapter import ModelArenaModuleAdapter

    class Event:
        def __init__(self) -> None:
            self.completed = False
            self.synchronize_calls = 0

        def query(self) -> bool:
            return self.completed

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            self.completed = True

    class Store:
        def __init__(self) -> None:
            self.unpinned: list[object] = []
            self.capacity_waits = 0
            self._debug_faults = False

        def unpin(self, binding: object) -> None:
            self.unpinned.append(binding)

        def record_capacity_event_wait(self) -> None:
            self.capacity_waits += 1

    store = Store()
    adapter = ModelArenaModuleAdapter(store)  # type: ignore[arg-type]
    oldest = Event()
    newer = Event()
    first_binding = SimpleNamespace(label="first")
    second_binding = SimpleNamespace(label="second")
    adapter._pending_unpins = [(oldest, [first_binding]), (newer, [second_binding])]

    assert adapter._retire_oldest_pending_unpin()
    assert oldest.synchronize_calls == 1
    assert newer.synchronize_calls == 0
    assert store.unpinned == [first_binding]
    assert store.capacity_waits == 1
    assert adapter._pending_unpins == [(newer, [second_binding])]


def test_execution_unit_retries_capacity_after_retiring_one_event() -> None:
    """A recoverable admission error must not fall back before one retry."""

    from ddo_vmm_allocator.model_arena_adapter import ModelArenaError, ModelArenaModuleAdapter

    class Event:
        def query(self) -> bool:
            return False

        def synchronize(self) -> None:
            return None

    class Store:
        def __init__(self) -> None:
            self._debug_faults = False
            self.calls = 0
            self.unpinned: list[object] = []
            self.capacity_waits = 0

        def fault_and_copy_group_if_needed(self, _bindings: list[object]) -> list[None]:
            self.calls += 1
            if self.calls == 1:
                raise ModelArenaError("model arena admission boundary reached by resident capacity")
            return [None]

        def unpin(self, binding: object) -> None:
            self.unpinned.append(binding)

        def record_capacity_event_wait(self) -> None:
            self.capacity_waits += 1

    store = Store()
    adapter = ModelArenaModuleAdapter(store)  # type: ignore[arg-type]
    prior = SimpleNamespace(label="prior")
    current = SimpleNamespace(label="current")
    adapter._pending_unpins = [(Event(), [prior])]

    assert adapter._fault_execution_unit_with_capacity_retry([current]) == [None]
    assert store.calls == 2
    assert store.unpinned == [prior]
    assert store.capacity_waits == 1


def test_observed_execution_queue_ignores_auxiliary_units() -> None:
    """Only a deviation among scheduled transformer blocks invalidates look-ahead."""

    from ddo_vmm_allocator.model_arena_adapter import _ModuleBinding, _ObservedExecutionQueue

    first = _ModuleBinding("transformer_blocks.0", None, [], {})
    second = _ModuleBinding("transformer_blocks.1", None, [], {})
    auxiliary = _ModuleBinding("embedding", None, [], {})
    queue = _ObservedExecutionQueue([first, second])

    assert queue.consume(auxiliary) is None
    assert queue.valid
    assert queue.consume(first) is second
    assert queue.valid


def test_execution_order_uses_the_numeric_module_suffix() -> None:
    """Look-ahead must follow runtime block order past a two-digit index."""

    from ddo_vmm_allocator.model_arena_adapter import _execution_order_key

    names = ["blocks.10", "blocks.2", "blocks.1", "blocks.0"]
    assert sorted(names, key=_execution_order_key) == ["blocks.0", "blocks.1", "blocks.2", "blocks.10"]


def _library_path() -> Path:
    configured = os.environ.get("DDO_VMM_LIBRARY_PATH")
    if configured:
        return Path(configured)
    return Path(__file__).parents[1] / "build" / "Release" / "ddo_vmm.dll"


def test_prefetch_observes_native_signatures_without_residency_policy():
    from ddo_vmm_allocator.model_arena_adapter import ModelArenaModuleAdapter, _ModuleBinding

    resident = SimpleNamespace(file_slice=True, native_prefetch_ticket=None, signature=11, offset=0, nbytes=4)
    evicted = SimpleNamespace(file_slice=True, native_prefetch_ticket=None, signature=12, offset=4, nbytes=4)
    reads = []
    transport = SimpleNamespace(size_bytes=8, begin_read=lambda batch: reads.append(batch) or "ticket")
    store = SimpleNamespace(
        _native_shard_transport=transport,
        native=SimpleNamespace(peek_range_signature=lambda offset, size: 11 if offset == 0 else None),
        _runtime_stats=dict(prefetch_attempts=0, prefetch_skips=0, prefetch_resident_skips=0),
    )
    adapter = ModelArenaModuleAdapter(store)
    target = _ModuleBinding("blocks.1", None, [resident, evicted], {})
    state = _ModuleBinding("blocks.0", None, [], {}, prefetch_next=target)
    adapter._prefetch_next_unit(state)
    assert reads == [[evicted]]
    assert resident.native_prefetch_ticket is None
    assert evicted.native_prefetch_ticket == "ticket"
    assert store._runtime_stats["prefetch_resident_skips"] == 1


@pytest.mark.vmm
def test_two_layer_adapter_preserves_forward() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm
    from ddo_vmm_allocator.torch_module_adapter import VmmModuleAdapter
    from ddo_vmm_allocator.weight_arena import VmmWeightArena

    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64, bias=False),
        torch.nn.Linear(64, 64, bias=False),
    ).to(device="cuda", dtype=torch.bfloat16)
    arena = VmmWeightArena(load_native_vmm(library_path), resident_budget_bytes=8 * 2**20)
    adapter = VmmModuleAdapter(arena)
    adapter.attach_module_parameters(model[0], "layer0")
    adapter.attach_module_parameters(model[1], "layer1")
    for _ in range(2):
        result = model(torch.randn(1, 64, device="cuda", dtype=torch.bfloat16))
        torch.cuda.synchronize()
        assert result.shape == (1, 64)
        assert torch.isfinite(result).all()
        arena.collect_completed_events(wait=True)
    adapter.close()
    arena.close()


@pytest.mark.vmm
def test_guarded_materialization_uses_stream_safe_temporary() -> None:
    """A denied VMM commit must use ordinary CUDA storage without unmapping live views."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm
    from ddo_vmm_allocator.torch_module_adapter import VmmModuleAdapter
    from ddo_vmm_allocator.weight_arena import VmmWeightArena

    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64, bias=False),
        torch.nn.Linear(64, 64, bias=False),
    ).to(device="cuda", dtype=torch.bfloat16)
    native = load_native_vmm(library_path)
    arena = VmmWeightArena(native, resident_budget_bytes=8 * 2**20)
    adapter = VmmModuleAdapter(arena)
    adapter.attach_module_tree(model, r"^(0|1)$")
    try:
        value = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
        first = model(value)
        torch.cuda.synchronize()
        assert torch.isfinite(first).all()
        arena.collect_completed_events(wait=True)
        arena.evict_unpinned()
        native.set_dxgi_budget_guard(device_ordinal=0, local_reserve_bytes=64 * 2**30)
        arena.reset_runtime_stats()
        second = model(value)
        torch.cuda.synchronize()
        assert torch.isfinite(second).all()
        assert arena.runtime_stats()["temporary_groups"] >= 1
    finally:
        native.set_dxgi_budget_guard(device_ordinal=0)
        adapter.close()
        arena.close()


@pytest.mark.vmm
def test_host_backing_exposes_cuda_tensor_view() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm
    from ddo_vmm_allocator.tensor_view import PersistentVmmTensorView

    library = load_native_vmm(library_path)
    region = library.create_region(2 * 2**20, backing="host")
    try:
        view = PersistentVmmTensorView.create(region)
        assert view.tensor.device.type == "cuda"
        assert view.tensor.data_ptr() == region.info.address
        view.tensor[:16] = 7
        torch.cuda.synchronize()
        assert torch.equal(view.tensor[:16].cpu(), torch.full((16,), 7, dtype=torch.uint8))
    finally:
        region.close()


@pytest.mark.vmm
def test_model_arena_file_backed_binding_stages_without_retaining_weight_tensor(tmp_path: Path) -> None:
    """A shard-backed binding must reload through bounded staging, not its CPU weight."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.model_arena_adapter import attach_model_linear_modules
    from ddo_vmm_allocator.native import load_native_vmm

    from diffusers_dynamic_offloader.vmm_weight_store import (
        PinnedStagingPool,
        SafetensorFileSlice,
        SafetensorSliceRegistry,
    )

    model = torch.nn.Linear(8, 8).eval()
    weight = model.weight.detach()
    bias = model.bias.detach()
    payload = weight.view(torch.uint8).numpy().tobytes() + bias.view(torch.uint8).numpy().tobytes()
    shard = tmp_path / "weights.bin"
    shard.write_bytes(payload)
    slices = SafetensorSliceRegistry()
    assert slices.bind(weight, SafetensorFileSlice(shard, 0, weight.numel() * weight.element_size()))
    assert slices.bind(
        bias,
        SafetensorFileSlice(shard, weight.numel() * weight.element_size(), bias.numel() * bias.element_size()),
    )
    staging = PinnedStagingPool(2 * 2**20)
    store = None
    adapter = None
    try:
        _report, store = attach_model_linear_modules(
            model,
            load_native_vmm(library_path),
            module_pattern=r"^$",
            resident_budget_bytes=None,
            staging_pool=staging,
            file_slices=slices,
        )
        adapter = _report.adapter
        result = model(torch.randn(2, 8, device="cuda"))
        torch.cuda.synchronize()
        assert torch.isfinite(result).all()
        stats = store.runtime_stats()
        assert stats["file_backed_bindings"] == 2
        assert stats["cpu_source_bindings"] == 0
        assert stats["file_uploads"] == 2
    finally:
        if adapter is not None:
            adapter.close()
        if store is not None:
            store.close()
        staging.close()


@pytest.mark.vmm
def test_model_arena_file_backed_binding_uses_native_shard_transport(tmp_path: Path) -> None:
    """The optional native source reads a shard without PyTorch staging tensors."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.model_arena_adapter import attach_model_linear_modules
    from ddo_vmm_allocator.native import load_native_vmm
    from diffusers_dynamic_offloader.vmm_weight_store import SafetensorFileSlice, SafetensorSliceRegistry

    model = torch.nn.Linear(8, 8).eval()
    weight = model.weight.detach()
    bias = model.bias.detach()
    shard = tmp_path / "weights.bin"
    shard.write_bytes(weight.view(torch.uint8).numpy().tobytes() + bias.view(torch.uint8).numpy().tobytes())
    slices = SafetensorSliceRegistry()
    assert slices.bind(weight, SafetensorFileSlice(shard, 0, weight.numel() * weight.element_size()))
    assert slices.bind(
        bias,
        SafetensorFileSlice(shard, weight.numel() * weight.element_size(), bias.numel() * bias.element_size()),
    )
    store = None
    adapter = None
    try:
        report, store = attach_model_linear_modules(
            model,
            load_native_vmm(library_path),
            module_pattern=r"^$",
            resident_budget_bytes=None,
            file_slices=slices,
            native_transport_slot_size_bytes=2 * 2**20,
        )
        adapter = report.adapter
        result = model(torch.randn(2, 8, device="cuda"))
        torch.cuda.synchronize()
        assert torch.isfinite(result).all()
        stats = store.runtime_stats()
        assert stats["file_backed_bindings"] == 2
        assert stats["cpu_source_bindings"] == 0
        assert stats["native_transport_read_bytes"] == weight.numel() * weight.element_size() + bias.numel() * bias.element_size()
    finally:
        if adapter is not None:
            adapter.close()
        if store is not None:
            store.close()


@pytest.mark.vmm
def test_native_shard_transport_segments_one_large_file_binding(tmp_path: Path) -> None:
    """A file range larger than one window reaches the arena through native windows."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.model_arena_adapter import attach_model_linear_modules
    from ddo_vmm_allocator.native import load_native_vmm
    from diffusers_dynamic_offloader.vmm_weight_store import SafetensorFileSlice, SafetensorSliceRegistry

    model = torch.nn.Linear(4096, 4096).eval()
    weight = model.weight.detach()
    bias = model.bias.detach()
    shard = tmp_path / "large_weights.bin"
    weight_bytes = weight.view(torch.uint8).numpy().tobytes()
    bias_bytes = bias.view(torch.uint8).numpy().tobytes()
    shard.write_bytes(weight_bytes + bias_bytes)
    slices = SafetensorSliceRegistry()
    assert slices.bind(weight, SafetensorFileSlice(shard, 0, len(weight_bytes)))
    assert slices.bind(bias, SafetensorFileSlice(shard, len(weight_bytes), len(bias_bytes)))
    store = None
    adapter = None
    try:
        report, store = attach_model_linear_modules(
            model,
            load_native_vmm(library_path),
            module_pattern=r"^$",
            resident_budget_bytes=128 * 2**20,
            file_slices=slices,
            native_transport_slot_size_bytes=128 * 2**20,
            native_transport_slots=2,
        )
        adapter = report.adapter
        transport = store._native_shard_transport
        assert transport is not None
        assert transport.window_size_bytes == 64 * 2**20
        assert transport.ring.slot_count == 4
        result = model(torch.randn(1, 4096, device="cuda"))
        torch.cuda.synchronize()
        assert torch.isfinite(result).all()
        assert store.runtime_stats()["native_transport_read_bytes"] == len(weight_bytes) + len(bias_bytes)
    finally:
        if adapter is not None:
            adapter.close()
        if store is not None:
            store.close()


@pytest.mark.vmm
def test_native_source_cache_adapter_reuses_weight_and_bypasses_bias(tmp_path: Path) -> None:
    """The model adapter shares native cache bytes while uncached ranges use the ring."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.model_arena_adapter import attach_model_linear_modules
    from ddo_vmm_allocator.native import load_native_vmm
    from diffusers_dynamic_offloader.vmm_weight_store import SafetensorFileSlice, SafetensorSliceRegistry

    model = torch.nn.Linear(8, 8).eval()
    weight, bias = model.weight.detach(), model.bias.detach()
    sample = torch.randn(2, 8, device="cuda")
    expected = model(sample.cpu()).to("cuda")
    weight_bytes = weight.numel() * weight.element_size()
    bias_bytes = bias.numel() * bias.element_size()
    shard = tmp_path / "weights.bin"
    shard.write_bytes(weight.view(torch.uint8).numpy().tobytes() + bias.view(torch.uint8).numpy().tobytes())
    slices = SafetensorSliceRegistry()
    assert slices.bind(weight, SafetensorFileSlice(shard, 0, weight_bytes))
    assert slices.bind(bias, SafetensorFileSlice(shard, weight_bytes, bias_bytes))
    store = adapter = None
    try:
        report, store = attach_model_linear_modules(
            model, load_native_vmm(library_path), module_pattern=r"^$",
            resident_budget_bytes=2 * 2**20, file_slices=slices,
            native_transport_slot_size_bytes=2 * 2**20, native_transport_slots=1,
            native_source_cache_slot_bytes=2 * 2**20, native_source_cache_slots=1,
            native_source_cache_process_limit_bytes=4 * 2**20,
        )
        adapter = report.adapter
        first = model(sample)
        torch.cuda.synchronize()
        torch.testing.assert_close(first, expected)
        adapter._collect_temporary_lifetimes(wait=True)
        before = store.runtime_stats()
        assert before["source_cache_registered_bytes"] == 2 * 2**20
        assert before["source_cache_source_read_bytes"] == weight_bytes
        assert before["native_transport_read_bytes"] == bias_bytes
        for state in adapter._bindings:
            for binding in state.tensors:
                binding.signature = None  # Force a second upload without changing model data.
        second = model(sample)
        torch.cuda.synchronize()
        torch.testing.assert_close(second, expected)
        after = store.runtime_stats()
        assert after["source_cache_hits"] >= 1
        assert after["source_cache_source_read_bytes"] == weight_bytes
        assert after["native_transport_read_bytes"] == 2 * bias_bytes
        assert after["source_cache_dma_bytes"] == 2 * weight_bytes
    finally:
        if adapter is not None:
            adapter.close()
        if store is not None:
            store.close()
