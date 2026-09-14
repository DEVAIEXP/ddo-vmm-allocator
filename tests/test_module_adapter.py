"""Hardware-backed regression tests for the DDO VMM module adapter."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _library_path() -> Path:
    configured = os.environ.get("DDO_VMM_LIBRARY_PATH")
    if configured:
        return Path(configured)
    return Path(__file__).parents[1] / "build" / "Release" / "ddo_vmm.dll"


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
