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
    result = model(torch.randn(1, 64, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    assert result.shape == (1, 64)
    assert torch.isfinite(result).all()
    arena.collect_completed_events(wait=True)
    adapter.close()
    arena.close()
