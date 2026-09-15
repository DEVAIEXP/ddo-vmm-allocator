"""Hardware-backed checks for native VMM range materialization."""

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
def test_region_can_materialize_granular_ranges() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm

    library = load_native_vmm(library_path)
    granularity = library.device_granularity(0)
    region = library.create_region(granularity * 2)
    try:
        assert region.resident_size == granularity * 2
        assert region.materialized

        region.evict()
        assert region.resident_size == 0
        assert not region.materialized

        region.materialize_range(0, 1)
        assert region.resident_size == granularity
        assert not region.materialized

        region.materialize_range(granularity // 2, granularity)
        assert region.resident_size == granularity * 2
        assert region.materialized

        region.evict()
        assert region.resident_size == 0

        region.materialize_range(0, 1)
        assert region.resident_size == granularity

        region.materialize_range(granularity, 1)
        assert region.resident_size == granularity * 2
        assert region.materialized
    finally:
        region.close()
