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


@pytest.mark.vmm
def test_model_arena_retains_admitted_prefix_when_budget_is_full() -> None:
    """A bounded sequential scan must not evict its earlier resident pages."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import NativeVmmError, load_native_vmm

    library = load_native_vmm(library_path)
    page = library.device_granularity(0)
    arena = library.create_model_arena(
        page * 3,
        resident_budget_bytes=page,
        page_size_bytes=page,
    )
    try:
        first_signature = arena.fault_range(0, page)
        arena.unpin_range(0, page)
        assert arena.info.resident_size == page

        with pytest.raises(NativeVmmError, match="admission boundary"):
            arena.fault_range(page, page)

        assert arena.info.resident_size == page
        assert arena.info.fault_ceiling == page
        assert arena.fault_range(0, page) == first_signature
    finally:
        arena.close()


@pytest.mark.vmm
def test_model_arena_view_bootstrap_releases_before_first_fault() -> None:
    """A DLPack view may outlive its short setup mapping without residency."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm
    from ddo_vmm_allocator.tensor_view import cuda_uint8_tensor_from_address

    library = load_native_vmm(library_path)
    page = library.device_granularity(0)
    arena = library.create_model_arena(page, page_size_bytes=page)
    try:
        arena.prepare_view_range(0, 256)
        view, buffer = cuda_uint8_tensor_from_address(
            arena.info.address,
            256,
            owner=arena,
            backing="device",
        )
        arena.release_view_range(0, 256)
        assert arena.info.resident_size == 0

        source = torch.full((256,), 0x5A, dtype=torch.uint8)
        arena.fault_range(0, source.numel())
        arena.copy_from_host(0, source)
        torch.cuda.synchronize()
        assert torch.equal(view.cpu(), source)
        arena.unpin_range(0, source.numel())
        del buffer
    finally:
        arena.close()
