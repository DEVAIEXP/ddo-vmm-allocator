"""Hardware-backed checks for native VMM range materialization."""

from __future__ import annotations

import ctypes
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
def test_host_ring_exposes_distinct_registered_slots() -> None:
    """The native transport source owns stable, registered host buffers."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm

    library = load_native_vmm(library_path)
    ring = library.create_host_ring(2 * 1024**2, 2)
    try:
        assert ring.slot_size_bytes == 2 * 1024**2
        assert ring.slot_count == 2
        assert ring.slot_address(0) != 0
        assert ring.slot_address(1) != 0
        assert ring.slot_address(0) != ring.slot_address(1)
        with pytest.raises(IndexError):
            ring.slot_address(2)
    finally:
        ring.close()


@pytest.mark.vmm
def test_host_ring_can_enqueue_copy_to_faulted_model_arena() -> None:
    """A native ring slot reaches the stable arena address without a host sync."""
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
    ring = library.create_host_ring(2 * 1024**2, 2)
    try:
        arena.prepare_view_range(0, 256)
        view, buffer = cuda_uint8_tensor_from_address(
            arena.info.address,
            256,
            owner=arena,
            backing="device",
        )
        arena.release_view_range(0, 256)
        arena.fault_range(0, 256)
        ctypes.memset(ring.slot_address(0), 0x5A, 256)
        ring.copy_to_arena_async(0, arena, size_bytes=256)
        ring.wait_slot_async(0)
        ready = torch.cuda.Event(enable_timing=False)
        ready.record()
        ready.synchronize()
        assert ring.slot_ready(0)
        ring.wait_slot(0)
        assert torch.equal(view.cpu(), torch.full((256,), 0x5A, dtype=torch.uint8))
        arena.unpin_range(0, 256)
        del buffer
    finally:
        ring.close()
        arena.close()


@pytest.mark.vmm
def test_host_ring_reads_file_directly_to_faulted_model_arena(tmp_path: Path) -> None:
    """The native allocator owns the Windows file read and the following DMA."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm
    from ddo_vmm_allocator.tensor_view import cuda_uint8_tensor_from_address

    payload = bytes(range(256))
    shard = tmp_path / "native_transport.bin"
    shard.write_bytes(payload)
    library = load_native_vmm(library_path)
    page = 2 * 1024**2
    arena = library.create_model_arena(page, page_size_bytes=page)
    ring = library.create_host_ring(page, 1)
    try:
        arena.prepare_view_range(0, len(payload))
        view, buffer = cuda_uint8_tensor_from_address(
            arena.info.address, len(payload), owner=arena, backing="device"
        )
        arena.release_view_range(0, len(payload))
        arena.fault_range(0, len(payload))
        ring.read_file_to_arena_async(0, shard, 0, arena, arena_offset_bytes=0, size_bytes=len(payload))
        ring.wait_slot_async(0)
        torch.cuda.synchronize()
        assert ring.slot_ready(0)
        assert torch.equal(view.cpu(), torch.tensor(list(payload), dtype=torch.uint8))
        arena.unpin_range(0, len(payload))
        del buffer
    finally:
        ring.close()
        arena.close()


@pytest.mark.vmm
def test_host_ring_native_file_read_ticket_feeds_async_dma(tmp_path: Path) -> None:
    """A native source ticket must finish before its registered-slot DMA begins."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm
    from ddo_vmm_allocator.tensor_view import cuda_uint8_tensor_from_address

    payload = bytes(range(256))
    shard = tmp_path / "native_ticket_transport.bin"
    shard.write_bytes(payload)
    library = load_native_vmm(library_path)
    page = 2 * 1024**2
    arena = library.create_model_arena(page, page_size_bytes=page)
    ring = library.create_host_ring(page, 1)
    try:
        arena.prepare_view_range(0, len(payload))
        view, buffer = cuda_uint8_tensor_from_address(
            arena.info.address, len(payload), owner=arena, backing="device"
        )
        arena.release_view_range(0, len(payload))
        arena.fault_range(0, len(payload))
        ring.read_file_async(0, shard, 0, size_bytes=len(payload))
        assert not ring.slot_ready(0)
        ring.wait_file_read(0)
        ring.copy_to_arena_async(0, arena, size_bytes=len(payload))
        ring.wait_slot_async(0)
        torch.cuda.synchronize()
        assert ring.slot_ready(0)
        assert torch.equal(view.cpu(), torch.tensor(list(payload), dtype=torch.uint8))
        arena.unpin_range(0, len(payload))
        del buffer
    finally:
        ring.close()
        arena.close()


@pytest.mark.vmm
def test_host_ring_native_file_tickets_share_shard_with_independent_offsets(tmp_path: Path) -> None:
    """Concurrent native tickets may reuse one shard handle without sharing a cursor."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm

    first = bytes(range(256))
    second = bytes(reversed(range(256)))
    shard = tmp_path / "native_persistent_ticket_transport.bin"
    shard.write_bytes(first + second)
    library = load_native_vmm(library_path)
    ring = library.create_host_ring(2 * 1024**2, 2)
    try:
        ring.read_file_async(0, shard, 0, size_bytes=len(first))
        ring.read_file_async(1, shard, len(first), size_bytes=len(second))
        ring.wait_file_read(0)
        ring.wait_file_read(1)
        assert ctypes.string_at(ring.slot_address(0), len(first)) == first
        assert ctypes.string_at(ring.slot_address(1), len(second)) == second
    finally:
        ring.close()


@pytest.mark.vmm
def test_host_ring_ticket_copies_each_completed_source_slot_to_arena(tmp_path: Path) -> None:
    """A multi-slot source ticket reaches disjoint arena ranges without a host sync."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm
    from ddo_vmm_allocator.tensor_view import cuda_uint8_tensor_from_address

    first = bytes(range(256))
    second = bytes(reversed(range(256)))
    shard = tmp_path / "native_progressive_ticket.bin"
    shard.write_bytes(first + second)
    library = load_native_vmm(library_path)
    page = library.device_granularity(0)
    arena = library.create_model_arena(page, page_size_bytes=page)
    ring = library.create_host_ring(2 * 1024**2, 2)
    try:
        arena.prepare_view_range(0, len(first) + len(second))
        view, buffer = cuda_uint8_tensor_from_address(
            arena.info.address, len(first) + len(second), owner=arena, backing="device"
        )
        arena.release_view_range(0, len(first) + len(second))
        arena.fault_range(0, len(first) + len(second))
        ring.read_file_async(0, shard, 0, size_bytes=len(first))
        ring.read_file_async(1, shard, len(first), size_bytes=len(second))
        ring.copy_ticket_to_arena_async([(0, 0, len(first)), (1, len(first), len(second))], arena)
        ring.wait_slot_async(0)
        ring.wait_slot_async(1)
        torch.cuda.synchronize()
        assert torch.equal(view[: len(first)].cpu(), torch.tensor(list(first), dtype=torch.uint8))
        assert torch.equal(view[len(first) :].cpu(), torch.tensor(list(second), dtype=torch.uint8))
        arena.unpin_range(0, len(first) + len(second))
        del buffer
    finally:
        ring.close()
        arena.close()


@pytest.mark.vmm
def test_host_ring_native_file_ticket_uses_positioned_2mib_chunks(tmp_path: Path) -> None:
    """One persistent ticket splits a larger source range into native 2 MiB reads."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm

    payload = b"\xA5" * (5 * 1024**2 + 17)
    shard = tmp_path / "native_chunked_ticket_transport.bin"
    shard.write_bytes(payload)
    library = load_native_vmm(library_path)
    ring = library.create_host_ring(6 * 1024**2, 9)
    try:
        ring.read_file_async(0, shard, 0, size_bytes=len(payload))
        ring.wait_file_read(0)
        assert ctypes.string_at(ring.slot_address(0), len(payload)) == payload
        assert ring.source_stats == {
            "workers": 8,
            "read_chunks": 3,
            "read_bytes": len(payload),
            "read_failures": 0,
        }
    finally:
        ring.close()


@pytest.mark.vmm
def test_host_ring_batch_copy_uses_one_slot_completion_event() -> None:
    """Packed shard ranges can DMA to disjoint arena ranges as one ticket."""
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
    ring = library.create_host_ring(2 * 1024**2, 2)
    try:
        arena.prepare_view_range(0, 256)
        view, buffer = cuda_uint8_tensor_from_address(
            arena.info.address, 256, owner=arena, backing="device"
        )
        arena.release_view_range(0, 256)
        arena.fault_range(0, 256)
        ctypes.memset(ring.slot_address(0), 0x11, 128)
        ctypes.memset(ring.slot_address(0) + 128, 0x22, 128)
        ring.copy_batch_to_arena_async(0, arena, [(0, 0, 128), (128, 128, 128)])
        ring.wait_slot_async(0)
        torch.cuda.synchronize()
        assert ring.slot_ready(0)
        assert torch.equal(view[:128].cpu(), torch.full((128,), 0x11, dtype=torch.uint8))
        assert torch.equal(view[128:].cpu(), torch.full((128,), 0x22, dtype=torch.uint8))
        arena.unpin_range(0, 256)
        del buffer
    finally:
        ring.close()
        arena.close()


@pytest.mark.vmm
def test_model_arena_reuses_unpinned_pages_when_budget_is_full() -> None:
    """A bounded sequential scan reuses an unpinned physical page safely."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm

    library = load_native_vmm(library_path)
    page = library.device_granularity(0)
    arena = library.create_model_arena(
        page * 3,
        resident_budget_bytes=page,
        page_size_bytes=page,
    )
    try:
        first_signature = arena.fault_range(0, page)
        counters = arena.residency_counters
        assert counters["materialized_pages"] == 1
        assert counters["reused_pages"] == 0
        assert counters["evicted_pages"] == 0
        arena.unpin_range(0, page)
        assert arena.info.resident_size == page

        second_signature = arena.fault_range(page, page)
        assert arena.residency_counters["materialized_pages"] == 2
        assert arena.residency_counters["evicted_pages"] == 1
        assert arena.info.resident_size == page
        assert arena.info.fault_ceiling == page * 3
        arena.unpin_range(page, page)
        assert arena.fault_range(0, page) != first_signature
        assert arena.residency_counters["evicted_pages"] == 2
        assert second_signature != first_signature
    finally:
        arena.close()


@pytest.mark.vmm
def test_model_arena_pressure_preserves_low_address_resident_prefix() -> None:
    """Capacity pressure evicts the eligible high-address tail first."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm

    library = load_native_vmm(library_path)
    page = library.device_granularity(0)
    arena = library.create_model_arena(
        page * 3,
        resident_budget_bytes=page * 2,
        page_size_bytes=page,
    )
    try:
        arena.fault_range(0, page)
        arena.unpin_range(0, page)
        arena.fault_range(page, page)
        arena.unpin_range(page, page)
        arena.fault_range(page * 2, page)
        counters = arena.residency_counters
        assert counters["tail_evictions"] == 1
        assert counters["resident_prefix_pages"] == 1
        arena.unpin_range(page * 2, page)
    finally:
        arena.close()


@pytest.mark.vmm
def test_global_native_pressure_reclaims_unpinned_model_arena_tail() -> None:
    """Allocator-wide pressure may reclaim an unpinned model page safely."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm

    library = load_native_vmm(library_path)
    page = library.device_granularity(0)
    arena = library.create_model_arena(
        page * 2,
        resident_budget_bytes=page * 2,
        page_size_bytes=page,
    )
    try:
        arena.fault_range(0, page)
        arena.unpin_range(0, page)
        arena.fault_range(page, page)
        arena.unpin_range(page, page)

        assert library.pressure_evict_one(0)
        assert arena.info.resident_size == page
        assert arena.residency_counters["resident_prefix_pages"] == 1
        assert arena.residency_counters["pressure_evictions"] == 1
    finally:
        arena.close()


@pytest.mark.vmm
def test_model_arena_honors_requested_logical_page_size() -> None:
    """A requested logical page must remain visible in native arena accounting."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    library_path = _library_path()
    if not library_path.exists():
        pytest.skip(f"VMM library unavailable: {library_path}")

    from ddo_vmm_allocator.native import load_native_vmm

    library = load_native_vmm(library_path)
    requested_page = 32 * 2**20
    granularity = library.device_granularity(0)
    assert requested_page % granularity == 0
    arena = library.create_model_arena(
        requested_page * 2,
        resident_budget_bytes=requested_page,
        page_size_bytes=requested_page,
    )
    try:
        assert arena.info.page_size == requested_page
        arena.fault_range(0, 1)
        assert arena.info.resident_size == requested_page
        assert arena.residency_counters["materialized_pages"] == 1
        arena.unpin_range(0, 1)
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
