"""CUDA source gates: each candidate execution requires explicit approval."""
import os
from pathlib import Path

import pytest

from ddo_vmm_allocator.native import load_native_vmm
from ddo_vmm_allocator.tensor_view import cuda_uint8_tensor_from_address

pytestmark = pytest.mark.vmm
PAGE = 2 * 2**20


@pytest.fixture
def runtime():
    import torch
    if os.name != "nt" or not torch.cuda.is_available():
        pytest.skip("Windows CUDA required")
    torch.empty(1, device="cuda:0")
    path = os.environ.get("DDO_VMM_LIBRARY_PATH")
    assert path, "explicit dispatcher candidate required"
    return torch, load_native_vmm(Path(path))


@pytest.mark.parametrize("slots", [1, 2], ids=["bounded-cache-ring-fallback", "full-source"])
def test_source_only_prepare_retains_file_and_defers_registration(runtime, tmp_path, slots):
    torch, lib = runtime
    path = tmp_path / "source-only.bin"
    path.write_bytes(b"\x11" * PAGE + b"\x22" * PAGE)
    with lib.create_host_ring(PAGE, 1) as ring, lib.open_source_file(path) as source, lib.create_source_cache(
        slot_bytes=PAGE, slots=slots, persistent_buffer=True,
        registered_limit_bytes=PAGE, process_limit_bytes=4 * PAGE,
    ) as cache, lib.create_model_arena(2 * PAGE, page_size_bytes=PAGE) as arena:
        assert cache.source_dispatcher, "select dispatcher candidate, not protected DLL"
        arena.prepare_view_range(0, 2 * PAGE)
        view, owner = cuda_uint8_tensor_from_address(arena.info.address, 2 * PAGE, owner=arena, backing="device")
        arena.release_view_range(0, 2 * PAGE)
        arena.fault_range(0, 2 * PAGE)
        batch = cache.create_batch([(source, 0, PAGE, 0), (source, PAGE, PAGE, PAGE)],
                                  ring, ring_slots=1, ring_slot_bytes=PAGE)
        try:
            assert batch.prepare_source()
            source.close()
            before = cache.snapshot()
            assert before["registered_bytes"] == before["dma_bytes"] == before["live_tickets"] == 0
            assert before["committed_bytes"] <= slots * PAGE
            stream = torch.cuda.Stream()
            for _ in range(2):
                batch.submit(arena, stream)
                batch.wait_on_stream(torch.cuda.current_stream())
                assert torch.all(view[:PAGE] == 17).item()
                assert torch.all(view[PAGE:] == 34).item()
            after = cache.snapshot()
            assert after["registered_bytes"] <= PAGE
            assert after["committed_bytes"] <= slots * PAGE
            if slots == 2:
                assert after["source_read_bytes"] == 2 * PAGE
                assert after["batch_bypass_bytes"] == 0
            else:
                assert after["batch_bypass_bytes"] >= 2 * PAGE
        finally:
            batch.close()
            arena.unpin_range(0, 2 * PAGE)
            del view, owner
        assert cache.snapshot()["live_tickets"] == 0


def test_dispatcher_more_windows_than_flight_limit_preserves_payload_and_clear(runtime, tmp_path):
    """New multi-window gate: written only; execution requires confirmation."""
    torch, lib = runtime
    windows = 10
    path = tmp_path / 'multi-window.bin'
    path.write_bytes(b''.join(bytes([index + 1]) * PAGE for index in range(windows)))
    with lib.create_host_ring(PAGE, 1) as ring, lib.open_source_file(path) as source, lib.create_source_cache(
        slot_bytes=PAGE, slots=windows, persistent_buffer=True,
        registered_limit_bytes=PAGE, process_limit_bytes=(windows + 2) * PAGE,
    ) as cache, lib.create_model_arena(windows * PAGE, page_size_bytes=PAGE) as arena:
        assert cache.source_dispatcher
        arena.prepare_view_range(0, windows * PAGE)
        view, owner = cuda_uint8_tensor_from_address(arena.info.address, windows * PAGE, owner=arena, backing='device')
        arena.release_view_range(0, windows * PAGE)
        arena.fault_range(0, windows * PAGE)
        batch = cache.create_batch([(source, index * PAGE, PAGE, index * PAGE) for index in range(windows)],
                                  ring, ring_slots=1, ring_slot_bytes=PAGE)
        try:
            batch.prepare_source()
            source.close()
            before = cache.snapshot()
            assert before['registered_bytes'] == before['dma_bytes'] == 0
            stream = torch.cuda.Stream()
            for _ in range(2):
                batch.submit(arena, stream)
                batch.wait_on_stream(torch.cuda.current_stream())
                for index in range(windows):
                    assert torch.all(view[index * PAGE:(index + 1) * PAGE] == index + 1).item()
            after = cache.snapshot()
            assert after['source_dispatcher_window_limit'] == 8
            # Tiny cached reads may finish immediately: do not demand a timing
            # race to reach eight. Model gate separately checks real concurrency.
            assert 1 <= after['source_dispatcher_peak_windows'] <= 8
            assert after['source_dispatcher_owned_windows'] == 0
            assert after['source_read_bytes'] == windows * PAGE
            assert after['registered_bytes'] <= PAGE
            assert after['committed_bytes'] == windows * PAGE
            assert after['batch_bypass_bytes'] == 0
        finally:
            batch.close()
            arena.unpin_range(0, windows * PAGE)
            del view, owner
        cache.clear()
        cleared = cache.snapshot()
        assert cleared['live_tickets'] == cleared['committed_bytes'] == cleared['registered_bytes'] == 0
        assert cleared['source_dispatcher_owned_windows'] == cleared['source_dispatcher_peak_windows'] == 0
        assert cleared['source_dispatcher_window_limit'] == 0


def test_dispatcher_clear_drains_reads_and_restarts_without_stale_window(runtime, tmp_path):
    torch, lib = runtime
    path = tmp_path / "clear-restart.bin"
    path.write_bytes(b"\x33" * (4 * PAGE))
    with lib.create_host_ring(PAGE, 1) as ring, lib.create_source_cache(
        slot_bytes=PAGE, slots=4, persistent_buffer=True,
        registered_limit_bytes=PAGE, process_limit_bytes=6 * PAGE,
    ) as cache:
        assert cache.source_dispatcher
        with lib.open_source_file(path) as source:
            batch = cache.create_batch([(source, 0, 4 * PAGE, 0)], ring,
                                      ring_slots=1, ring_slot_bytes=PAGE)
            try:
                batch.prepare_source()
            finally:
                batch.close()
        # May still have queued/active reads. Clear must drain, not infer
        # completion from zero DMA tickets or from deleted batch metadata.
        cache.clear()
        cleared = cache.snapshot()
        assert cleared["live_tickets"] == cleared["committed_bytes"] == cleared["registered_bytes"] == 0
        path.write_bytes(b"\x66" * PAGE)  # No retained read owner may deny this.
        with lib.open_source_file(path) as source, lib.create_model_arena(PAGE, page_size_bytes=PAGE) as arena:
            arena.prepare_view_range(0, PAGE)
            view, owner = cuda_uint8_tensor_from_address(arena.info.address, PAGE, owner=arena, backing="device")
            arena.release_view_range(0, PAGE)
            arena.fault_range(0, PAGE)
            try:
                with cache.request(source, 0, PAGE) as ticket:
                    stream = torch.cuda.Stream()
                    ticket.copy_to_arena_async(arena, 0, stream)
                    ticket.wait_on_stream(torch.cuda.current_stream())
                    assert torch.all(view == 102).item()
                assert cache.snapshot()["source_read_bytes"] == cleared["source_read_bytes"] + PAGE
            finally:
                arena.unpin_range(0, PAGE)
                del view, owner
