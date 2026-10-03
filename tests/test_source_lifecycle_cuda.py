"""Small isolated native failure gates; explicit CUDA approval required."""
import ctypes as c
import os
from pathlib import Path

import pytest

from ddo_vmm_allocator.native import load_native_vmm
from ddo_vmm_allocator.source import NativeSourceError
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
    assert path, "explicit isolated lifecycle library required"
    lib = load_native_vmm(Path(path))
    dll = c.CDLL(path)
    inject = dll.ddo_vmm_source_test_failures
    inject.argtypes = [c.c_void_p, c.c_int64, c.c_int]
    inject.restype = c.c_int
    return torch, lib, inject


@pytest.mark.parametrize("failure", ["partial-copy", "close-refusal"])
def test_native_failure_retains_batch_and_cache_then_closes_safely(runtime, tmp_path, failure):
    torch, lib, inject = runtime
    path = tmp_path / "weights.bin"
    path.write_bytes(b"\x11" * PAGE + b"\x22" * PAGE)
    with lib.create_host_ring(PAGE, 1) as ring, lib.open_source_file(path) as source, lib.create_source_cache(
        slot_bytes=PAGE, slots=2, persistent_buffer=True,
        registered_limit_bytes=2 * PAGE, process_limit_bytes=4 * PAGE,
    ) as cache, lib.create_model_arena(2 * PAGE, page_size_bytes=PAGE) as arena:
        arena.prepare_view_range(0, 2 * PAGE)
        view, view_owner = cuda_uint8_tensor_from_address(arena.info.address, 2 * PAGE, owner=arena, backing="device")
        arena.release_view_range(0, 2 * PAGE)
        arena.fault_range(0, 2 * PAGE)
        batch = cache.create_batch([(source, 0, PAGE, 0), (source, PAGE, PAGE, PAGE)],
                                  ring, ring_slots=1, ring_slot_bytes=PAGE)
        try:
            stream = torch.cuda.current_stream()
            view.zero_()
            if failure == "partial-copy":
                assert inject(cache._handle, 1, 0) == 0
                with pytest.raises(NativeSourceError, match="injected copy refusal") as error:
                    batch.submit(arena, stream)
                assert error.value.status == 7
                state = cache.snapshot()
                assert state["dma_bytes"] == PAGE and state["transfer_owned_events"] == 1
            else:
                batch.submit(arena, stream)
                assert inject(cache._handle, -1, 1) == 0
                with pytest.raises(NativeSourceError, match="injected ticket close refusal"):
                    batch.close()
            assert batch._handle.value and batch._destinations == [arena]
            assert cache.snapshot()["live_tickets"] == 2
            with pytest.raises(NativeSourceError) as busy:
                cache.close()
            assert busy.value.status == 9 and cache._handle.value
            batch.close()
            state = cache.snapshot()
            assert state["live_tickets"] == state["transfer_owned_events"] == state["batch_completion_events"] == 0
            assert 0 < state["pooled_events"] <= 256
            assert torch.all(view[:PAGE] == 17).item()
            assert torch.all(view[PAGE:] == (0 if failure == "partial-copy" else 34)).item()
            cache.clear()
            assert cache.snapshot()["committed_bytes"] == cache.snapshot()["registered_bytes"] == 0
        finally:
            batch.close()
            arena.unpin_range(0, 2 * PAGE)
            del view, view_owner


def test_native_event_ownership_stabilizes_over_repeated_batches(runtime, tmp_path):
    torch, lib, _inject = runtime
    path = tmp_path / "repeated.bin"
    path.write_bytes(b"\x33" * PAGE)
    with lib.create_host_ring(PAGE, 1) as ring, lib.open_source_file(path) as source, lib.create_source_cache(
        slot_bytes=PAGE, slots=1, persistent_buffer=True,
        registered_limit_bytes=PAGE, process_limit_bytes=3 * PAGE,
    ) as cache, lib.create_model_arena(PAGE, page_size_bytes=PAGE) as arena:
        arena.prepare_view_range(0, PAGE)
        view, view_owner = cuda_uint8_tensor_from_address(arena.info.address, PAGE, owner=arena, backing="device")
        arena.release_view_range(0, PAGE)
        arena.fault_range(0, PAGE)
        try:
            first = None
            for _ in range(16):
                batch = cache.create_batch([(source, 0, PAGE, 0)], ring, ring_slots=1, ring_slot_bytes=PAGE)
                try:
                    assert cache.snapshot()["batch_completion_events"] == 1
                    batch.submit(arena, torch.cuda.current_stream())
                finally:
                    batch.close()
                assert torch.all(view == 51).item()
                state = cache.snapshot()
                assert state["transfer_owned_events"] == state["batch_completion_events"] == state["live_tickets"] == 0
                current = (state["pooled_events"], state["events_created"], state["source_read_bytes"],
                           state["committed_bytes"], state["registered_bytes"])
                first = current if first is None else first
                assert current == first
        finally:
            arena.unpin_range(0, PAGE)
            del view, view_owner
