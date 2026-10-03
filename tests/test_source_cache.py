"""Small hardware gates for retained native source bytes; no model harness."""
import os
import ctypes
from types import SimpleNamespace
from pathlib import Path

import pytest

from ddo_vmm_allocator.native import NativeVmmError, load_native_vmm
from ddo_vmm_allocator.source import NativeSourceError
from ddo_vmm_allocator.tensor_view import cuda_uint8_tensor_from_address
from ddo_vmm_allocator.source_cache import _Config, _Stats

pytestmark = pytest.mark.vmm
PAGE = 2 * 2**20


def test_persistent_storage_survives_registration_recycling_without_reread(runtime, tmp_path):
    """Two retained segments, one registered segment, live DMA tickets."""
    torch, lib = runtime
    path = tmp_path / "persistent.bin"
    path.write_bytes(b"\x11" * PAGE + b"\x22" * PAGE)
    with lib.open_source_file(path) as source, lib.create_source_cache(
        slot_bytes=PAGE, slots=2, persistent_buffer=True,
        registered_limit_bytes=PAGE, process_limit_bytes=2 * PAGE,
    ) as cache, lib.create_model_arena(2 * PAGE, page_size_bytes=PAGE) as arena:
        arena.prepare_view_range(0, 2 * PAGE)
        view, owner = cuda_uint8_tensor_from_address(arena.info.address, 2 * PAGE, owner=arena, backing="device")
        arena.release_view_range(0, 2 * PAGE)
        arena.fault_range(0, 2 * PAGE)
        try:
            first = cache.request(source, 0, 16)
            second = cache.request(source, PAGE, 16)
            assert first is not None and second is not None
            try:
                first.wait()
                second.wait()
                before = cache.snapshot()
                assert before["committed_bytes"] == 2 * PAGE
                assert before["registered_bytes"] == 0
                stream = torch.cuda.Stream()
                first.copy_to_arena_async(arena, 0, stream)
                # Keep first ticket alive while reusing its completed registration.
                second.copy_to_arena_async(arena, PAGE, stream)
                second.wait_on_stream(torch.cuda.current_stream())
                assert view[:16].cpu().tolist() == [17] * 16
                assert view[PAGE:PAGE + 16].cpu().tolist() == [34] * 16
            finally:
                second.close()
                first.close()
            cache.trim(0)
            unregistered = cache.snapshot()
            assert unregistered["registered_bytes"] == 0
            assert unregistered["committed_bytes"] == 2 * PAGE
            assert unregistered["retained_bytes"] == 2 * PAGE
            assert unregistered["unregistrations"] >= 2
            with cache.request(source, 0, 16) as repeated:
                repeated.copy_to_arena_async(arena, 0, stream)
                repeated.wait_on_stream(torch.cuda.current_stream())
                assert view[:16].cpu().tolist() == [17] * 16
            after = cache.snapshot()
            assert after["source_read_bytes"] == before["source_read_bytes"] == 2 * PAGE
            assert after["registered_bytes"] <= PAGE
            assert after["registered_limit_bytes"] == PAGE
            assert after["registrations"] >= 3
            assert after["version"] == 8
            assert after["registration_seconds"] > 0
            assert after["unregistration_seconds"] > 0
            assert after["registration_wait_seconds"] >= 0
            assert after["source_ready_seconds"] >= 0
            assert after["dma_submit_seconds"] > 0
            assert after["dma_event_seconds"] > 0
            assert after["ticket_close_wait_seconds"] >= 0
            assert after["host_copy_bytes"] == 0
            cache.clear()
            cleared = cache.snapshot()
            assert cleared["committed_bytes"] == cleared["registered_bytes"] == 0
        finally:
            arena.unpin_range(0, 2 * PAGE)
            del view, owner


def test_persistent_full_registration_survives_repeated_dma(runtime, tmp_path):
    """Full coverage removes normal-path register/unregister churn, not DMA."""
    torch, lib = runtime
    path = tmp_path / "registered-source.bin"
    path.write_bytes(b"\x11" * PAGE + b"\x22" * PAGE)
    with lib.open_source_file(path) as source, lib.create_source_cache(
        slot_bytes=PAGE, slots=2, persistent_buffer=True,
        registered_limit_bytes=2 * PAGE, process_limit_bytes=2 * PAGE,
    ) as cache, lib.create_model_arena(PAGE, page_size_bytes=PAGE) as arena:
        arena.prepare_view_range(0, PAGE)
        view, owner = cuda_uint8_tensor_from_address(
            arena.info.address, PAGE, owner=arena, backing="device",
        )
        arena.release_view_range(0, PAGE)
        arena.fault_range(0, PAGE)
        try:
            upload = torch.cuda.Stream()
            for pass_index in range(3):
                for index, value in enumerate((17, 34)):
                    with cache.request(source, index * PAGE, PAGE) as ticket:
                        assert ticket is not None
                        ticket.copy_to_arena_async(arena, 0, upload)
                        ticket.wait_on_stream(torch.cuda.current_stream())
                        # Validate the full payload and finish its consumer
                        # before overwriting the same destination next time.
                        assert torch.all(view.cpu() == value).item()
                stats = cache.snapshot()
                assert stats["source_read_bytes"] == 2 * PAGE
                assert stats["registered_bytes"] == stats["committed_bytes"] == 2 * PAGE
                assert stats["registrations"] == 2
                assert stats["unregistrations"] == stats["registration_waits"] == 0
                assert stats["dma_bytes"] == (pass_index + 1) * 2 * PAGE
                assert stats["ring_fallback_bytes"] == stats["host_copy_bytes"] == 0
                assert stats["live_tickets"] == 0
            cache.trim(0)
            trimmed = cache.snapshot()
            assert trimmed["registered_bytes"] == 0
            assert trimmed["committed_bytes"] == trimmed["retained_bytes"] == 2 * PAGE
            assert trimmed["unregistrations"] == 2
            with cache.request(source, 0, PAGE) as ticket:
                ticket.copy_to_arena_async(arena, 0, upload)
                ticket.wait_on_stream(torch.cuda.current_stream())
                assert torch.all(view.cpu() == 17).item()
            restored = cache.snapshot()
            assert restored["source_read_bytes"] == 2 * PAGE
            assert restored["registrations"] == 3
            assert restored["registered_bytes"] == PAGE
            cache.clear()
            cleared = cache.snapshot()
            assert cleared["registered_bytes"] == cleared["committed_bytes"] == 0
            assert cleared["unregistrations"] == cleared["registrations"]
        finally:
            arena.unpin_range(0, PAGE)
            del view, owner


@pytest.mark.parametrize("size", [16, PAGE + 16])
def test_persistent_registration_fallback_copies_retained_bytes_through_existing_ring(runtime, tmp_path, size):
    torch, lib = runtime
    path = tmp_path / "fallback-retained.bin"
    path.write_bytes(b"\x37" * (2 * PAGE))
    with lib.create_host_ring(PAGE, 1) as ring, lib.open_source_file(path) as source, lib.create_source_cache(
        slot_bytes=2 * PAGE, slots=1, persistent_buffer=True,
        registered_limit_bytes=2 * PAGE, process_limit_bytes=3 * PAGE,
    ) as cache, lib.create_model_arena(2 * PAGE, page_size_bytes=PAGE) as arena:
        arena.prepare_view_range(0, size)
        view, owner = cuda_uint8_tensor_from_address(arena.info.address, size, owner=arena, backing="device")
        arena.release_view_range(0, size)
        arena.fault_range(0, size)
        try:
            stream = torch.cuda.Stream()
            for _ in range(2):
                with cache.request(source, 0, size) as ticket:
                    ring.wait_slot(0)
                    ticket.copy_via_ring_async(ring, 0, arena, 0, stream)
                    ticket.wait_on_stream(torch.cuda.current_stream())
                    assert torch.all(view == 55).item()
            stats = cache.snapshot()
            assert stats["source_read_bytes"] == 2 * PAGE
            assert stats["committed_bytes"] == 2 * PAGE
            assert stats["registered_bytes"] == 0
            assert stats["host_copy_bytes"] == stats["ring_fallback_bytes"] == 2 * size
        finally:
            arena.unpin_range(0, size)
            del view, owner


@pytest.mark.parametrize("version", [4, 5, 6, 7])
def test_timing_stats_preserve_older_snapshot_prefix(runtime, version):
    """Old callers receive only their prefix, never new timing suffix bytes."""
    _, lib = runtime
    with lib.create_source_cache(slot_bytes=PAGE, slots=1) as cache:
        stats = _Stats(struct_size=ctypes.sizeof(_Stats), version=version,
                       registration_seconds=-42, unregistration_seconds=-42,
                       registration_wait_seconds=-42, source_ready_seconds=-43,
                       dma_submit_seconds=-43, dma_event_seconds=-43,
                       ticket_close_wait_seconds=-43,
                       request_total_seconds=-44, copy_total_seconds=-44,
                       close_total_seconds=-44, events_created=777,
                       batch_submissions=777)
        cache._check(cache._dll.ddo_vmm_source_cache_snapshot(cache._handle, ctypes.byref(stats)))
        expected_size = {4: _Stats.committed_bytes.offset,
                         5: _Stats.registration_seconds.offset,
                         6: _Stats.source_ready_seconds.offset,
                         7: _Stats.request_total_seconds.offset}[version]
        assert stats.struct_size == expected_size
        assert stats.version == version
        expected_registration = 0 if version >= 6 else -42
        assert stats.registration_seconds == stats.unregistration_seconds == stats.registration_wait_seconds == expected_registration
        expected_transfer = 0 if version >= 7 else -43
        assert stats.source_ready_seconds == stats.dma_submit_seconds == stats.dma_event_seconds == stats.ticket_close_wait_seconds == expected_transfer
        assert stats.request_total_seconds == stats.copy_total_seconds == stats.close_total_seconds == -44
        assert stats.events_created == stats.batch_submissions == 777


def test_immutable_batch_retains_sources_recycles_events_and_guards_close(runtime, tmp_path):
    torch, lib = runtime
    path = tmp_path / "batch.bin"
    path.write_bytes(b"\x11" * PAGE + b"\x22" * PAGE)
    with lib.create_host_ring(PAGE, 2) as ring, lib.open_source_file(path) as source, lib.create_source_cache(
        slot_bytes=PAGE, slots=2, persistent_buffer=True,
        registered_limit_bytes=PAGE, process_limit_bytes=4 * PAGE,
    ) as cache, lib.create_model_arena(2 * PAGE, page_size_bytes=PAGE) as arena:
        arena.prepare_view_range(0, 2 * PAGE)
        view, owner = cuda_uint8_tensor_from_address(arena.info.address, 2 * PAGE, owner=arena, backing="device")
        arena.release_view_range(0, 2 * PAGE)
        arena.fault_range(0, 2 * PAGE)
        batch = cache.create_batch([(source, 0, PAGE, 0), (source, PAGE, PAGE, PAGE)], ring,
                                   ring_slots=2, ring_slot_bytes=PAGE)
        try:
            with pytest.raises(NativeSourceError) as error:
                cache.close()
            assert error.value.status == 9
            source.close()  # Immutable native descriptors retain independent owners.
            stream = torch.cuda.Stream()
            with pytest.raises(NativeSourceError) as error:
                batch.submit(arena, stream, dest_base=PAGE)
            assert error.value.status == 1
            for _ in range(2):
                batch.submit(arena, stream)
                batch.wait_on_stream(torch.cuda.current_stream())
                assert torch.all(view[:PAGE] == 17).item()
                assert torch.all(view[PAGE:] == 34).item()
            stats = cache.snapshot()
            assert stats["version"] == 8
            assert stats["source_read_bytes"] == 2 * PAGE
            assert stats["batch_submissions"] == 2
            assert stats["events_reused"] > 0
            assert stats["batch_submit_seconds"] > 0
            assert stats["registered_bytes"] <= PAGE
        finally:
            batch.close()
            arena.unpin_range(0, 2 * PAGE)
            del view, owner


def test_collect_completed_retires_only_ready_batch_owned_tickets(runtime, tmp_path):
    torch, lib = runtime
    path = tmp_path / "collect.bin"
    path.write_bytes(b"\x5a" * PAGE)
    with lib.create_host_ring(PAGE, 2) as ring, lib.open_source_file(path) as source, lib.create_source_cache(
        slot_bytes=PAGE, slots=1, persistent_buffer=True, process_limit_bytes=4 * PAGE,
    ) as cache, lib.create_model_arena(PAGE, page_size_bytes=PAGE) as arena:
        arena.prepare_view_range(0, PAGE)
        view, owner = cuda_uint8_tensor_from_address(arena.info.address, PAGE, owner=arena, backing="device")
        arena.release_view_range(0, PAGE)
        arena.fault_range(0, PAGE)
        batch = cache.create_batch([(source, 0, PAGE, 0)], ring, ring_slots=2, ring_slot_bytes=PAGE)
        try:
            stream = torch.cuda.Stream()
            batch.submit(arena, stream)
            stream.synchronize()
            cache.collect_completed()
            assert cache.snapshot()["live_tickets"] == 0
            external = cache.request(source, 0, PAGE)
            try:
                external.wait()
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(250_000_000)
                batch.submit(arena, stream)
                assert not stream.query(), "diagnostic delay completed before collection"
                cache.collect_completed()
                assert cache.snapshot()["live_tickets"] == 2
                batch.wait_on_stream(torch.cuda.current_stream())
                assert torch.equal(view.cpu(), torch.full((PAGE,), 0x5A, dtype=torch.uint8))
                cache.collect_completed()
                stats = cache.snapshot()
                assert stats["live_tickets"] == 1  # External owner is never stolen.
                assert stats["source_read_bytes"] == PAGE
                assert stats["dma_bytes"] == 2 * PAGE
                assert stats["events_reused"] >= 1
            finally:
                external.close()
            assert cache.snapshot()["live_tickets"] == 0
        finally:
            batch.close()
            arena.unpin_range(0, PAGE)
            del view, owner


def test_speculative_cache_hit_and_ring_bypass_feed_one_dma_dependency(runtime, tmp_path):
    from ddo_vmm_allocator.model_arena_adapter import _NativeShardTransport

    torch, lib = runtime
    payload = bytes(range(256)) * (PAGE // 256) + b"tail-source!!"
    path = tmp_path / "speculative.bin"
    path.write_bytes(payload)
    bindings = [
        SimpleNamespace(nbytes=8, offset=0, file_slice=SimpleNamespace(path=path, offset_bytes=0)),
        SimpleNamespace(nbytes=13, offset=8, file_slice=SimpleNamespace(path=path, offset_bytes=PAGE)),
    ]
    with lib.create_model_arena(PAGE, page_size_bytes=PAGE) as arena:
        transport = _NativeShardTransport(
            lib, slot_size_bytes=PAGE, slots=2, device_ordinal=0,
            cache_slot_bytes=PAGE, cache_slots=1, cache_process_limit_bytes=3 * PAGE,
            cache_model_windows=True,
        )
        arena.prepare_view_range(0, 21)
        view, buffer = cuda_uint8_tensor_from_address(arena.info.address, 21, owner=arena, backing="device")
        arena.release_view_range(0, 21)
        arena.fault_range(0, 21)
        try:
            upload = torch.cuda.Stream()
            ticket = transport.begin_read(bindings)
            assert not transport._cache_dma_tickets
            ready = transport.upload_ticket(ticket, arena, upload)
            torch.cuda.current_stream().wait_event(ready)
            assert bytes(view.cpu().tolist()) == payload[:8] + payload[PAGE:]
            transport.reap_cache_dma()
            assert not transport._cache_read_tickets
            assert transport.cache.snapshot()["live_tickets"] == 0
            # Reused-source speculation may be cancelled without DMA or reread.
            cancelled = transport.begin_read(bindings)
            transport.discard_ticket(cancelled)
            stats = transport.cache.snapshot()
            assert stats["source_read_bytes"] == PAGE
            assert stats["live_tickets"] == 0
            assert stats["hits"] == 1
        finally:
            transport.close()
            arena.unpin_range(0, 21)
            del view, buffer


@pytest.fixture
def runtime():
    import torch
    if os.name != "nt" or not torch.cuda.is_available():
        pytest.skip("Windows CUDA required")
    torch.empty(1, device="cuda:0")  # Establish primary context explicitly.
    path = Path(os.environ.get("DDO_VMM_LIBRARY_PATH", Path(__file__).parents[1] / "build/Release/ddo_vmm.dll"))
    return torch, load_native_vmm(path)


def test_cached_source_survives_gpu_eviction_without_reread(runtime, tmp_path):
    torch, lib = runtime
    payload = bytes(range(256)) * 1024
    path = tmp_path / "weights.bin"
    path.write_bytes(payload)
    with lib.open_source_file(path) as source, lib.create_source_cache(slot_bytes=PAGE, slots=2) as cache:
        with lib.create_model_arena(PAGE * 2, resident_budget_bytes=PAGE, page_size_bytes=PAGE) as arena:
            arena.prepare_view_range(0, len(payload))
            view, buffer = cuda_uint8_tensor_from_address(arena.info.address, len(payload), owner=arena, backing="device")
            arena.release_view_range(0, len(payload))
            try:
                for _ in range(2):
                    arena.fault_range(0, len(payload))
                    with cache.request(source, 0, len(payload)) as ticket:
                        ticket.copy_to_arena_async(arena)
                        ticket.wait_on_stream()
                        assert torch.equal(view.cpu(), torch.tensor(list(payload), dtype=torch.uint8))
                    arena.unpin_range(0, len(payload))
                    arena.fault_range(PAGE, 1)  # Evict the first physical page.
                    arena.unpin_range(PAGE, 1)
                stats = cache.snapshot()
                assert stats["hits"] == 1 and stats["misses"] == 1
                assert stats["source_read_bytes"] == len(payload)
                assert stats["dma_bytes"] == 2 * len(payload)
                assert stats["live_tickets"] == 0
            finally:
                del view, buffer


def test_model_windows_share_neighbor_ranges_and_clamp_eof(runtime, tmp_path):
    torch, lib = runtime
    payload = bytes(range(256)) * (PAGE // 256) + b"tail!"
    path = tmp_path / "windowed.bin"
    path.write_bytes(payload)
    with lib.open_source_file(path) as source, lib.create_source_cache(
        slot_bytes=PAGE, slots=2, model_windows=True,
    ) as cache, lib.create_model_arena(PAGE, page_size_bytes=PAGE) as arena:
        # Adjacent tensors share one native file window, not one slot each.
        arena.prepare_view_range(0, 37)
        view, buffer = cuda_uint8_tensor_from_address(arena.info.address, 37, owner=arena, backing="device")
        arena.release_view_range(0, 37)
        arena.fault_range(0, 37)
        try:
            with cache.request(source, 13, 32) as first, cache.request(source, 100, 17) as neighbor:
                first.copy_to_arena_async(arena)
                first.wait_on_stream()
                assert torch.equal(view[:32].cpu(), torch.tensor(list(payload[13:45]), dtype=torch.uint8))
                neighbor.wait()
            with cache.request(source, PAGE, 5) as tail:
                tail.copy_to_arena_async(arena, 32)
                tail.wait_on_stream()
                assert bytes(view[32:37].cpu().tolist()) == b"tail!"
            # A ticket must never straddle its native owner window.
            assert cache.request(source, PAGE - 1, 2) is None
            stats = cache.snapshot()
            # Misses include requests rejected before admission, not just fills.
            assert stats["misses"] == 3 and stats["hits"] == 1
            assert stats["bypasses"] == 1
            assert stats["source_read_bytes"] == len(payload)
            assert stats["retained_bytes"] == len(payload)
            assert stats["dma_bytes"] == 37
            assert stats["live_tickets"] == 0
            cache.clear()
            assert cache.snapshot()["registered_bytes"] == 0
        finally:
            arena.unpin_range(0, 37)
            del view, buffer


def test_native_signature_observation_does_not_fault_pin_or_change_counters(runtime):
    _torch, lib = runtime
    with lib.create_model_arena(3 * PAGE, resident_budget_bytes=PAGE, page_size_bytes=PAGE) as arena:
        assert arena.peek_range_signature(0, 16) is None
        signature = arena.fault_range(0, 16)
        arena.unpin_range(0, 16)
        before = arena.residency_counters
        assert arena.peek_range_signature(0, 16) == signature
        assert arena.residency_counters == before
        # If observation pinned the old page, this capacity fault would fail.
        arena.fault_range(PAGE, 16)
        arena.unpin_range(PAGE, 16)
        assert arena.peek_range_signature(0, 16) is None
        replacement = arena.fault_range(0, 16)
        arena.unpin_range(0, 16)
        assert replacement != signature
        assert arena.peek_range_signature(0, 16) == replacement


def test_partial_cache_bypasses_without_discarding_hit(runtime, tmp_path):
    _, lib = runtime
    path = tmp_path / "weights.bin"
    path.write_bytes(b"a" * 256)
    with lib.open_source_file(path) as source, lib.create_source_cache(slot_bytes=PAGE, slots=1) as cache:
        with cache.request(source, 0, 128) as ticket:
            ticket.wait()
        for _ in range(3):
            assert cache.request(source, 128, 128) is None
            with cache.request(source, 0, 128) as ticket:
                ticket.wait()
        assert cache.snapshot()["source_read_bytes"] == 128
        assert cache.snapshot()["bypasses"] == 3


def test_contained_alias_and_concurrent_fill_share_source(runtime, tmp_path):
    _, lib = runtime
    path = tmp_path / "weights.bin"
    alias = tmp_path / "alias.bin"
    path.write_bytes(b"z" * (PAGE + 13))
    os.link(path, alias)
    with lib.open_source_file(path) as source, lib.open_source_file(alias) as other:
        with lib.create_source_cache(slot_bytes=2 * PAGE, slots=1) as cache:
            with cache.request(source, 0, PAGE + 13) as first:
                with cache.request(other, 10, PAGE) as second:
                    first.wait()
                    second.wait()
            assert cache.snapshot()["source_read_bytes"] == PAGE + 13
            assert cache.snapshot()["hits"] == 1


def test_live_tickets_protect_cache_close_and_clear(runtime, tmp_path):
    _, lib = runtime
    path = tmp_path / "weights.bin"
    path.write_bytes(b"z" * PAGE)
    source = lib.open_source_file(path)
    cache = lib.create_source_cache(slot_bytes=PAGE, slots=1)
    ticket = cache.request(source, 0, PAGE)
    source.close()
    try:
        for operation in (cache.clear, cache.close):
            with pytest.raises(NativeSourceError) as error:
                operation()
            assert error.value.status == 9
        ticket.close()  # Cancellation leaves fill owned by the cache.
        cache.clear()  # Drains I/O before dropping the file owner.
        assert cache.snapshot()["retained_bytes"] == 0
        path.unlink()
    finally:
        ticket.close()
        cache.close()


def test_independent_dma_tickets_on_two_streams(runtime, tmp_path):
    torch, lib = runtime
    payload = bytes(range(256)) * 512
    path = tmp_path / "weights.bin"
    path.write_bytes(payload)
    with lib.open_source_file(path) as source, lib.create_source_cache(slot_bytes=PAGE, slots=1) as cache:
        with lib.create_model_arena(PAGE * 2, page_size_bytes=PAGE) as arena:
            views = []
            for offset in (0, PAGE):
                arena.prepare_view_range(offset, len(payload))
                views.append(cuda_uint8_tensor_from_address(arena.info.address + offset, len(payload), owner=arena, backing="device"))
                arena.release_view_range(offset, len(payload))
                arena.fault_range(offset, len(payload))
            first, second = cache.request(source, 0, len(payload)), cache.request(source, 0, len(payload))
            try:
                with pytest.raises(NativeSourceError) as error:
                    first.dma_ready()
                assert error.value.status == 1
                a, b = torch.cuda.Stream(), torch.cuda.Stream()
                first.copy_to_arena_async(arena, 0, a)
                second.copy_to_arena_async(arena, PAGE, b)
                assert isinstance(first.dma_ready(), bool)
                assert isinstance(second.dma_ready(), bool)
                a.synchronize()
                assert first.dma_ready()
                first.close()
                with pytest.raises(NativeSourceError):
                    cache.clear()
                second.wait_on_stream(torch.cuda.current_stream())
                b.synchronize()
                assert second.dma_ready()
                for view, _ in views:
                    assert torch.equal(view.cpu(), torch.tensor(list(payload), dtype=torch.uint8))
                second.close()
                cache.clear()
            finally:
                first.close(); second.close()
                for offset in (0, PAGE):
                    arena.unpin_range(offset, len(payload))
                views.clear()


def test_headroom_denial_and_changed_source(runtime, tmp_path):
    _, lib = runtime
    with pytest.raises(NativeSourceError) as error:
        lib.create_source_cache(slot_bytes=PAGE, slots=1, host_headroom_bytes=2**63)
    assert error.value.status == 8
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abc")
    with lib.open_source_file(path) as source, lib.create_source_cache(slot_bytes=PAGE, slots=1) as cache:
        with cache.request(source, 0, 3) as ticket:
            ticket.wait()
            stamp = path.stat()
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2_000_000_000))
            with pytest.raises(NativeSourceError) as error:
                ticket.wait()
            assert error.value.status == 5


def test_progressive_registration_and_clear_release_capacity(runtime, tmp_path):
    _, lib = runtime
    path = tmp_path / "weights.bin"
    path.write_bytes(b"p" * (PAGE * 2))
    with lib.open_source_file(path) as source, lib.create_source_cache(slot_bytes=PAGE, slots=2) as cache:
        assert cache.snapshot()["registered_bytes"] == 0
        assert cache.snapshot()["capacity_bytes"] == PAGE * 2
        class LegacyStats(ctypes.Structure):
            _fields_ = _Stats._fields_[:10]

        class GuardedLegacyStats(ctypes.Structure):
            _fields_ = [("stats", LegacyStats), ("sentinel", ctypes.c_uint64)]

        legacy = GuardedLegacyStats()
        legacy.stats.struct_size = ctypes.sizeof(LegacyStats)
        legacy.stats.version = 1
        legacy.sentinel = 0xFEDCBA9876543210
        status = cache._dll.ddo_vmm_source_cache_snapshot(
            cache._handle, ctypes.cast(ctypes.byref(legacy), ctypes.POINTER(_Stats)))
        assert status == 0 and legacy.stats.version == 1
        assert legacy.sentinel == 0xFEDCBA9876543210
        class V2Stats(ctypes.Structure):
            _fields_ = _Stats._fields_[:15]

        class GuardedV2Stats(ctypes.Structure):
            _fields_ = [("stats", V2Stats), ("sentinel", ctypes.c_uint64)]

        v2 = GuardedV2Stats()
        v2.stats.struct_size = ctypes.sizeof(V2Stats)
        v2.stats.version = 2
        v2.sentinel = 0x0123456789ABCDEF
        status = cache._dll.ddo_vmm_source_cache_snapshot(
            cache._handle, ctypes.cast(ctypes.byref(v2), ctypes.POINTER(_Stats)))
        assert status == 0 and v2.stats.version == 2
        assert v2.sentinel == 0x0123456789ABCDEF
        class V3Stats(ctypes.Structure):
            _fields_ = _Stats._fields_[:16]

        class GuardedV3Stats(ctypes.Structure):
            _fields_ = [("stats", V3Stats), ("sentinel", ctypes.c_uint64)]

        v3 = GuardedV3Stats()
        v3.stats.struct_size = ctypes.sizeof(V3Stats)
        v3.stats.version = 3
        v3.sentinel = 0xABCDEF0123456789
        status = cache._dll.ddo_vmm_source_cache_snapshot(
            cache._handle, ctypes.cast(ctypes.byref(v3), ctypes.POINTER(_Stats)))
        assert status == 0 and v3.stats.version == 3
        assert v3.sentinel == 0xABCDEF0123456789
        with cache.request(source, 0, PAGE) as ticket:
            ticket.wait()
        assert cache.snapshot()["registered_bytes"] == PAGE
        with cache.request(source, 0, PAGE) as ticket:
            ticket.wait()
        assert cache.snapshot()["registered_bytes"] == PAGE
        with cache.request(source, PAGE, PAGE) as ticket:
            ticket.wait()
        assert cache.snapshot()["registered_bytes"] == PAGE * 2
        cache.clear()
        stats = cache.snapshot()
        assert stats["registered_bytes"] == stats["retained_bytes"] == 0
        assert stats["peak_registered_bytes"] == PAGE * 2
        with cache.request(source, 0, PAGE) as ticket:
            ticket.wait()
        assert cache.snapshot()["registered_bytes"] == PAGE


def test_two_caches_share_capacity_and_refund_it(runtime, tmp_path):
    _, lib = runtime
    path = tmp_path / "weights.bin"
    path.write_bytes(b"a" * (PAGE * 3))
    with lib.open_source_file(path) as source:
        with lib.create_source_cache(slot_bytes=PAGE, slots=2, process_limit_bytes=2 * PAGE) as first:
            with lib.create_source_cache(slot_bytes=PAGE, slots=2, process_limit_bytes=2 * PAGE) as second:
                with first.request(source, 0, PAGE) as ticket:
                    ticket.wait()
                with second.request(source, PAGE, PAGE) as ticket:
                    ticket.wait()
                assert first.snapshot()["process_registered_bytes"] == PAGE * 2
                assert first.request(source, 2 * PAGE, PAGE) is None
                assert first.snapshot()["admission_denials"] == 1
                second.clear()
                assert first.snapshot()["process_registered_bytes"] == PAGE
                with first.request(source, 2 * PAGE, PAGE) as ticket:
                    ticket.wait()
                assert first.snapshot()["process_registered_bytes"] == PAGE * 2
    with lib.create_source_cache(slot_bytes=PAGE, slots=1) as empty:
        assert empty.snapshot()["process_registered_bytes"] == 0


def test_strictest_live_limit_wins_and_unsafe_lower_limit_is_rejected(runtime, tmp_path):
    _, lib = runtime
    path = tmp_path / "weights.bin"
    path.write_bytes(b"x" * (PAGE * 2))
    with lib.open_source_file(path) as source:
        with lib.create_source_cache(slot_bytes=PAGE, slots=2, process_limit_bytes=2 * PAGE) as first:
            with first.request(source, 0, PAGE) as ticket:
                ticket.wait()
            with lib.create_source_cache(slot_bytes=PAGE, slots=1, process_limit_bytes=PAGE):
                assert first.snapshot()["process_limit_bytes"] == PAGE
                assert first.request(source, PAGE, PAGE) is None
            assert first.snapshot()["process_limit_bytes"] == PAGE * 2
            with first.request(source, PAGE, PAGE) as ticket:
                ticket.wait()
            with pytest.raises(NativeSourceError) as error:
                lib.create_source_cache(slot_bytes=PAGE, slots=1, process_limit_bytes=PAGE)
            assert error.value.status == 8
            assert first.snapshot()["process_limit_bytes"] == PAGE * 2


def test_staging_ring_consumes_and_refunds_cache_admission_capacity(runtime, tmp_path):
    _, lib = runtime
    path = tmp_path / "weights.bin"
    path.write_bytes(b"s" * (PAGE * 2))
    with lib.open_source_file(path) as source:
        with lib.create_source_cache(slot_bytes=PAGE, slots=2, process_limit_bytes=PAGE * 2) as cache:
            with cache.request(source, 0, PAGE) as ticket:
                ticket.wait()
            assert cache.snapshot()["process_staging_registered_bytes"] == 0
            with lib.create_host_ring(PAGE, 1):
                stats = cache.snapshot()
                assert stats["process_registered_bytes"] == PAGE
                assert stats["process_staging_registered_bytes"] == PAGE
                assert cache.request(source, PAGE, PAGE) is None
                assert cache.snapshot()["admission_denials"] == 1
            assert cache.snapshot()["process_staging_registered_bytes"] == 0
            with cache.request(source, PAGE, PAGE) as ticket:
                ticket.wait()
            assert cache.snapshot()["process_registered_bytes"] == PAGE * 2


def test_existing_staging_ring_blocks_undersized_cache_cap(runtime):
    _, lib = runtime
    with lib.create_host_ring(PAGE, 2):
        with pytest.raises(NativeSourceError) as error:
            lib.create_source_cache(slot_bytes=PAGE, slots=1, process_limit_bytes=PAGE)
        assert error.value.status == 8
    with lib.create_source_cache(slot_bytes=PAGE, slots=1, process_limit_bytes=PAGE) as cache:
        assert cache.snapshot()["process_staging_registered_bytes"] == 0


def test_live_cache_guards_later_ring_and_preserves_accounting(runtime):
    _, lib = runtime
    with lib.create_source_cache(slot_bytes=PAGE, slots=1, process_limit_bytes=PAGE * 2) as cache:
        with pytest.raises(NativeVmmError, match="active source-cache process limit"):
            lib.create_host_ring(PAGE, 3)
        assert cache.snapshot()["process_staging_registered_bytes"] == 0
        with lib.create_host_ring(PAGE, 2):
            assert cache.snapshot()["process_staging_registered_bytes"] == PAGE * 2
        assert cache.snapshot()["process_staging_registered_bytes"] == 0


def test_extended_process_cap_preserves_legacy_config_ceiling(runtime):
    _, lib = runtime
    with lib.create_source_cache(slot_bytes=PAGE, slots=1, process_limit_bytes=256 * 2**20) as cache:
        assert cache.snapshot()["registered_bytes"] == 0
        assert cache.snapshot()["capacity_bytes"] == PAGE
        assert cache.snapshot()["process_limit_bytes"] == 256 * 2**20
        with lib.create_source_cache(slot_bytes=PAGE, slots=1):
            assert cache.snapshot()["process_limit_bytes"] == 128 * 2**20
        assert cache.snapshot()["process_limit_bytes"] == 256 * 2**20
        # The C ABI must reject oversized v2 caps even when the Python wrapper
        # has learned to select config v3 for explicitly larger process caps.
        for version, process_limit in ((2, 256 * 2**20), (3, 2**30 + 1)):
            config = _Config(ctypes.sizeof(_Config), version, 0, 1, PAGE, 2**30, process_limit)
            handle = ctypes.c_void_p()
            status = cache._dll.ddo_vmm_source_cache_create(ctypes.byref(config), ctypes.byref(handle))
            assert status == 1 and not handle.value
        assert cache.snapshot()["process_limit_bytes"] == 256 * 2**20


def test_trim_skips_live_tail_and_reuses_remaining_source(runtime, tmp_path):
    _, lib = runtime
    path = tmp_path / "weights.bin"
    path.write_bytes(b"t" * (PAGE * 2))
    with lib.open_source_file(path) as source, lib.create_source_cache(slot_bytes=PAGE, slots=2) as cache:
        with cache.request(source, 0, PAGE) as first:
            first.wait()
        tail = cache.request(source, PAGE, PAGE)
        try:
            tail.wait()
            with pytest.raises(NativeSourceError) as error:
                cache.trim(PAGE)
            assert error.value.status == 9
            assert cache.snapshot()["registered_bytes"] == PAGE * 2
        finally:
            tail.close()
        cache.trim(PAGE)
        stats = cache.snapshot()
        assert stats["registered_bytes"] == PAGE
        assert stats["pressure_evictions"] == 1
        assert stats["pressure_released_bytes"] == PAGE
        reads = stats["source_read_bytes"]
        with cache.request(source, 0, PAGE) as first:
            first.wait()
        assert cache.snapshot()["source_read_bytes"] == reads
        with cache.request(source, PAGE, PAGE) as tail:
            tail.wait()
        assert cache.snapshot()["source_read_bytes"] == reads + PAGE
