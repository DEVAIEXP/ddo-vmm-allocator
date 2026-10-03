"""CPU-only regression for window ownership before native ring submission."""

from types import SimpleNamespace

import pytest
from ddo_vmm_allocator.model_arena_adapter import ModelArenaError, _NativeShardTransport


class _Ring:
    def __init__(self, payload: bytes, slots: int) -> None:
        self.payload = payload
        self.data = [None] * slots
        self.busy = [False] * slots
        self.reads = []
        self.batches = []
        self.waits = 0

    def slot_ready(self, slot):
        return not self.busy[slot]

    def wait_slot(self, slot):
        self.waits += 1
        self.busy[slot] = False

    def wait_file_read(self, slot):
        pass

    def read_file_async(self, slot, _path, offset, *, size_bytes):
        assert not self.busy[slot], "overwriting a slot whose DMA is still active"
        self.data[slot] = self.payload[offset:offset + size_bytes]
        self.reads.append((slot, offset, size_bytes))

    def copy_ticket_to_arena_async(self, ranges, arena, *, stream):
        assert len({slot for slot, _, _ in ranges}) == len(ranges)
        self.batches.append(list(ranges))
        for slot, offset, size in ranges:
            arena[offset:offset + size] = self.data[slot][:size]
            self.busy[slot] = True


def _transport(payload: bytes, *, window_bytes: int, slots: int):
    transport = object.__new__(_NativeShardTransport)
    transport.cache = None
    transport.window_size_bytes = window_bytes
    transport.slots = slots
    transport.ring = _Ring(payload, slots)
    transport._next_slot = 0
    transport._pending = {}
    transport._cache_read_tickets = []
    transport._cache_dma_tickets = []
    transport.read_seconds = transport.read_bytes = 0
    transport.slot_wait_seconds = transport.slot_waits = 0
    return transport


def _binding(source_offset, size, arena_offset):
    return SimpleNamespace(
        nbytes=size, offset=arena_offset,
        file_slice=SimpleNamespace(path="synthetic.bin", offset_bytes=source_offset),
    )


@pytest.mark.parametrize("cache_bypass", [False, True], ids=["ring-only", "cache-bypass"])
def test_tiny_ranges_do_not_recycle_a_slot_inside_one_ticket(cache_bypass):
    transport = _transport(b"abcdWXYZ", window_bytes=16, slots=1)
    if cache_bypass:
        transport.cache = SimpleNamespace(request=lambda *_args: None, model_windows=False)
        transport._source = lambda path: path
        transport.reap_cache_dma = lambda: None
    arena = bytearray(12)
    ready = transport.upload_batch([_binding(0, 4, 0), _binding(4, 4, 8)], arena, "upload")
    assert arena[:4] == b"abcd"
    assert arena[8:] == b"WXYZ"
    assert len(transport.ring.batches) == 2
    assert transport.ring.waits == 1
    assert transport.read_bytes == 8
    assert not transport._pending
    assert ready.slot_index == 0


def test_one_range_larger_than_the_ring_progresses_through_bounded_tickets():
    payload = bytes(range(13))
    transport = _transport(payload, window_bytes=4, slots=2)
    arena = bytearray(13)
    transport.upload_batch([_binding(0, 13, 0)], arena, "upload")
    assert arena == payload
    assert [len(batch) for batch in transport.ring.batches] == [2, 2]
    assert transport.ring.waits == 2
    assert transport.read_bytes == 13
    assert not transport._pending


def test_speculative_ticket_rejects_excess_windows_before_any_read():
    transport = _transport(b"abcdWXYZ", window_bytes=16, slots=1)
    with pytest.raises(ModelArenaError, match="ring window count"):
        transport.begin_read([_binding(0, 4, 0), _binding(4, 4, 8)])
    assert not transport.ring.reads


@pytest.mark.parametrize("cancel", [False, True], ids=["consume", "cancel"])
def test_cache_prefetch_owns_hit_and_bypass_without_duplicate_read(monkeypatch, cancel):
    import torch

    payload = b"abcdWXYZ"
    calls = []
    transport = _transport(payload, window_bytes=4, slots=2)

    class SourceTicket:
        def copy_to_arena_async(self, arena, offset, stream):
            calls.append("dma")
            arena[offset:offset + 4] = payload[:4]

        def close(self):
            calls.append("close")

    def request(source, offset, size):
        calls.append(("request", offset, size))
        return SourceTicket() if offset == 0 else None

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            calls.append("event")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    transport.cache = SimpleNamespace(model_windows=True, slot_bytes=4, request=request)
    transport._source = lambda path: path
    transport.reap_cache_dma = lambda: None
    ticket = transport.begin_read([_binding(0, 4, 0), _binding(4, 4, 4)])
    assert transport.ticket_active(ticket)
    assert calls == [("request", 0, 4), ("request", 4, 4)]
    assert transport.ring.reads == [(0, 4, 4)]  # No ring reread of cache hit.
    assert not transport._cache_dma_tickets  # Never poll an unsubmitted DMA.
    if cancel:
        transport.discard_ticket(ticket)
        assert calls[-1] == "close"
        assert not transport.ring.batches
    else:
        arena = bytearray(8)
        transport.upload_ticket(ticket, arena, "upload")
        assert arena == payload
        assert calls[-2:] == ["dma", "event"]
        assert len(transport._cache_dma_tickets) == 1
    assert not transport.ticket_active(ticket)
    assert not transport._pending
    assert not transport._cache_read_tickets


def test_cache_prefetch_failure_cancels_already_acquired_sources():
    transport = _transport(b"abcdefgh", window_bytes=4, slots=2)
    closed = []

    def request(source, offset, size):
        if offset:
            raise RuntimeError("injected source failure")
        return SimpleNamespace(close=lambda: closed.append(offset))

    transport.cache = SimpleNamespace(model_windows=True, slot_bytes=4, request=request)
    transport._source = lambda path: path
    transport.reap_cache_dma = lambda: None
    with pytest.raises(RuntimeError, match="injected source failure"):
        transport.begin_read([_binding(0, 8, 0)])
    assert closed == [0]
    assert not transport._cache_read_tickets
    assert not transport.ring.reads
    assert not transport._pending


@pytest.mark.parametrize("model_windows", [False, True])
def test_cache_reads_are_submitted_before_dma_and_split_at_file_windows(monkeypatch, model_windows):
    import torch

    calls = []
    payload = b"0123456789"
    arena = bytearray(8)

    class Ticket:
        def __init__(self, offset, size):
            self.offset, self.size = offset, size

        def copy_to_arena_async(self, target, offset, stream):
            calls.append(("dma", self.offset, self.size))
            target[offset:offset + self.size] = payload[self.offset:self.offset + self.size]

    def request(_source, offset, size):
        calls.append(("read", offset, size))
        return Ticket(offset, size)

    class Event:
        def __init__(self, **_kwargs):
            pass

        def record(self, stream):
            calls.append(("event", stream))

    monkeypatch.setattr(torch.cuda, "Event", Event)
    transport = _transport(payload, window_bytes=4, slots=2)
    transport.cache = SimpleNamespace(model_windows=model_windows, slot_bytes=4, request=request)
    transport._source = lambda path: path
    transport.reap_cache_dma = lambda: None
    transport._cache_dma_tickets = []
    transport.upload_batch([_binding(1, 8, 0)], arena, "upload")
    expected = [(1, 3), (4, 4), (8, 1)] if model_windows else [(1, 8)]
    assert calls == [("read", *item) for item in expected] + [("dma", *item) for item in expected] + [("event", "upload")]
    assert arena == payload[1:9]
    assert len(transport._cache_dma_tickets) == len(expected)
    assert not transport.ring.reads
