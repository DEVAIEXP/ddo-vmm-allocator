"""CPU ownership contracts for full-unit native GPU look-ahead."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from ddo_vmm_allocator.model_arena_adapter import (
    ModelArenaError,
    ModelArenaModuleAdapter,
    ModelArenaWeightStore,
    _ModuleBinding,
    _NativeShardTransport,
)
from ddo_vmm_allocator.source import NativeSourceError


def test_retained_source_registration_refusal_uses_native_ring_without_read():
    transport = _NativeShardTransport.__new__(_NativeShardTransport)
    transport.ring = object()
    transport._acquire_slot = lambda: 7
    calls = []

    def refuse(*args):
        raise NativeSourceError(8, "registration refused")

    ticket = SimpleNamespace(copy_to_arena_async=refuse, copy_via_ring_async=lambda *args: calls.append(args))
    window = SimpleNamespace(arena_offset=64)
    transport._copy_cached_source(ticket, window, "arena", "stream")
    assert calls == [(transport.ring, 7, "arena", 64, "stream")]


def test_retained_source_cuda_error_does_not_silently_fallback():
    transport = _NativeShardTransport.__new__(_NativeShardTransport)

    def fail(*args):
        raise NativeSourceError(3, "fatal CUDA error")

    with pytest.raises(NativeSourceError, match="fatal CUDA"):
        transport._copy_cached_source(SimpleNamespace(copy_to_arena_async=fail), SimpleNamespace(arena_offset=0), None, None)


def make_adapter(monkeypatch, *, capacity_denied=False):
    trace = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            trace.append("compute_record")

        def query(self):
            return False

        def synchronize(self):
            trace.append("compute_drain")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "compute")
    stream = SimpleNamespace(synchronize=lambda: trace.append("upload_drain"))
    store = SimpleNamespace(
        _native_shard_transport=object(),
        _upload_stream=stream,
        _device_ordinal=0,
        _debug_faults=False,
        _runtime_stats=defaultdict(int),
        wait_for_compute=lambda event: trace.append("wait:" + event),
        unpin=lambda binding: trace.append("unpin:" + binding.label),
    )

    def fault(bindings):
        name = bindings[0].label
        trace.append("fault_dma:" + name)
        if name == "next" and capacity_denied:
            raise ModelArenaError("resident capacity")
        for binding in bindings:
            binding.ready_event = name
        return [name]

    store.fault_and_copy_group_if_needed = fault
    adapter = ModelArenaModuleAdapter(store)
    states = []
    for name in ("current", "next"):
        module = torch.nn.Identity()
        module.register_forward_hook(lambda m, args, out: trace.append("compute_enqueue"))
        binding = SimpleNamespace(label=name, stable=SimpleNamespace(device=SimpleNamespace(type="cuda")))
        states.append(_ModuleBinding(name, module, [binding], {}))
    states[0].prefetch_next = states[1]
    adapter._execution_units = states
    for state in states:
        adapter._register_execution_unit(state)
    return adapter, states, trace


def test_gpu_prefetch_after_compute_and_consumed_without_second_fault(monkeypatch):
    adapter, states, trace = make_adapter(monkeypatch)
    states[0].module(torch.ones(1))
    assert trace == ["fault_dma:current", "wait:current", "compute_enqueue", "compute_record", "fault_dma:next"]
    assert states[1].prefetched
    assert not any(item.startswith("unpin") for item in trace)
    states[1].module(torch.ones(1))
    assert trace.count("fault_dma:next") == 1
    assert "wait:next" in trace
    assert not states[1].prefetched
    assert adapter.store._runtime_stats["gpu_prefetch_hits"] == 1


def test_gpu_prefetch_capacity_denial_never_waits_or_creates_fallback(monkeypatch):
    adapter, states, trace = make_adapter(monkeypatch, capacity_denied=True)
    states[0].module(torch.ones(1))
    assert not states[1].prefetched
    assert not any(item.endswith("drain") for item in trace)
    assert adapter.store._runtime_stats["gpu_prefetch_capacity_skips"] == 1
    assert len(adapter._pending_unpins) == 1  # Current compute remains protected.


def test_abandoned_gpu_prefetch_drains_upload_before_unpin(monkeypatch):
    adapter, states, trace = make_adapter(monkeypatch)
    states[0].module(torch.ones(1))
    adapter._cancel_gpu_prefetches()
    assert trace[-2:] == ["upload_drain", "unpin:next"]
    assert not states[1].prefetched
    adapter._cancel_gpu_prefetches()
    assert trace.count("unpin:next") == 1
    assert adapter.store._runtime_stats["gpu_prefetch_cancelled_units"] == 1


def test_failed_upload_drains_destination_before_invalidating_and_unpinning():
    trace = []
    store = ModelArenaWeightStore.__new__(ModelArenaWeightStore)
    store._staging_pool = None
    store._native_shard_transport = object()
    store._runtime_stats = defaultdict(int)
    store.native = SimpleNamespace(fault_range=lambda offset, size: 9)
    store._upload_stream = SimpleNamespace(synchronize=lambda: trace.append("drain"))
    bindings = [SimpleNamespace(file_slice=True, offset=0, nbytes=4, signature=3, ready_event="old")]
    store.unpin = lambda binding: trace.append(("unpin", binding.signature, binding.ready_event))

    def fail(bindings, signatures):
        raise OSError("injected later-batch read failure")

    store._upload_faulted_group = fail
    with pytest.raises(OSError, match="later-batch"):
        store.fault_and_copy_group_if_needed(bindings)
    assert trace == ["drain", ("unpin", None, None)]


def test_failed_upload_drain_retains_pins():
    store = ModelArenaWeightStore.__new__(ModelArenaWeightStore)
    store._staging_pool = None
    store._native_shard_transport = object()
    store._runtime_stats = defaultdict(int)
    store.native = SimpleNamespace(fault_range=lambda offset, size: 9)
    unpins = []
    store.unpin = unpins.append

    def fail(*args):
        raise RuntimeError("injected fatal stream failure")

    store._upload_faulted_group = fail
    store._upload_stream = SimpleNamespace(synchronize=fail)
    binding = SimpleNamespace(file_slice=True, offset=0, nbytes=4, signature=None)
    with pytest.raises(RuntimeError, match="fatal stream"):
        store.fault_and_copy_group_if_needed([binding])
    assert unpins == []
