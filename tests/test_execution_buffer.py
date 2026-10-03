"""CPU-only ownership/dispatch contracts; no CUDA availability probe."""

from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from ddo_vmm_allocator.model_arena_adapter import NativeExecutionBuffer, _TensorBinding


def make_buffer(monkeypatch):
    from ddo_vmm_allocator import model_arena_adapter as module

    trace = []
    native = SimpleNamespace(
        info=SimpleNamespace(address=4096),
        wait_compute=lambda stream: trace.append(("reuse_wait", stream)),
        fault_range=lambda offset, size: trace.append(("materialize", offset, size)),
        record_compute=lambda stream: trace.append(("fence", stream)),
        drain_compute=lambda: trace.append(("drain",)),
        close=lambda: trace.append(("close",)),
    )
    captured = []

    def upload(bindings, arena, stream):
        captured.extend(bindings)
        trace.append(("upload", arena, stream))
        return "ready"

    store = SimpleNamespace(
        page_bytes=512, _device_ordinal=0, _backing="device",
        _library=SimpleNamespace(create_model_arena=lambda *args, **kwargs: native),
        _upload_stream="upload_stream", _runtime_stats=defaultdict(int),
        _native_shard_transport=SimpleNamespace(upload_batch=upload),
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(module, "cuda_uint8_tensor_from_address",
                        lambda address, size, **kwargs: (torch.zeros(size, dtype=torch.uint8), native))
    source = torch.zeros(3, dtype=torch.float32)
    binding = _TensorBinding(None, "weight", source, object(), source, 9000, 12)
    return NativeExecutionBuffer(store, 12), binding, trace, captured


def test_buffer_reuses_native_mapping_and_fences_compute(monkeypatch):
    buffer, binding, trace, captured = make_buffer(monkeypatch)
    views, ready = buffer.upload([binding])
    assert ready == "ready" and views[0][0] is binding
    assert captured[0].offset == 0 and captured[0].file_slice is binding.file_slice
    assert binding.offset == 9000  # Never mutate immutable model offsets.
    buffer.finish_compute("compute_stream")
    buffer.upload([binding])
    assert [item[0] for item in trace] == [
        "reuse_wait", "materialize", "upload", "fence", "fence",
        "reuse_wait", "upload", "fence",
    ]
    buffer.close()
    assert trace[-2:] == [("drain",), ("close",)]


def test_partial_upload_failure_still_records_native_fence(monkeypatch):
    buffer, binding, trace, _ = make_buffer(monkeypatch)

    def fail(*args):
        raise RuntimeError("partial upload")

    buffer.store._native_shard_transport.upload_batch = fail
    with pytest.raises(RuntimeError, match="partial upload"):
        buffer.upload([binding])
    assert trace[-1] == ("fence", "upload_stream")


def test_failed_close_drain_retains_owner(monkeypatch):
    buffer, _, trace, _ = make_buffer(monkeypatch)

    def fail():
        raise RuntimeError("compute still live")

    buffer.native.drain_compute = fail
    with pytest.raises(RuntimeError, match="compute still live"):
        buffer.close()
    assert ("close",) not in trace


def test_failed_fence_never_releases_backing(monkeypatch):
    buffer, _, trace, _ = make_buffer(monkeypatch)

    def fail(stream):
        raise RuntimeError("event record failed")

    buffer.native.record_compute = fail
    with pytest.raises(RuntimeError, match="event record failed"):
        buffer.finish_compute("compute_stream")
    with pytest.raises(RuntimeError, match="backing retained"):
        buffer.close()
    assert ("close",) not in trace


def test_native_pool_leases_rotate_and_views_are_cached(monkeypatch):
    from ddo_vmm_allocator import model_arena_adapter as module

    _, binding, trace, _ = make_buffer(monkeypatch)
    native_arenas = [SimpleNamespace(info=SimpleNamespace(address=4096 + i * 1024)) for i in range(2)]

    class Pool:
        capacity_bytes, slots, resident_bytes = 512, 2, 1024

        def __init__(self, *args, **kwargs):
            self.leased, self.next = set(), 0

        def acquire(self, stream):
            slot = self.next
            assert slot not in self.leased
            self.leased.add(slot)
            self.next = 1 - slot
            trace.append(("acquire", slot))
            return slot, native_arenas[slot]

        def release(self, slot, stream):
            self.leased.remove(slot)
            trace.append(("release", slot, stream))

        def close(self):
            assert not self.leased

    monkeypatch.setattr(module, "NativeExecutionPool", Pool)
    store = SimpleNamespace(_library=None, native=None, _device_ordinal=0,
        _upload_stream="upload", _backing="device", _runtime_stats=defaultdict(int),
        _native_shard_transport=SimpleNamespace(upload_batch=lambda *args: "ready"))
    pool = module.NativeExecutionBuffers(store, 12)
    first = pool.upload([binding])
    second = pool.upload([binding])
    assert (first.slot, second.slot) == (0, 1)
    pool.finish_compute(first, "compute")
    pool.cancel(second)
    third = pool.upload([binding])
    assert third.slot == 0 and third.views is first.views
    pool.finish_compute(third, "compute")
    pool.close()
    assert trace[-1] == ("release", 0, "compute")


def test_immutable_transport_batch_is_prepared_once_without_python_windows():
    from ddo_vmm_allocator.model_arena_adapter import _NativeShardTransport

    trace = []
    transport = _NativeShardTransport.__new__(_NativeShardTransport)
    batch = SimpleNamespace(submit=lambda arena, stream: trace.append(("submit", arena, stream)))

    def create(spans, ring, **kwargs):
        trace.append(("create", spans, kwargs))
        return batch

    transport.cache = SimpleNamespace(persistent_buffer=True, create_batch=create)
    transport._batches = {}
    transport._source = lambda path: "immutable-source"
    transport.ring, transport.slots, transport.window_size_bytes = object(), 3, 64
    binding = SimpleNamespace(label="unit.weight", offset=12, nbytes=160,
                              file_slice=SimpleNamespace(path="weights", offset_bytes=5))
    assert transport.upload_batch([binding], "arena-a", "stream") is batch
    assert transport.upload_batch([binding], "arena-b", "stream") is batch
    assert [item[0] for item in trace] == ["create", "submit", "submit"]
    assert trace[0][1] == [("immutable-source", 5, 160, 12)]
    assert transport._batch_timings["batch_transport_total_calls"] == 2
    assert transport._batch_timings["batch_transport_create_calls"] == 1
    assert transport._batch_timings["batch_transport_submit_calls"] == 2


def test_native_batch_retains_destination_until_failed_close_can_be_retried():
    import ctypes
    from threading import RLock
    from ddo_vmm_allocator.source_cache import NativeSourceBatch

    def check(status):
        if status:
            raise RuntimeError("completion not proven")

    batch = NativeSourceBatch.__new__(NativeSourceBatch)
    batch._handle = ctypes.c_void_p(123)
    batch._destinations = []
    batch._cache = SimpleNamespace(_lock=RLock(), _check=check,
        _dll=SimpleNamespace(ddo_vmm_source_batch_submit=lambda *args: 3,
                             ddo_vmm_source_batch_close=lambda *args: 3))
    arena = SimpleNamespace(_require_handle=lambda: ctypes.c_void_p(456))
    with pytest.raises(RuntimeError, match="completion not proven"):
        batch.submit(arena, None)
    assert batch._destinations == [arena]
    with pytest.raises(RuntimeError, match="completion not proven"):
        batch.close()
    assert batch._handle.value == 123 and batch._destinations == [arena]

    def close(owner):
        owner._obj.value = None
        return 0

    batch._cache._dll.ddo_vmm_source_batch_close = close
    batch.close()
    assert not batch._handle.value and not batch._destinations


@pytest.mark.parametrize("fail", [False, True])
def test_batch_bridge_scopes_reconcile_host_time_without_cuda(monkeypatch, fail):
    import ctypes
    from threading import RLock
    from ddo_vmm_allocator import source_cache as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])

    class Stream:
        @property
        def cuda_stream(self):
            clock[0] += 2
            return 123

    def submit(*args):
        clock[0] += 7
        return 3 if fail else 0

    def check(status):
        clock[0] += 3
        if status:
            raise RuntimeError("partial native failure")

    batch = module.NativeSourceBatch.__new__(module.NativeSourceBatch)
    batch._handle, batch._destinations = ctypes.c_void_p(123), []
    batch._cache = SimpleNamespace(_lock=RLock(), _check=check,
        _dll=SimpleNamespace(ddo_vmm_source_batch_submit=submit))
    arena = SimpleNamespace(_require_handle=lambda: ctypes.c_void_p(456))
    if fail:
        with pytest.raises(RuntimeError, match="partial native failure"):
            batch.submit(arena, Stream())
    else:
        batch.submit(arena, Stream())
    metrics = batch._cache._batch_timings
    assert metrics["batch_bridge_submit_total_seconds"] == 12
    assert metrics["batch_bridge_submit_lock_wait_seconds"] == 0
    assert metrics["batch_bridge_submit_prepare_seconds"] == 2
    assert metrics["batch_bridge_submit_ffi_seconds"] == 7
    assert metrics["batch_bridge_submit_result_seconds"] == 3
    assert metrics["batch_bridge_submit_ffi_calls"] == 1
    assert batch._destinations == [arena]


def test_transport_scopes_separate_one_time_creation_from_repeated_submit(monkeypatch):
    from ddo_vmm_allocator import model_arena_adapter as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])

    def submit(*args):
        clock[0] += 7

    batch = SimpleNamespace(submit=submit)

    def create(*args, **kwargs):
        clock[0] += 5
        return batch

    transport = module._NativeShardTransport.__new__(module._NativeShardTransport)
    transport.cache = SimpleNamespace(persistent_buffer=True, create_batch=create)
    transport._batches = {}
    transport._source = lambda path: "source"
    transport.ring, transport.slots, transport.window_size_bytes = object(), 2, 64
    binding = SimpleNamespace(label="weight", offset=0, nbytes=16,
        file_slice=SimpleNamespace(path="file", offset_bytes=0))
    for _ in range(2):
        transport.upload_batch([binding], "arena", "stream")
    metrics = transport._batch_timings
    assert metrics["batch_transport_total_seconds"] == 19
    assert metrics["batch_transport_prepare_seconds"] == 5
    assert metrics["batch_transport_create_seconds"] == 5
    assert metrics["batch_transport_submit_seconds"] == 14
    assert metrics["batch_transport_create_calls"] == 1


@pytest.mark.parametrize("fail", [False, True])
def test_upload_stream_partitions_entry_body_exit_without_extra_operations(monkeypatch, fail):
    from ddo_vmm_allocator import model_arena_adapter as module

    clock, calls, stats = [0.0], [], {}
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])

    class Context:
        def __enter__(self):
            calls.append("enter")
            clock[0] += 3

        def __exit__(self, *args):
            calls.append("exit")
            clock[0] += 2

    monkeypatch.setattr(torch.cuda, "stream", lambda stream: Context())

    def work():
        with module._measure_upload_stream("stream", stats):
            calls.append("body")
            clock[0] += 7
            if fail:
                raise RuntimeError("partial upload")

    if fail:
        with pytest.raises(RuntimeError, match="partial upload"):
            work()
    else:
        work()
    assert calls == ["enter", "body", "exit"]
    assert stats == {"upload_stream_enter_seconds": 3, "upload_stream_body_seconds": 7,
                     "upload_stream_exit_seconds": 2, "upload_stream_scope_calls": 1}
