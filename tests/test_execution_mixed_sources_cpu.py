"""Mixed-source dispatch and failure fences; CUDA calls are mocked, not executed."""

from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from ddo_vmm_allocator import model_arena_adapter as adapter


@pytest.fixture
def harness(monkeypatch):
    trace = []
    active = []

    @contextmanager
    def stream(value):
        active.append(value)
        try:
            yield
        finally:
            active.pop()

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, value):
            assert active[-1] == value
            trace.append("ready")

    def upload(bindings, arena, value):
        assert all(b.file_slice is not None for b in bindings)
        for binding in bindings:
            binding.stable.fill_(7)
        trace.append("files")
        return "file-ready"

    native = SimpleNamespace(
        info=SimpleNamespace(address=4096),
        wait_compute=lambda value: None,
        fault_range=lambda *args: None,
        record_compute=lambda value: trace.append(("fence", value)),
    )

    class Pool:
        capacity_bytes, slots, resident_bytes = 1024, 2, 2048

        def __init__(self, *args, **kwargs):
            pass

        def acquire(self, value):
            trace.append("acquire")
            return 0, native

        def release(self, slot, value):
            trace.append(("release", slot, value))

    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(adapter, "NativeExecutionPool", Pool)
    monkeypatch.setattr(adapter, "cuda_uint8_tensor_from_address",
                        lambda address, size, **kwargs: (torch.zeros(size, dtype=torch.uint8), native))
    store = SimpleNamespace(
        page_bytes=512, _device_ordinal=0, _backing="device", native=native,
        _library=SimpleNamespace(create_model_arena=lambda *args, **kwargs: native),
        _upload_stream="upload", _runtime_stats=defaultdict(int),
        _native_shard_transport=SimpleNamespace(upload_batch=upload),
    )
    return store, native, trace, Event


def binding(file_backed):
    source = torch.tensor([1.0, 2.0, 3.0])
    return adapter._TensorBinding(None, "weight", source,
        object() if file_backed else None, source, 9000, 12)


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("files", [False, True])
def test_mixed_and_host_only_units_keep_sources_offsets_and_completion(harness, pooled, files):
    store, _, trace, event_type = harness
    buffer = (adapter.NativeExecutionBuffers if pooled else adapter.NativeExecutionBuffer)(store, 1024)
    host = binding(False)
    bindings = [binding(True), host] if files else [host]
    result = buffer.upload(bindings)
    views, ready = (result.views, result.ready) if pooled else result
    assert isinstance(ready, event_type)
    assert trace.count("files") == int(files)
    assert trace.index("ready") > trace.index("files") if files else "ready" in trace
    torch.testing.assert_close(views[-1][1], host.source)
    if files:
        torch.testing.assert_close(views[0][1], torch.full((3,), 7.0))
    assert all(b.offset == 9000 for b in bindings)
    assert views[-1][0] is host  # The lease retains the CPU-source binding.
    assert store._runtime_stats["upload_bytes"] == 12 * len(bindings)
    if pooled:
        buffer.finish_compute(result, "compute")
        assert trace[-1] == ("release", 0, "compute")
    else:
        buffer.finish_compute("compute")
        assert trace[-1] == ("fence", "compute")
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("pooled", [False, True])
def test_host_copy_failure_after_file_submission_fences_upload(harness, monkeypatch, pooled):
    store, _, trace, _ = harness
    buffer = (adapter.NativeExecutionBuffers if pooled else adapter.NativeExecutionBuffer)(store, 1024)

    def fail(*args, **kwargs):
        raise RuntimeError("host copy failed")

    monkeypatch.setattr(torch.Tensor, "copy_", fail)
    with pytest.raises(RuntimeError, match="host copy failed"):
        buffer.upload([binding(True), binding(False)])
    assert "files" in trace and "ready" not in trace
    expected = ("release", 0, "upload") if pooled else ("fence", "upload")
    assert trace[-1] == expected


@pytest.mark.parametrize("pooled", [False, True])
def test_final_event_failure_keeps_partial_upload_fence(harness, monkeypatch, pooled):
    store, _, trace, event_type = harness
    buffer = (adapter.NativeExecutionBuffers if pooled else adapter.NativeExecutionBuffer)(store, 1024)

    def fail(*args):
        raise RuntimeError("ready record failed")

    monkeypatch.setattr(event_type, "record", fail)
    with pytest.raises(RuntimeError, match="ready record failed"):
        buffer.upload([binding(True), binding(False)])
    assert "files" in trace
    expected = ("release", 0, "upload") if pooled else ("fence", "upload")
    assert trace[-1] == expected


def test_file_only_fast_path_keeps_original_completion(harness):
    store, _, trace, _ = harness
    buffer = adapter.NativeExecutionBuffers(store, 1024)
    lease = buffer.upload([binding(True)])
    assert lease.ready == "file-ready"
    assert trace == ["acquire", "files"]
