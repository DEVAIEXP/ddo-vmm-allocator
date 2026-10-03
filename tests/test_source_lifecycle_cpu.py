"""Failure-retention contracts with fake ABI calls; never load CUDA or a DLL."""
import ctypes as c
import os
from threading import RLock
from types import SimpleNamespace

import pytest

from ddo_vmm_allocator.source import NativeSourceError
from ddo_vmm_allocator.source_cache import NativeSourceCache, NativeSourceBatch, NativeSourceTicket, _close_context_owner
from ddo_vmm_allocator.model_arena_adapter import _NativeShardTransport


def fake_cache():
    cache = NativeSourceCache.__new__(NativeSourceCache)
    cache._lock = RLock()
    cache._handle = c.c_void_p(101)
    cache._dll = SimpleNamespace(ddo_vmm_source_cache_last_error=lambda: b"completion unproven")
    return cache


@pytest.mark.parametrize("kind", ["cache", "ticket", "batch"])
@pytest.mark.parametrize("status", [7, 4])
def test_failed_close_preserves_handle_and_owners_until_success(kind, status):
    cache = fake_cache()
    arena, ring = object(), object()
    if kind == "cache":
        owner = cache
    elif kind == "ticket":
        owner = NativeSourceTicket(cache)
        owner._handle = c.c_void_p(102)
        owner._arena, owner._ring = arena, ring
    else:
        owner = NativeSourceBatch.__new__(NativeSourceBatch)
        owner._cache, owner._ring = cache, ring
        owner._handle = c.c_void_p(103)
        owner._destinations = [arena]
    original = owner._handle.value
    state = {"status": status, "calls": 0}

    def close(pointer):
        state["calls"] += 1
        handle = c.cast(pointer, c.POINTER(c.c_void_p))
        if state["status"] == 0:
            handle[0] = None
        return state["status"]

    setattr(cache._dll, f"ddo_vmm_source_{kind}_close", close)
    with pytest.raises(NativeSourceError, match="completion unproven") as error:
        owner.close()
    assert error.value.status == status
    assert owner._handle.value == original
    if kind == "ticket":
        assert owner._arena is arena and owner._ring is ring
    if kind == "batch":
        assert owner._destinations == [arena] and owner._ring is ring
    state["status"] = 0
    owner.close()
    owner.close()
    assert owner._handle.value is None and state["calls"] == 3
    if kind == "ticket":
        assert owner._arena is None and owner._ring is None
    if kind == "batch":
        assert owner._destinations == []


def test_failed_batch_submission_retains_destination():
    cache = fake_cache()
    batch = NativeSourceBatch.__new__(NativeSourceBatch)
    batch._cache, batch._ring = cache, object()
    batch._handle, batch._destinations = c.c_void_p(104), []
    arena = SimpleNamespace(_require_handle=lambda: c.c_void_p(105))
    cache._dll.ddo_vmm_source_batch_submit = lambda *_args: 4
    with pytest.raises(NativeSourceError, match="completion unproven"):
        batch.submit(arena, None)
    assert batch._destinations == [arena]
    assert batch._handle.value == 104


@pytest.mark.parametrize("kind", ["cache", "ticket"])
def test_context_cleanup_failure_keeps_first_error_and_handle(kind):
    cache = fake_cache()
    owner = cache if kind == "cache" else NativeSourceTicket(cache)
    owner._handle = c.c_void_p(106)
    setattr(cache._dll, f"ddo_vmm_source_{kind}_close", lambda *_args: 4)
    original = RuntimeError("original upload failed")
    with pytest.raises(RuntimeError, match="original upload failed") as caught:
        with owner:
            raise original
    assert caught.value is original
    assert owner._handle.value == 106
    assert any("completion unproven" in note for note in original.__notes__)
    # Explicit close still reports the failure; successful body cannot hide it.
    with pytest.raises(NativeSourceError, match="completion unproven"):
        with owner:
            pass


@pytest.mark.parametrize("failed_stage", ["ticket", "batch", "cache", "source"])
def test_transport_close_stops_before_downstream_owners(failed_stage):
    transport = _NativeShardTransport.__new__(_NativeShardTransport)
    calls, failure = [], {"enabled": True}

    def resource(name):
        def close():
            calls.append(name)
            if failure["enabled"] and name == failed_stage:
                raise NativeSourceError(4, name + " completion unproven")
        return SimpleNamespace(close=close)

    ticket, batch, cache, source, ring = [resource(n) for n in
                                         ("ticket", "batch", "cache", "source", "ring")]
    transport._cache_read_tickets, transport._pending = [], {}
    transport._cache_dma_tickets = [ticket]
    transport._batches = {"unit": batch}
    transport.cache, transport._cache_sources, transport.ring = cache, {"file": source}, ring
    order = ["ticket", "batch", "cache", "source", "ring"]
    with pytest.raises(NativeSourceError, match=failed_stage):
        transport.close()
    assert calls == order[:order.index(failed_stage) + 1]
    if failed_stage in ("ticket", "batch"):
        assert transport.cache is cache
    assert transport._cache_sources == {"file": source}
    assert transport.ring is ring
    failure["enabled"] = False
    transport.close()
    assert calls[-1] == "ring"
    assert transport.cache is None
    assert transport._cache_sources == {} and transport._batches == {}
    assert transport._cache_dma_tickets == []


@pytest.mark.parametrize("optional_export", [False, True])
def test_owned_event_snapshot_is_optional_and_detached(optional_export):
    cache = fake_cache()
    cache._stats_version = 8
    cache._dll.ddo_vmm_source_cache_snapshot = lambda *_args: 0
    if optional_export:
        def snapshot(_cache, transfers, pooled, batches):
            for output, value in zip((transfers, pooled, batches), (2, 3, 4)):
                c.cast(output, c.POINTER(c.c_size_t))[0] = value
            return 0
        cache._dll.ddo_vmm_source_owned_events_v1 = snapshot
    result = cache.snapshot()
    if optional_export:
        assert [result[k] for k in ("transfer_owned_events", "pooled_events", "batch_completion_events")] == [2, 3, 4]
    else:
        assert "transfer_owned_events" not in result
    cache._handle.value = None
    cache._dll.ddo_vmm_source_cache_close = lambda *_args: 0


def test_actual_lifecycle_exports_reject_null_without_cuda():
    path = os.environ.get("DDO_VMM_LIBRARY_PATH")
    assert path, "select isolated lifecycle build explicitly"
    dll = c.CDLL(path)
    fn = getattr(dll, "ddo_vmm_source_owned_events_v1", None)
    if fn is None:
        pytest.skip("older protected DLL omits optional ownership export")
    fn.argtypes = [c.c_void_p] + [c.POINTER(c.c_size_t)] * 3
    fn.restype = c.c_int
    values = [c.c_size_t(99) for _ in range(3)]
    assert fn(None, *(c.byref(v) for v in values)) == 1
    assert [v.value for v in values] == [99] * 3
    inject = getattr(dll, "ddo_vmm_source_test_failures", None)
    if inject is not None:
        inject.argtypes, inject.restype = [c.c_void_p, c.c_int64, c.c_int], c.c_int
        assert inject(None, 1, 0) == 1
    import torch
    assert not torch.cuda.is_initialized()


def test_first_error_is_preserved_without_python311_add_note():
    class OlderError(RuntimeError):
        add_note = None
    def close():
        raise NativeSourceError(7, "close refused")
    original = OlderError("first error")
    _close_context_owner(SimpleNamespace(close=close), original)
    assert "close refused" in original.__notes__[0]
