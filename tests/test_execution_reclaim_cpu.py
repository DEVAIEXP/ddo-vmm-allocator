"""Execution-pool ABI contracts without loading CUDA."""
import ctypes
from types import SimpleNamespace
import pytest
from ddo_vmm_allocator.native import NativeExecutionPool, NativeVmmError


class Function:
    def __init__(self, call):
        self.call = call

    def __call__(self, *args):
        return self.call(*args)


def make_pool(**functions):
    pool = NativeExecutionPool.__new__(NativeExecutionPool)
    pool._handle, pool.slots = ctypes.c_void_p(123), 2
    pool._owner = SimpleNamespace(
        _library=SimpleNamespace(**{key: Function(value) for key, value in functions.items()}),
        _last_error=lambda: "original native refusal")
    return pool


def test_old_artifact_refuses_enablement_without_fabricating_stats():
    pool = make_pool()
    assert pool.reclaim_stats() == {}
    with pytest.raises(NativeVmmError, match="compatible allocator"):
        pool.configure_reclaim(enabled=True)


def test_configuration_preserves_margin_and_native_failure():
    calls = []
    pool = make_pool(ddo_vmm_execution_pool_configure_reclaim=
                     lambda handle, enabled, margin: calls.append((enabled, margin)) or 0)
    pool.configure_reclaim(enabled=True)
    pool.configure_reclaim(enabled=False, margin_bytes=0)
    assert calls == [(1, 256 * 2**20), (0, 0)]
    pool = make_pool(ddo_vmm_execution_pool_configure_reclaim=lambda *args: 2)
    with pytest.raises(NativeVmmError, match="original native refusal"):
        pool.configure_reclaim(enabled=True)


@pytest.mark.parametrize("enabled,margin", [(1, 0), (True, -1), (True, 2**64), (True, 0.5)])
def test_invalid_configuration_refuses_before_native_call(enabled, margin):
    with pytest.raises(ValueError):
        make_pool().configure_reclaim(enabled=enabled, margin_bytes=margin)


def test_reclaim_returns_proven_bytes_and_preserves_refusal():
    def reclaim(handle, index, released):
        assert handle.value == 123 and index == 1
        released._obj.value = 4 * 2**20
        return 0
    pool = make_pool(ddo_vmm_execution_pool_reclaim_idle=reclaim)
    assert pool.reclaim_idle(1) == 4 * 2**20
    with pytest.raises(ValueError):
        pool.reclaim_idle(2)
    pool = make_pool(ddo_vmm_execution_pool_reclaim_idle=lambda *args: 5)
    with pytest.raises(NativeVmmError, match="original native refusal"):
        pool.reclaim_idle(0)


def test_stats_copy_and_schema_validation():
    def stats(handle, values, count):
        assert handle.value == 123 and count == 16
        for index, value in enumerate([1, 16, 1, 256, 9, 4, 1, 3, 12, 3, 12, 0, 0, 0, 0, 0]):
            values[index] = value
        return 0
    pool = make_pool(ddo_vmm_execution_pool_reclaim_stats=stats)
    report = pool.reclaim_stats()
    assert report["reclaims"] == 3 and report["released_bytes"] == 12
    assert report["recreated_bytes"] == 12 and report["unavailable"] == 0
    with pytest.raises(NativeVmmError, match="schema"):
        make_pool(ddo_vmm_execution_pool_reclaim_stats=lambda *args: 0).reclaim_stats()
