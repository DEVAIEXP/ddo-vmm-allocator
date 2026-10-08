"""CPU-only optional pressure-snapshot ABI contracts."""
from types import SimpleNamespace
import ctypes as c
import os

import pytest

from ddo_vmm_allocator.native import NativeVmmError
from ddo_vmm_allocator.source_cache import _PRESSURE_FIELDS, _pressure_snapshot


class Snapshot:
    def __init__(self, version=1, status=0):
        self.version = version
        self.status = status

    def __call__(self, handle, values, count):
        assert count == 23 == len(_PRESSURE_FIELDS)
        values[0], values[9], values[11] = self.version, 2, 1024
        return self.status


def test_old_binary_reports_unavailable():
    assert _pressure_snapshot(SimpleNamespace(), None, lambda _: None) == {}


def test_fixed_schema_values():
    statuses = []
    dll = SimpleNamespace(ddo_vmm_source_cache_pressure_snapshot_v1=Snapshot())
    values = _pressure_snapshot(dll, None, statuses.append)
    assert statuses == [0]
    assert len(values) == 23
    assert values["pressure_observation_last_reason"] == 2
    assert values["pressure_observation_last_entry_commit_bytes"] == 1024


def test_unknown_schema_refused():
    dll = SimpleNamespace(ddo_vmm_source_cache_pressure_snapshot_v1=Snapshot(version=99))
    with pytest.raises(NativeVmmError, match="schema"):
        _pressure_snapshot(dll, None, lambda _: None)


def test_native_failure_not_suppressed():
    def check(status):
        if status:
            raise NativeVmmError("snapshot failed")
    dll = SimpleNamespace(ddo_vmm_source_cache_pressure_snapshot_v1=Snapshot(status=1))
    with pytest.raises(NativeVmmError, match="snapshot failed"):
        _pressure_snapshot(dll, None, check)


def test_actual_export_rejects_null_without_cuda():
    path = os.environ.get("DDO_VMM_LIBRARY_PATH")
    if not path:
        pytest.skip("select a candidate DLL explicitly for the CPU ABI check")
    dll = c.CDLL(path)
    snapshot = getattr(dll, "ddo_vmm_source_cache_pressure_snapshot_v1", None)
    if snapshot is None:
        pytest.skip("older binary omits optional pressure observations")
    snapshot.argtypes = [c.c_void_p, c.POINTER(c.c_uint64), c.c_size_t]
    snapshot.restype = c.c_int
    values = (c.c_uint64 * 23)(*([99] * 23))
    assert snapshot(None, values, 23) == 1
    assert list(values) == [99] * 23
    import torch
    assert not torch.cuda.is_initialized()
