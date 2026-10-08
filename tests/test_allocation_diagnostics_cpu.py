"""CPU-only optional allocation diagnostic ABI contracts."""
from types import SimpleNamespace
import ctypes
import os
import pytest
from ddo_vmm_allocator.native import NativeVmmLibrary, NativeVmmError


class Snapshot:
    def __init__(self, version=1, status=0):
        self.version, self.status = version, status

    def __call__(self, values, count):
        assert count == 26
        values[0], values[1], values[2], values[3], values[4] = self.version, 1, 5, 250000000, 2
        values[23], values[24], values[25] = 8, 3, 4096
        return self.status


def native_with(**functions):
    obj = NativeVmmLibrary.__new__(NativeVmmLibrary)
    obj._library = SimpleNamespace(**functions)
    return obj


def test_old_binary_absent_is_unavailable():
    native = native_with()
    assert native.allocation_diagnostics_snapshot() == {}
    with pytest.raises(NativeVmmError, match="diagnostic binary"):
        native.set_allocation_diagnostics(True)


def test_fixed_schema_and_units():
    result = native_with(ddo_vmm_allocation_diagnostics_snapshot_v1=Snapshot()).allocation_diagnostics_snapshot()
    assert result["create_calls"] == 5 and result["create_failed_calls"] == 2
    assert result["create_host_seconds"] == .25
    assert result["pool_new_backings"] == 3 and result["pool_reuse_calls"] == 8
    assert result["pool_observed_peak_mapped_bytes"] == 4096


@pytest.mark.parametrize("version,status", [(99, 0), (1, 1)])
def test_malformed_snapshot_refused(version, status):
    with pytest.raises(NativeVmmError, match="snapshot/schema"):
        native_with(ddo_vmm_allocation_diagnostics_snapshot_v1=Snapshot(version, status)).allocation_diagnostics_snapshot()


def test_unsupported_snapshot_is_unavailable():
    assert native_with(ddo_vmm_allocation_diagnostics_snapshot_v1=Snapshot(status=2)).allocation_diagnostics_snapshot() == {}


def test_busy_scope_refused():
    class Busy:
        def __call__(self, enabled):
            assert enabled == 1
            return 1
    with pytest.raises(NativeVmmError, match="already owned"):
        native_with(ddo_vmm_allocation_diagnostics_enable_v1=Busy()).set_allocation_diagnostics(True)


class ExtendedSnapshot:
    def __call__(self, values, count):
        assert count == 36
        values[0], values[1] = 2, 1
        values[26], values[27], values[28] = 2, 3000000000, 1
        values[29], values[30] = 9, 4000000000
        values[32], values[33], values[34], values[35] = 6, 2, 0, 1
        return 0


def test_v2_preferred_and_nested_timers_decoded():
    result = native_with(ddo_vmm_allocation_diagnostics_snapshot_v2=ExtendedSnapshot(),
                         ddo_vmm_allocation_diagnostics_snapshot_v1=Snapshot()).allocation_diagnostics_snapshot()
    assert result['schema_version'] == 2
    assert result['access_retry_sync_calls'] == 2
    assert result['access_retry_sync_host_seconds'] == 3
    assert result['access_retry_sync_failed_calls'] == 1
    assert result['fault_range_host_seconds'] == 4
    assert 'fault_range_failed_calls' not in result
    assert result['access_return_not_ready_calls'] == 2
    assert result['access_return_other_calls'] == 1


@pytest.mark.skipif(not os.environ.get("DDO_VMM_LIBRARY_PATH"), reason="diagnostic binary optional")
def test_actual_v2_export_is_cpu_only():
    import torch
    assert not torch.cuda.is_initialized()
    dll = ctypes.CDLL(os.environ['DDO_VMM_LIBRARY_PATH'])
    fn = dll.ddo_vmm_allocation_diagnostics_snapshot_v2
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    fn.restype = ctypes.c_int
    values = (ctypes.c_uint64 * 36)(*([99] * 36))
    assert fn(values, 35) == 1 and list(values) == [99] * 36
    assert fn(None, 36) == 1
    assert fn(values, 36) == 0 and values[0] == 2 and values[1] == 0
    assert all(value == 0 for value in values[2:])
    assert not torch.cuda.is_initialized()


@pytest.mark.skipif(not os.environ.get("DDO_VMM_LIBRARY_PATH"), reason="diagnostic binary optional")
def test_actual_export_is_cpu_only():
    import torch
    assert not torch.cuda.is_initialized()
    dll = ctypes.CDLL(os.environ["DDO_VMM_LIBRARY_PATH"])
    snapshot = dll.ddo_vmm_allocation_diagnostics_snapshot_v1
    snapshot.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    snapshot.restype = ctypes.c_int
    enable = dll.ddo_vmm_allocation_diagnostics_enable_v1
    enable.argtypes, enable.restype = [ctypes.c_int], ctypes.c_int
    values = (ctypes.c_uint64 * 26)(*([99] * 26))
    assert snapshot(values, 25) == 1 and list(values) == [99] * 26
    assert snapshot(None, 26) == 1
    assert enable(9) == 1
    assert snapshot(values, 26) == 0 and values[0] == 1 and values[1] == 0
    assert enable(1) == 0 and enable(1) == 1
    try:
        assert snapshot(values, 26) == 0 and values[1] == 1
        assert all(value == 0 for value in values[2:])
    finally:
        assert enable(0) == 0
    assert not torch.cuda.is_initialized()
