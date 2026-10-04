"""Actual Linux shared-library/file ABI, with no CUDA calls or Torch import."""
import ctypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ddo_vmm_allocator.native import load_native_vmm
from ddo_vmm_allocator.source import NativeSourceError

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux file ABI contract")


@pytest.fixture
def library():
    path = Path(os.environ.get("DDO_VMM_LIBRARY_PATH", Path(__file__).parents[1] / "build-linux/libddo_vmm.so"))
    assert path.is_file(), f"Compile the Linux library first: {path}"
    return load_native_vmm(path)


def test_linux_exports_and_capabilities_without_cuda(library):
    native = library._library
    for name in ("source_abi_version", "source_cache_abi_version", "source_dispatcher_enabled"):
        function = getattr(native, "ddo_vmm_" + name)
        function.argtypes = []
        function.restype = ctypes.c_uint32
    assert native.ddo_vmm_source_abi_version() == 1
    assert native.ddo_vmm_source_cache_abi_version() == 11
    assert native.ddo_vmm_source_dispatcher_enabled() == 0
    for name in ("source_batch_submit", "torch_malloc", "torch_pool_stats_v1", "host_ring_read_file_async"):
        assert getattr(native, "ddo_vmm_" + name) is not None


def test_linux_retained_descriptor_reads_replaced_path_concurrently(library, tmp_path):
    payload = bytes(range(251)) * 18000
    path = tmp_path / "weights-unicode-\u00e9.bin"
    path.write_bytes(payload)
    source = library.open_source_file(path)
    owners = [source.retain() for _ in range(4)]
    identity = source.info
    source.close()
    replacement = tmp_path / "new.bin"
    replacement.write_bytes(b"new")
    replacement.replace(path)
    try:
        def read(owner):
            data = bytearray(len(payload))
            owner.read_into(data)
            assert data == payload
            assert owner.info == identity
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(read, owners))
        with library.open_source_file(path) as newer:
            assert newer.info.generation != identity.generation
            assert newer.info.file_id != identity.file_id
    finally:
        for owner in owners:
            owner.close()


def test_linux_changed_file_and_invalid_range_fail_explicitly(library, tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"immutable")
    with library.open_source_file(path) as source:
        with pytest.raises(NativeSourceError):
            source.read_into(bytearray(10))
        # POSIX does not lock out external writers. Metadata invalidation is
        # detected; never claim Windows deny-write semantics on Linux.
        path.write_bytes(b"changed and resized")
        with pytest.raises(NativeSourceError):
            source.read_into(bytearray(1))
