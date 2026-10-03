"""Native file contracts. Load the DLL but never initialize or call CUDA."""

from concurrent.futures import ThreadPoolExecutor
import ctypes
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ddo_vmm_allocator.native import NativeVmmError, load_native_vmm
from ddo_vmm_allocator.source import NativeFileSource, NativeSourceError, _FileInfo


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows native source contract")


@pytest.fixture
def library():
    path = Path(os.environ.get("DDO_VMM_LIBRARY_PATH", Path(__file__).parents[1] / "build/Release/ddo_vmm.dll"))
    if not path.exists():
        pytest.fail(f"Build native source DLL before running its contracts: {path}")
    return load_native_vmm(path)


def test_positioned_read_crosses_chunks_and_preserves_sentinels(library, tmp_path):
    payload = bytes(range(251)) * 18000
    path = tmp_path / "weights-unicode-\u00e9.bin"
    path.write_bytes(payload)
    with library.open_source_file(path) as source:
        destination = bytearray(b"!" * (len(payload) + 2))
        assert source.read_into(memoryview(destination)[1:-1]) == len(payload)
        assert destination == b"!" + payload + b"!"
        tail = bytearray(13)
        source.read_into(tail, len(payload) - 13)
        assert tail == payload[-13:]
        assert source.info.size_bytes == len(payload)


def test_retained_owner_keeps_identity_and_write_protection(library, tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"immutable")
    original = library.open_source_file(path)
    retained = original.retain()
    try:
        identity = original.info
        original.close()
        original.close()
        assert retained.info == identity
        with pytest.raises(OSError):
            with path.open("r+b"):
                pass
        with pytest.raises(OSError):
            path.unlink()
        data = bytearray(9)
        retained.read_into(data)
        assert data == b"immutable"
        with pytest.raises(NativeVmmError, match="closed"):
            original.retain()
    finally:
        original.close()
        retained.close()
    path.write_bytes(b"new generation")
    with library.open_source_file(path) as reopened:
        assert reopened.info.generation != identity.generation
        assert reopened.info.size_bytes == 14


def test_alias_paths_have_same_file_identity(library, tmp_path):
    path = tmp_path / "weights.bin"
    alias = tmp_path / "alias.bin"
    path.write_bytes(b"abc")
    os.link(path, alias)
    with library.open_source_file(path) as first, library.open_source_file(alias) as second:
        assert (first.info.volume_id, first.info.file_id) == (second.info.volume_id, second.info.file_id)


@pytest.mark.parametrize("offset,size", [(0, 4), (3, 1), (2**64 - 1, 1)])
def test_native_bounds_rejected_before_destination_write(library, tmp_path, offset, size):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abc")
    with library.open_source_file(path) as source:
        buffer = bytearray(b"!" * size)
        with pytest.raises(NativeSourceError, match="bounds") as error:
            source.read_into(buffer, offset)
        assert error.value.status == 1
        assert buffer == b"!" * size


def test_invalid_abi_and_closed_handle(library, tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abc")
    with library.open_source_file(path) as source:
        wrong = _FileInfo(struct_size=ctypes.sizeof(_FileInfo), version=2)
        assert source._dll.ddo_vmm_source_file_get_info(source._handle, ctypes.byref(wrong)) == 1
        wrong.version = 1
        wrong.struct_size = 8
        assert source._dll.ddo_vmm_source_file_get_info(source._handle, ctypes.byref(wrong)) == 1
        buffer = ctypes.create_string_buffer(1)
        assert source._dll.ddo_vmm_source_file_read(source._handle, 0, buffer, 0) == 1
    with pytest.raises(NativeVmmError, match="closed"):
        source.read_into(bytearray(1))


def test_missing_file_and_invalid_python_arguments(library, tmp_path):
    with pytest.raises(NativeSourceError):
        library.open_source_file(tmp_path / "missing")
    for path in ("", "has\0nul"):
        with pytest.raises(ValueError):
            library.open_source_file(path)
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abc")
    with library.open_source_file(path) as source:
        for destination in (b"abc", bytearray(), memoryview(bytearray(4))[::2]):
            with pytest.raises(ValueError):
                source.read_into(destination)
        for offset in (-1, 2**64):
            with pytest.raises(ValueError):
                source.read_into(bytearray(1), offset)


def test_concurrent_retained_reads_do_not_share_file_cursor(library, tmp_path):
    payload = bytes(range(251)) * 1000
    path = tmp_path / "weights.bin"
    path.write_bytes(payload)
    root = library.open_source_file(path)
    owners = [root.retain() for _ in range(8)]
    root.close()

    def read(owner, index):
        with owner:
            for iteration in range(12):
                offset = index * 701 + iteration * 31
                result = bytearray(4001)
                owner.read_into(result, offset)
                assert result == payload[offset:offset + len(result)]

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(read, owner, index) for index, owner in enumerate(owners)]
            for future in futures:
                future.result()
    finally:
        for owner in owners:
            owner.close()
    path.unlink()  # Last owner released the underlying OS handle.


def test_existing_writer_blocks_immutable_source(library, tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abc")
    with path.open("r+b"):
        with pytest.raises(NativeSourceError):
            library.open_source_file(path)
    with library.open_source_file(path) as source:
        assert source.info.size_bytes == 3


def test_old_dll_reports_missing_capability():
    owner = SimpleNamespace(_library=SimpleNamespace())
    with pytest.raises(NativeVmmError, match="rebuild"):
        NativeFileSource.open(owner, "unused.bin")


def test_repeated_success_and_failed_open_release_os_handles(library, tmp_path):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessHandleCount.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.GetProcessHandleCount.restype = ctypes.c_int

    def handle_count():
        count = ctypes.c_ulong()
        assert kernel.GetProcessHandleCount(kernel.GetCurrentProcess(), ctypes.byref(count))
        return count.value

    path = tmp_path / "weights.bin"
    path.write_bytes(b"abc")
    def cycle():
        with library.open_source_file(path) as source:
            with source.retain() as retained:
                retained.read_into(bytearray(3))
        with pytest.raises(NativeSourceError):
            library.open_source_file(tmp_path / "missing")

    # Prime success, retain AND exception paths; runtime initialization must
    # not be counted as a leaked per-operation file/event handle.
    cycle()
    before = handle_count()
    for _ in range(100):
        cycle()
    assert handle_count() <= before


def test_source_does_not_initialize_cuda(library, tmp_path):
    import torch

    assert not torch.cuda.is_initialized(), "Run CPU source contracts in a fresh process"
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abc")
    with library.open_source_file(path) as source:
        source.read_into(bytearray(3))
    assert not torch.cuda.is_initialized()


def test_changed_metadata_is_rejected_before_read(library, tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abc")
    with library.open_source_file(path) as source:
        original = path.stat()
        # Windows can allow attribute-only changes even when data writes are
        # denied by sharing. Revalidation must catch this generation change.
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 2_000_000_000))
        result = bytearray(b"!!!")
        with pytest.raises(NativeSourceError) as error:
            source.read_into(result)
        assert error.value.status == 5
        assert result == b"!!!"
