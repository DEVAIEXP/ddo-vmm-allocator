"""Explicit ownership bridge for native immutable sources (no CUDA calls)."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from threading import RLock
from typing import Self

from .native import NativeVmmError, NativeVmmLibrary


class NativeSourceError(NativeVmmError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class _FileInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("version", ctypes.c_uint32),
        ("volume_id", ctypes.c_uint64), ("file_id", ctypes.c_uint64),
        ("size_bytes", ctypes.c_uint64), ("last_write_time", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class NativeFileInfo:
    volume_id: int
    file_id: int
    size_bytes: int
    last_write_time: int
    generation: int


def _bind(owner: NativeVmmLibrary):
    dll = owner._library
    try:
        version = dll.ddo_vmm_source_abi_version
        version.argtypes = []
        version.restype = ctypes.c_uint32
        if version() != 1:
            raise NativeVmmError("native file source requires ABI version 1")
        for name, arguments in {
            "open": [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
            "retain": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
            "get_info": [ctypes.c_void_p, ctypes.POINTER(_FileInfo)],
            "read": [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t],
            "close": [ctypes.POINTER(ctypes.c_void_p)],
        }.items():
            function = getattr(dll, f"ddo_vmm_source_file_{name}")
            function.argtypes = arguments
            function.restype = ctypes.c_int
        dll.ddo_vmm_source_last_error.argtypes = []
        dll.ddo_vmm_source_last_error.restype = ctypes.c_char_p
    except AttributeError as error:
        raise NativeVmmError("DLL does not provide native file sources; rebuild the allocator") from error
    return dll


class NativeFileSource:
    """An owned native file handle; retained owners survive each other's close.

    Operations on one wrapper are serialized against close. Retain a separate
    owner for concurrent readers. No model weights or Python file mappings are
    retained. This is the file foundation, not the registered cache itself.
    """

    def __init__(self, owner: NativeVmmLibrary, dll, handle: ctypes.c_void_p) -> None:
        self._owner = owner
        self._dll = dll
        self._handle = handle
        self._lock = RLock()

    @classmethod
    def open(cls, owner: NativeVmmLibrary, path: str | os.PathLike[str]) -> Self:
        path = os.fsdecode(path)
        if not path or "\0" in path:
            raise ValueError("source path must be nonempty and contain no NUL")
        dll = _bind(owner)
        source = cls(owner, dll, ctypes.c_void_p())
        source._check(dll.ddo_vmm_source_file_open(path.encode("utf-8"), ctypes.byref(source._handle)))
        return source

    def _check(self, status: int) -> None:
        if status:
            message = self._dll.ddo_vmm_source_last_error()
            raise NativeSourceError(status, message.decode("utf-8", errors="replace") if message else "native source failed")

    def _require_open(self) -> None:
        if not self._handle.value:
            raise NativeVmmError("native file source is closed")

    @property
    def closed(self) -> bool:
        with self._lock:
            return not self._handle.value

    @property
    def info(self) -> NativeFileInfo:
        with self._lock:
            self._require_open()
            info = _FileInfo(struct_size=ctypes.sizeof(_FileInfo), version=1)
            self._check(self._dll.ddo_vmm_source_file_get_info(self._handle, ctypes.byref(info)))
            return NativeFileInfo(info.volume_id, info.file_id, info.size_bytes, info.last_write_time, info.generation)

    def retain(self) -> Self:
        with self._lock:
            self._require_open()
            retained = type(self)(self._owner, self._dll, ctypes.c_void_p())
            self._check(self._dll.ddo_vmm_source_file_retain(self._handle, ctypes.byref(retained._handle)))
            return retained

    def read_into(self, destination: bytearray | memoryview, offset_bytes: int = 0) -> int:
        if not isinstance(offset_bytes, int) or not 0 <= offset_bytes <= 2**64 - 1:
            raise ValueError("offset must be an unsigned 64-bit integer")
        view = memoryview(destination)
        if view.readonly or not view.c_contiguous or not view.nbytes:
            raise ValueError("destination must be writable, contiguous and nonempty")
        buffer = (ctypes.c_ubyte * view.nbytes).from_buffer(view)
        with self._lock:
            self._require_open()
            self._check(self._dll.ddo_vmm_source_file_read(self._handle, offset_bytes, buffer, view.nbytes))
        return view.nbytes

    def close(self) -> None:
        with self._lock:
            self._check(self._dll.ddo_vmm_source_file_close(ctypes.byref(self._handle)))

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *_exception) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass  # Explicit close is the observable lifecycle contract.
