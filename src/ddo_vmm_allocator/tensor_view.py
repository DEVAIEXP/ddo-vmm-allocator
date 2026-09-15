"""Persistent PyTorch tensor views over explicitly managed CUDA VMM addresses."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
from typing import Any

from .native import NativeModelArena, NativeVmmRegion


class VmmTensorViewError(RuntimeError):
    """Raised when a stable VMM address cannot be represented as a CUDA tensor."""


def cuda_uint8_tensor_from_address(address: int, size_bytes: int, *, owner: Any, backing: str = "device") -> tuple[Any, Any]:
    """Create a PyTorch uint8 CUDA tensor view over an existing CUDA VA range."""

    try:
        import torch
        from cuda.core import Buffer
    except ImportError as error:
        raise VmmTensorViewError("PyTorch and cuda-python[cuda.core] are required") from error

    buffer = Buffer.from_handle(address, size_bytes, owner=owner)
    if backing == "host":
        capsule = buffer.__dlpack__()
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        get_pointer.restype = ctypes.c_void_p
        managed_tensor = get_pointer(capsule, b"dltensor")
        if not managed_tensor:
            raise VmmTensorViewError("cuda-python returned an invalid DLPack capsule")
        device_type = ctypes.cast(
            managed_tensor + ctypes.sizeof(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int32),
        )
        if device_type[0] != 3:
            raise VmmTensorViewError(
                f"unexpected host VMM DLPack device type: {device_type[0]}"
            )
        device_type[0] = 2
        tensor = torch.from_dlpack(capsule).view(torch.uint8)
    else:
        tensor = torch.from_dlpack(buffer).view(torch.uint8)
    if tensor.device.type != "cuda" or tensor.data_ptr() != address:
        raise VmmTensorViewError("cuda-python did not expose the requested address as a CUDA tensor")
    return tensor, buffer


@dataclass
class PersistentVmmTensorView:
    """One uint8 CUDA tensor whose address stays valid across rematerialization.

    The tensor must never be read or written while ``region.materialized`` is
    false.  Eviction discards its data; a caller must repopulate weights before
    the next layer uses the view.
    """

    region: NativeVmmRegion
    tensor: Any
    _buffer: Any

    @classmethod
    def create(cls, region: NativeVmmRegion) -> "PersistentVmmTensorView":
        if not region.materialized:
            raise VmmTensorViewError("VMM region must be materialized before creating a tensor view")
        info = region.info
        tensor, buffer = cuda_uint8_tensor_from_address(
            info.address, info.mapped_size, owner=region, backing=region.backing
        )
        return cls(region=region, tensor=tensor, _buffer=buffer)

    @property
    def address(self) -> int:
        return int(self.tensor.data_ptr())

    def require_materialized(self) -> None:
        if not self.region.materialized:
            raise VmmTensorViewError("attempted tensor use while VMM backing is evicted")
