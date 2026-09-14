"""Persistent PyTorch tensor views over explicitly managed CUDA VMM addresses."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
from typing import Any

from .native import NativeVmmRegion


class VmmTensorViewError(RuntimeError):
    """Raised when a stable VMM address cannot be represented as a CUDA tensor."""


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
        try:
            import torch
            from cuda.core import Buffer
        except ImportError as error:
            raise VmmTensorViewError("PyTorch and cuda-python[cuda.core] are required") from error

        info = region.info
        # Buffer owns neither the address reservation nor its backing.  Keeping
        # `region` as owner ties the DLPack view lifetime to the VMM region.
        buffer = Buffer.from_handle(info.address, info.mapped_size, owner=region)
        # cuda-python exposes an untyped byte buffer as int8 through DLPack.
        # Reinterpret it as uint8 without a copy; weight adapters can create
        # dtype/shape-specific views from this stable byte storage later.
        if region.backing == "host":
            # cuda-python correctly describes a CU_MEM_LOCATION_TYPE_HOST
            # allocation as kDLCUDAHost (3). PyTorch rejects that DLPack tag,
            # although CUDA kernels can address the mapped pointer. Rewrite
            # only the DLPack metadata to kDLCUDA (2), retaining the original
            # owner/deleter and the same mapped address. This is equivalent to
            # the driver-side CUDA tensor view and performs no data copy.
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
        if tensor.device.type != "cuda" or tensor.data_ptr() != info.address:
            raise VmmTensorViewError("cuda-python did not expose the requested VMM address as a CUDA tensor")
        return cls(region=region, tensor=tensor, _buffer=buffer)

    @property
    def address(self) -> int:
        return int(self.tensor.data_ptr())

    def require_materialized(self) -> None:
        if not self.region.materialized:
            raise VmmTensorViewError("attempted tensor use while VMM backing is evicted")
