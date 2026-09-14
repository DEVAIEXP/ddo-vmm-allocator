"""Persistent PyTorch tensor views over explicitly managed CUDA VMM addresses."""

from __future__ import annotations

from dataclasses import dataclass
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
