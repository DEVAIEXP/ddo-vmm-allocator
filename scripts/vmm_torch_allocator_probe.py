"""Run in a fresh Python process to validate the PyTorch VMM allocator bridge."""

from __future__ import annotations

import argparse
import ctypes
import gc
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--megabytes", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.megabytes <= 0:
        raise ValueError("--megabytes must be positive")

    # Importing torch is harmless; initializing CUDA before changing the global
    # allocator is not.  This script must always be launched as a fresh process.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA is already initialized; launch this probe in a fresh Python process")

    library_path = args.library.resolve(strict=True)
    native = ctypes.WinDLL(str(library_path)) if hasattr(ctypes, "WinDLL") else ctypes.CDLL(str(library_path))
    native.ddo_vmm_torch_live_allocation_count.argtypes = []
    native.ddo_vmm_torch_live_allocation_count.restype = ctypes.c_size_t
    native.ddo_vmm_torch_live_mapped_bytes.argtypes = []
    native.ddo_vmm_torch_live_mapped_bytes.restype = ctypes.c_size_t

    allocator = torch.cuda.memory.CUDAPluggableAllocator(
        str(library_path), "ddo_vmm_torch_malloc", "ddo_vmm_torch_free"
    )
    torch.cuda.memory.change_current_allocator(allocator)

    bytes_requested = args.megabytes * 1024**2
    with torch.cuda.device(args.device):
        tensor = torch.empty(bytes_requested, dtype=torch.uint8, device="cuda")
        tensor.fill_(0xA5)
        torch.cuda.synchronize(args.device)
        allocation_count = int(native.ddo_vmm_torch_live_allocation_count())
        mapped_bytes = int(native.ddo_vmm_torch_live_mapped_bytes())
        if allocation_count < 1 or mapped_bytes < bytes_requested:
            raise RuntimeError("PyTorch allocation was not retained by the VMM allocator")
        address = tensor.data_ptr()
        del tensor
        gc.collect()
        torch.cuda.synchronize(args.device)

    print("PyTorch CUDAPluggableAllocator VMM probe: PASS")
    print(f"  torch: {torch.__version__}")
    print(f"  requested: {args.megabytes:.2f} MiB")
    print(f"  tensor address: 0x{address:x}")
    print(f"  VMM live allocations during use: {allocation_count}")
    print(f"  VMM mapped during use: {mapped_bytes / 1024**2:.2f} MiB")
    print(f"  VMM live allocations after free: {int(native.ddo_vmm_torch_live_allocation_count())}")


if __name__ == "__main__":
    run(parse_args())

