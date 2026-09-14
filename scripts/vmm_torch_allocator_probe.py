"""Run in a fresh Python process to validate the PyTorch VMM allocator bridge."""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--megabytes", type=int, default=64)
    parser.add_argument("--cache-mib", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--verify-stream-lifetime",
        action="store_true",
        help="Exercise allocator reuse after an asynchronous CUDA-stream free.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.megabytes <= 0 or args.cache_mib < 0:
        raise ValueError("--megabytes must be positive and --cache-mib cannot be negative")
    if args.cache_mib:
        os.environ["DDO_VMM_TORCH_CACHE_MIB"] = str(args.cache_mib)

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
    native.ddo_vmm_torch_cached_allocation_count.argtypes = []
    native.ddo_vmm_torch_cached_allocation_count.restype = ctypes.c_size_t
    native.ddo_vmm_torch_cached_mapped_bytes.argtypes = []
    native.ddo_vmm_torch_cached_mapped_bytes.restype = ctypes.c_size_t

    allocator = torch.cuda.memory.CUDAPluggableAllocator(
        str(library_path), "ddo_vmm_torch_malloc", "ddo_vmm_torch_free"
    )
    torch.cuda.memory.change_current_allocator(allocator)

    bytes_requested = args.megabytes * 1024**2
    def allocate_and_write() -> int:
        tensor = torch.empty(bytes_requested, dtype=torch.uint8, device="cuda")
        tensor.fill_(0xA5)
        torch.cuda.synchronize(args.device)
        address = tensor.data_ptr()
        del tensor
        gc.collect()
        torch.cuda.synchronize(args.device)
        return address

    with torch.cuda.device(args.device):
        address = allocate_and_write()
        allocation_count = int(native.ddo_vmm_torch_live_allocation_count())
        mapped_bytes = int(native.ddo_vmm_torch_live_mapped_bytes())
        cached_count = int(native.ddo_vmm_torch_cached_allocation_count())
        cached_bytes = int(native.ddo_vmm_torch_cached_mapped_bytes())
        if allocation_count != 0 or mapped_bytes != 0:
            raise RuntimeError("PyTorch allocation was not released by the VMM allocator")
        if args.cache_mib and (cached_count < 1 or cached_bytes < bytes_requested):
            raise RuntimeError("VMM cache did not retain the released allocation")
        reused_address = allocate_and_write() if args.cache_mib else None
        if reused_address is not None and reused_address != address:
            raise RuntimeError("VMM cache did not reuse the stable virtual address")

        if args.verify_stream_lifetime:
            stream = torch.cuda.Stream(device=args.device)
            with torch.cuda.stream(stream):
                asynchronous = torch.empty(bytes_requested, dtype=torch.uint8, device="cuda")
                asynchronous.fill_(0x5A)
                asynchronous_address = asynchronous.data_ptr()
                del asynchronous
            gc.collect()
            # Reusing a cached mapping must wait for the freeing stream.  A
            # write on the default stream makes a premature unmap observable.
            replacement = torch.empty(bytes_requested, dtype=torch.uint8, device="cuda")
            replacement.fill_(0x3C)
            torch.cuda.synchronize(args.device)
            if replacement.data_ptr() != asynchronous_address:
                raise RuntimeError("VMM cache did not reuse the asynchronous stream allocation")
            del replacement
            gc.collect()
            torch.cuda.synchronize(args.device)

    print("PyTorch CUDAPluggableAllocator VMM probe: PASS")
    print(f"  torch: {torch.__version__}")
    print(f"  requested: {args.megabytes:.2f} MiB")
    print(f"  tensor address: 0x{address:x}")
    print("  VMM live allocations after free: 0")
    print(f"  VMM cached allocations after free: {int(native.ddo_vmm_torch_cached_allocation_count())}")
    print(f"  VMM cached after free: {int(native.ddo_vmm_torch_cached_mapped_bytes()) / 1024**2:.2f} MiB")
    if reused_address is not None:
        print(f"  stable address reused: 0x{reused_address:x}")
    if args.verify_stream_lifetime:
        print("  asynchronous stream lifetime: verified")


if __name__ == "__main__":
    run(parse_args())
