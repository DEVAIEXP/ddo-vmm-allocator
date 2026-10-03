"""Fresh-process CUDA span-pool gate. Running this requires explicit approval."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path


def run(library: Path) -> None:
    os.environ["DDO_VMM_TORCH_CACHE_MIB"] = "512"
    import torch
    from ddo_vmm_allocator.native import NativeVmmLibrary

    if torch.cuda.is_initialized():
        raise RuntimeError("run this gate in a fresh process")
    owner = NativeVmmLibrary(library)
    if not owner.torch_pool_stats():
        raise RuntimeError("selected DLL has no active native span pool")
    allocator = torch.cuda.memory.CUDAPluggableAllocator(
        str(owner.path), "ddo_vmm_torch_malloc", "ddo_vmm_torch_free")
    torch.cuda.memory.change_current_allocator(allocator)

    def snapshot():
        result = owner.torch_pool_stats()
        if result["pool_failures"] or result["pool_quarantined_bytes"]:
            raise RuntimeError(f"pool ownership failure: {result}")
        return result

    # Adjacent small leases must share a physical mapping, stay aligned and
    # preserve each other's payload. No large reservation or artificial delay.
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        left = torch.empty(512, dtype=torch.uint8, device="cuda")
        right = torch.empty(1024, dtype=torch.uint8, device="cuda")
        left.fill_(0x11)
        right.fill_(0x22)
        left_address, right_address = left.data_ptr(), right.data_ptr()
        before = snapshot()
        if (left_address % 512 or right_address != left_address + 512
                or before["pool_backing_count"] != 1
                or before["pool_leased_bytes"] != 1536):
            raise RuntimeError(f"interior pointer/alignment/unique-backing gate failed: {before}")
        del left
    gc.collect()
    with torch.cuda.stream(consumer):
        replacement = torch.empty(512, dtype=torch.uint8, device="cuda")
        if replacement.data_ptr() != left_address:
            raise RuntimeError("pending interior span was not reused")
        replacement.fill_(0x33)
        replacement_cpu = replacement.cpu()
        del replacement
    producer.synchronize()
    right_cpu = right.cpu()
    if not bool(torch.all(right_cpu == 0x22)) or not bool(torch.all(replacement_cpu == 0x33)):
        raise RuntimeError("neighbor/cross-stream payload corruption")
    del right
    gc.collect()
    torch.cuda.synchronize()
    final = snapshot()
    if final["pool_live_allocation_count"] or final["pool_reserved_allocation_count"]:
        raise RuntimeError(f"live lease leaked: {final}")
    if final["pool_mapped_bytes"] != before["pool_mapped_bytes"]:
        raise RuntimeError("small-span reuse changed physical backing consumption")
    print("Native span pool interior-pointer/cross-stream CUDA gate: PASS")
    print(final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    run(parser.parse_args().library)
