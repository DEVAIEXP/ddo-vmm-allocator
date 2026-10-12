"""Small fresh-process CUDA trim gate; requires explicit user approval."""

import argparse
import gc
import json
import os
from pathlib import Path


def run(library):
    os.environ["DDO_VMM_TORCH_CACHE_MIB"] = "512"
    import torch
    from ddo_vmm_allocator.native import NativeVmmLibrary
    from ddo_vmm_allocator.torch_allocator import install_pytorch_vmm_allocator

    if torch.cuda.is_initialized():
        raise RuntimeError("gate requires a fresh process")
    install_pytorch_vmm_allocator(library)
    owner = NativeVmmLibrary(library)
    producer = torch.cuda.Stream()
    live = torch.full((1024,), 17, dtype=torch.uint8, device="cuda")
    with torch.cuda.stream(producer):
        dead = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        dead.fill_(29)
        del dead
    gc.collect()
    before = owner.torch_pool_stats()
    result = owner.trim_idle_torch_pool()  # Pending idle backing must prove completion.
    if result["status"] != "completed":
        raise RuntimeError(f"trim unavailable: {result}")
    after = owner.torch_pool_stats()
    if after["pool_mapped_bytes"] >= before["pool_mapped_bytes"]:
        raise RuntimeError("whole-idle backing was not released")
    if live.cpu().tolist() != [17] * 1024:
        raise RuntimeError("live neighbor payload changed")
    if after["pool_failures"] or after["pool_quarantined_bytes"]:
        raise RuntimeError(f"ownership anomaly: {after}")
    del live
    gc.collect()
    torch.cuda.synchronize()
    owner.trim_idle_torch_pool()
    owner.trim_idle_torch_pool()  # Repeated empty trim is harmless.
    final = owner.torch_pool_stats()
    if any(
        final[k] for k in ("pool_mapped_bytes", "pool_live_allocation_count", "pool_failures", "pool_quarantined_bytes")
    ):
        raise RuntimeError(f"unexpected retained backing: {final}")
    fresh = torch.full((1024,), 31, dtype=torch.uint8, device="cuda")
    if fresh.cpu().tolist() != [31] * 1024:
        raise RuntimeError("allocation after trim failed")
    del fresh
    gc.collect()
    torch.cuda.synchronize()
    owner.trim_idle_torch_pool()
    print(json.dumps({"status": "PASS", "before": before, "after": after, "final": owner.torch_pool_stats()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    run(parser.parse_args().library.resolve(strict=True))
