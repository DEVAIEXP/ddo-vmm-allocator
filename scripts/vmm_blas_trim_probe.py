"""Small cuBLAS workspace/idle trim CUDA gate; explicit approval required."""

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

    from diffusers_dynamic_offloader import release_native_vmm_idle_cache

    install_pytorch_vmm_allocator(library)
    owner = NativeVmmLibrary(library)

    def gemm():
        a = torch.ones((64, 64), dtype=torch.float32, device="cuda")
        b = torch.ones_like(a)
        c = a @ b
        if c.cpu().tolist() != [[64.0] * 64 for _ in range(64)]:
            raise RuntimeError("GEMM payload mismatch")

    gemm()
    gc.collect()
    torch.cuda.synchronize()
    owner.trim_idle_torch_pool()
    before = owner.torch_pool_stats()
    if not before["pool_live_allocation_count"]:
        raise RuntimeError("GEMM did not retain a workspace; hypothesis not tested")
    report = release_native_vmm_idle_cache(enabled=True, phase="probe", clear_cublas_workspaces=True)
    after = owner.torch_pool_stats()
    if any(
        after[k] for k in ("pool_mapped_bytes", "pool_live_allocation_count", "pool_failures", "pool_quarantined_bytes")
    ):
        raise RuntimeError(f"workspace/backing not released: {after}")
    gemm()  # Workspace must recreate safely, without disabling future GEMM.
    release_native_vmm_idle_cache(enabled=True, phase="probe_repeat", clear_cublas_workspaces=True)
    final = owner.torch_pool_stats()
    if any(
        final[k] for k in ("pool_mapped_bytes", "pool_live_allocation_count", "pool_failures", "pool_quarantined_bytes")
    ):
        raise RuntimeError(f"repeat cleanup failed: {final}")
    print(json.dumps({"status": "PASS", "before": before, "after": after, "final": final, "report": report.as_dict()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    run(parser.parse_args().library.resolve(strict=True))
