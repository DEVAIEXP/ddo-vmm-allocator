"""Validate one PyTorch tensor view across VMM backing eviction/rematerialization."""

from __future__ import annotations

import argparse

from ddo_vmm_allocator.native import load_native_vmm
from ddo_vmm_allocator.tensor_view import PersistentVmmTensorView


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--megabytes", type=int, default=32)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.megabytes <= 0:
        raise ValueError("--megabytes must be positive")
    import torch

    bytes_requested = args.megabytes * 1024**2
    library = load_native_vmm(args.library)
    with library.create_region(bytes_requested) as region:
        view = PersistentVmmTensorView.create(region)
        original_address = view.address
        view.tensor.fill_(0xA5)
        torch.cuda.synchronize()
        torch.testing.assert_close(view.tensor[:4096].cpu(), torch.full((4096,), 0xA5, dtype=torch.uint8))

        region.evict()
        if region.materialized:
            raise RuntimeError("region remained materialized after eviction")
        region.materialize()
        view.require_materialized()
        if view.address != original_address:
            raise RuntimeError("persistent PyTorch VMM tensor changed address after rematerialization")
        view.tensor.fill_(0x5A)
        torch.cuda.synchronize()
        torch.testing.assert_close(view.tensor[:4096].cpu(), torch.full((4096,), 0x5A, dtype=torch.uint8))

        print("Persistent PyTorch VMM tensor probe: PASS")
        print(f"  torch: {torch.__version__}")
        print(f"  bytes: {args.megabytes:.2f} MiB")
        print(f"  stable address: 0x{original_address:x}")


if __name__ == "__main__":
    run(parse_args())

