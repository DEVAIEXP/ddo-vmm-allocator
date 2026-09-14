"""Validate VMM weight reloads and CUDA-event-safe unpinning."""

from __future__ import annotations

import argparse

from ddo_vmm_allocator.native import load_native_vmm
from ddo_vmm_allocator.weight_arena import VmmWeightArena


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--megabytes", type=int, default=16)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.megabytes <= 0:
        raise ValueError("--megabytes must be positive")
    import torch

    bytes_per_group = args.megabytes * 1024**2
    source_a = torch.full((bytes_per_group,), 0x2A, dtype=torch.uint8)
    source_b = torch.full((bytes_per_group,), 0x4B, dtype=torch.uint8)
    library = load_native_vmm(args.library)

    with VmmWeightArena(library, resident_budget_bytes=bytes_per_group) as arena:
        address_a = arena.reserve("a", bytes_per_group).address
        arena.reserve("b", bytes_per_group)
        arena.bind_cpu_tensor("a", source_a)
        arena.bind_cpu_tensor("b", source_b)

        weight_a = arena.pin_loaded_tensor("a")
        torch.testing.assert_close(weight_a[:4096].cpu(), source_a[:4096])
        arena.release_after_current_stream("a")
        torch.cuda.synchronize()
        if arena.collect_completed_events() != 1:
            raise RuntimeError("CUDA event did not release group a")

        weight_b = arena.pin_loaded_tensor("b")
        torch.testing.assert_close(weight_b[:4096].cpu(), source_b[:4096])
        arena.release_after_current_stream("b")
        torch.cuda.synchronize()
        arena.collect_completed_events()

        reloaded_a = arena.pin_loaded_tensor("a")
        if reloaded_a.data_ptr() != address_a:
            raise RuntimeError("weight a did not retain its virtual address after eviction")
        torch.testing.assert_close(reloaded_a[:4096].cpu(), source_a[:4096])
        arena.release_after_current_stream("a")
        torch.cuda.synchronize()
        arena.collect_completed_events()

        print("VMM tensor weight arena probe: PASS")
        print(f"  groups: 2 x {args.megabytes} MiB")
        print(f"  resident budget: {args.megabytes} MiB")
        print(f"  group a stable tensor address: 0x{address_a:x}")


if __name__ == "__main__":
    run(parse_args())

