"""Validate bounded stable-address residency for three independent VMM groups."""

from __future__ import annotations

import argparse

from ddo_vmm_allocator.native import load_native_vmm
from ddo_vmm_allocator.weight_arena import VmmWeightArena


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--group-megabytes", type=int, default=32)
    parser.add_argument("--resident-megabytes", type=int, default=64)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.group_megabytes <= 0 or args.resident_megabytes <= 0:
        raise ValueError("group and resident sizes must be positive")
    group_bytes = args.group_megabytes * 1024**2
    resident_bytes = args.resident_megabytes * 1024**2
    library = load_native_vmm(args.library)

    with VmmWeightArena(library, resident_budget_bytes=resident_bytes) as arena:
        original_a = arena.reserve("a", group_bytes).address
        arena.reserve("b", group_bytes)
        arena.reserve("c", group_bytes)

        with arena.use("a"):
            pass
        with arena.use("b"):
            pass
        with arena.use("c"):
            pass

        after_c = arena.view("a")
        if after_c.materialized:
            raise RuntimeError("arena did not evict the least-recently-used group under budget pressure")

        with arena.use("a") as rematerialized_a:
            if rematerialized_a.address != original_a:
                raise RuntimeError("rematerialized group changed its virtual address")

        print("VMM weight arena probe: PASS")
        print(f"  groups: 3 x {args.group_megabytes} MiB")
        print(f"  resident budget: {args.resident_megabytes} MiB")
        print(f"  group a stable address: 0x{original_a:x}")
        print(f"  resident after LRU cycling: {arena.resident_bytes / 1024**2:.2f} MiB")


if __name__ == "__main__":
    run(parse_args())

