"""Verify native pressure eviction honors pinned VMM regions."""

from __future__ import annotations

import argparse

from ddo_vmm_allocator.native import load_native_vmm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--megabytes", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.megabytes <= 0:
        raise ValueError("--megabytes must be positive")
    library = load_native_vmm(args.library)
    size_bytes = args.megabytes * 1024**2
    with library.create_region(size_bytes, args.device) as pinned, library.create_region(size_bytes, args.device) as evictable:
        pinned.set_pressure_pinned(True)
        evictable.set_pressure_pinned(False)
        if not library.pressure_evict_one(args.device):
            raise RuntimeError("native pressure path found no evictable region")
        if not pinned.materialized or evictable.materialized:
            raise RuntimeError("native pressure path violated pinned-region safety")
        pinned.set_pressure_pinned(False)
        if not library.pressure_evict_one(args.device) or pinned.materialized:
            raise RuntimeError("native pressure path did not evict an unpinned region")

    print("Native VMM pressure probe: PASS")
    print(f"  region size: {args.megabytes:.2f} MiB")
    print("  pinned region preserved: yes")
    print("  unpinned regions evicted: yes")


if __name__ == "__main__":
    run(parse_args())
