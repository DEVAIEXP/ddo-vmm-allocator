"""Read the Windows DXGI budgets used by the native VMM budget guard."""

from __future__ import annotations

import argparse

from ddo_vmm_allocator.native import NativeVmmError, load_native_vmm


def _gib(value: int) -> float:
    return value / 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="Path to ddo_vmm.dll")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--local-reserve-gib", type=float, default=0.0)
    parser.add_argument("--shared-reserve-gib", type=float, default=4.0)
    parser.add_argument("--system-reserve-gib", type=float, default=4.0)
    parser.add_argument(
        "--verify-safe-refusal",
        action="store_true",
        help="Verify that an impossible system reserve rejects rematerialization without allocating more GPU memory.",
    )
    return parser.parse_args()


def to_bytes(value: float, flag: str) -> int:
    if value < 0:
        raise SystemExit(f"{flag} must be non-negative")
    return int(value * 1024**3)


def main() -> None:
    args = parse_args()
    library = load_native_vmm(args.library)
    before = library.windows_memory_info(args.device)
    library.set_dxgi_budget_guard(
        device_ordinal=args.device,
        local_reserve_bytes=to_bytes(args.local_reserve_gib, "--local-reserve-gib"),
        shared_reserve_bytes=to_bytes(args.shared_reserve_gib, "--shared-reserve-gib"),
        system_reserve_bytes=to_bytes(args.system_reserve_gib, "--system-reserve-gib"),
    )
    print("Windows VMM DXGI budget guard probe: PASS")
    print(f"  local: used {_gib(before.local_usage_bytes):.2f} GiB / budget {_gib(before.local_budget_bytes):.2f} GiB")
    print(f"  shared: used {_gib(before.nonlocal_usage_bytes):.2f} GiB / budget {_gib(before.nonlocal_budget_bytes):.2f} GiB")
    print(f"  available physical RAM: {_gib(before.available_physical_bytes):.2f} GiB")
    print(f"  available commit: {_gib(before.available_commit_bytes):.2f} GiB")
    print(
        "  configured reserve: "
        f"local={args.local_reserve_gib:.2f} GiB "
        f"shared={args.shared_reserve_gib:.2f} GiB "
        f"system={args.system_reserve_gib:.2f} GiB"
    )
    if args.verify_safe_refusal:
        region = library.create_region(64 * 1024**2, args.device)
        try:
            region.evict()
            library.set_dxgi_budget_guard(
                device_ordinal=args.device,
                system_reserve_bytes=before.available_physical_bytes + 1,
            )
            try:
                region.materialize()
            except NativeVmmError:
                print("  impossible reserve: safe refusal PASS")
            else:
                raise RuntimeError("an impossible DXGI budget guard reserve unexpectedly materialized a region")
        finally:
            region.close()


if __name__ == "__main__":
    main()
