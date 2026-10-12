"""Bounded, opt-in host registration capacity diagnostic; not a runtime tuner."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def search_capacity(attempt, ceiling, window):
    """Find an observed passing capacity assuming monotonic admission."""
    records = []

    def check(slots):
        result = attempt(slots)
        records.append(result)
        if result["status"] not in {"PASS", "OOM"}:
            raise RuntimeError(f"Unexpected probe failure; stop: {result}")
        return result["status"] == "PASS"

    if not check(1):
        return 0, records
    low, high = 1, ceiling // window
    if high > 1:
        if check(high):
            low = high
        else:
            while high - low > 1:
                middle = (low + high) // 2
                if check(middle):
                    low = middle
                else:
                    high = middle
    # Repeat the selected observation in another fresh process.
    if not check(low):
        raise RuntimeError("Selected capacity did not repeat; no recommendation")
    return low * window, records


def child_probe(args):
    import torch
    from ddo_vmm_allocator.native import NativeVmmError, load_native_vmm

    try:
        # Match the standalone controls, not the native model bootstrap.
        torch.empty(1, device=f"cuda:{args.device}")
        library = load_native_vmm(args.library)
        with library.create_host_ring(
            args.window_mib * 1024**2, args.child_slots, device_ordinal=args.device
        ):
            pass
    except NativeVmmError as error:
        status = "OOM" if "cuMemHostRegister failed: CUDA_ERROR_OUT_OF_MEMORY" in str(error) else "ERROR"
        print(json.dumps({"status": status, "error": str(error)}), flush=True)
        return 2 if status == "OOM" else 3
    except Exception as error:
        print(json.dumps({"status": "ERROR", "error": repr(error)}), flush=True)
        return 3
    print(json.dumps({"status": "PASS"}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=os.getenv("DDO_VMM_LIBRARY_PATH"))
    parser.add_argument("--max-mib", type=int, required=True, help="Explicit total probe ceiling; <=4096 MiB")
    parser.add_argument("--window-mib", type=int, default=64)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child-slots", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.library is None:
        parser.error("Set --library or DDO_VMM_LIBRARY_PATH")
    args.library = args.library.resolve(strict=True)
    if not (0 < args.window_mib <= args.max_mib <= 4096) or args.max_mib % args.window_mib:
        parser.error("Require 0 < window <= max <=4096 MiB and max divisible by window")
    if args.timeout_sec <= 0 or args.device < 0:
        parser.error("Require a positive timeout and non-negative device")
    if args.child_slots is not None:
        if not 0 < args.child_slots <= args.max_mib // args.window_mib:
            parser.error("Child capacity exceeds the ceiling")
        return child_probe(args)

    report = {
        "schema_version": 1,
        "platform": platform.platform(),
        "python": sys.version,
        "library": str(args.library),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "allocator": "normal_torch_control",
        "max_mib": args.max_mib,
        "window_mib": args.window_mib,
        "device": args.device,
        "scope": "No model, no DMA validation, no runtime settings changed",
        "limitations": "Observed fresh-process capacity, not a guaranteed maximum; admission may vary with load",
        "attempts": [],
    }

    def attempt(slots):
        command = [sys.executable, "-B", str(Path(__file__).resolve()),
                   "--library", str(args.library), "--max-mib", str(args.max_mib),
                   "--window-mib", str(args.window_mib), "--device", str(args.device),
                   "--child-slots", str(slots)]
        start = time.monotonic()
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout_sec)
            lines = process.stdout.strip().splitlines()
            result = json.loads(lines[-1]) if lines else {"status": "ERROR"}
            expected = 0 if result.get("status") == "PASS" else 2
            if process.returncode != expected:
                result["status"] = "ERROR"
            result.update(returncode=process.returncode, stderr=process.stderr[-4000:])
        except subprocess.TimeoutExpired:
            result = {"status": "TIMEOUT", "error": "Child killed and waited by subprocess.run"}
        except (ValueError, OSError) as error:
            result = {"status": "ERROR", "error": repr(error)}
        result.update(total_mib=slots * args.window_mib, slots=slots,
                      elapsed_seconds=time.monotonic() - start)
        report["attempts"].append(result)
        print(json.dumps(result), flush=True)
        return result

    try:
        capacity, _ = search_capacity(attempt, args.max_mib, args.window_mib)
        report.update(status="PASS" if capacity else "NO_CAPACITY", observed_pass_mib=capacity,
                      ceiling_passed=capacity == args.max_mib)
    except RuntimeError as error:
        report.update(status="ERROR", error=str(error))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
