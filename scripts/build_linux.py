"""Build isolated CPU contracts or the experimental Linux library; never run CUDA.

Run with --dry-run on any host to review commands. Real execution requires Linux.
This helper neither installs dependencies nor modifies the Windows build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def build_commands(root: Path, mode: str, jobs: int) -> list[list[str]]:
    if mode not in {"cpu", "library"}:
        raise ValueError("mode must be cpu or library")
    if not 1 <= jobs <= 64:
        raise ValueError("jobs must be between 1 and 64")
    build = root / ("build-linux-cpu" if mode == "cpu" else "build-linux")
    source = root / "native" / "tests" / "cpu" if mode == "cpu" else root
    configure = ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"]
    if mode == "library":
        configure += [
            "-DDDO_VMM_BUILD_PROBES=OFF",
            "-DDDO_VMM_NATIVE_SPAN_POOL=ON",
            "-DDDO_VMM_NATIVE_SOURCE_DISPATCHER=OFF",
            "-DDDO_VMM_SOURCE_TEST_HOOKS=OFF",
        ]
    commands = [configure, ["cmake", "--build", str(build), "--parallel", str(jobs)]]
    if mode == "cpu":
        commands.append(["ctest", "--test-dir", str(build), "--output-on-failure", "--no-tests=error"])
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "library"), default="cpu")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="print argv arrays without executing")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        commands = build_commands(root, args.mode, args.jobs)
    except ValueError as error:
        parser.error(str(error))
    if not args.dry_run and sys.platform != "linux":
        parser.error("execution requires Linux; use --dry-run to review from Windows")
    for command in commands:
        print(json.dumps(command), flush=True)
        if args.dry_run:
            continue
        try:
            result = subprocess.run(command, check=False)
        except OSError as error:
            print(f"Build tool unavailable: {error}", file=sys.stderr)
            return 1
        if result.returncode:
            return result.returncode
    if args.mode == "library" and not args.dry_run:
        library = root / "build-linux" / "libddo_vmm.so"
        if not library.is_file():
            print(f"Expected shared library absent: {library}", file=sys.stderr)
            return 1
        digest = hashlib.sha256()
        with library.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        print(f"SHA256 {digest.hexdigest()}  {library}")
        print("Compilation only: Linux cache/transport and CUDA safety still require validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
