"""Launch a Python runner with the experimental VMM PyTorch allocator installed first."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


BOOTSTRAP = """
from ddo_vmm_allocator.torch_allocator import install_pytorch_vmm_allocator
import runpy
import sys
from pathlib import Path

install_pytorch_vmm_allocator()
target = sys.argv[1]
sys.path.insert(0, str(Path(target).resolve().parent))
sys.argv = [target, *sys.argv[2:]]
runpy.run_path(target, run_name="__main__")
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True, help="Path to ddo_vmm.dll or libddo_vmm.so")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter that owns the target runner's PyTorch installation",
    )
    parser.add_argument(
        "--experimental-allow-global-allocator",
        action="store_true",
        help="Acknowledge that this replaces PyTorch's CUDA allocator in the child process",
    )
    parser.add_argument(
        "--allocator-cache-mib",
        type=int,
        default=0,
        help="Maximum physical VMM cache retained by the child allocator; 0 releases allocations after stream completion.",
    )
    parser.add_argument("target", type=Path, help="Python runner to execute")
    parser.add_argument("target_args", nargs=argparse.REMAINDER, help="Arguments passed to the target runner")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if not args.experimental_allow_global_allocator:
        raise SystemExit("pass --experimental-allow-global-allocator to start a child with this prototype")
    if args.allocator_cache_mib < 0:
        raise SystemExit("--allocator-cache-mib cannot be negative")

    library = args.library.resolve(strict=True)
    target = args.target.resolve(strict=True)
    python = args.python.resolve(strict=True)
    source_root = Path(__file__).resolve().parents[1] / "src"

    environment = os.environ.copy()
    environment["DDO_VMM_LIBRARY_PATH"] = str(library)
    environment["DDO_VMM_TORCH_CACHE_MIB"] = str(args.allocator_cache_mib)
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source_root) if not existing_python_path else str(source_root) + os.pathsep + existing_python_path

    command = [str(python), "-c", BOOTSTRAP, str(target), *args.target_args]
    completed = subprocess.run(command, env=environment, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
