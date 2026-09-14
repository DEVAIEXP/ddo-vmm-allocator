"""Synthetic two-layer validation for the explicit VMM module adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=1024)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    import torch

    from ddo_vmm_allocator.native import load_native_vmm
    from ddo_vmm_allocator.torch_module_adapter import VmmModuleAdapter
    from ddo_vmm_allocator.weight_arena import VmmWeightArena

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dimension = args.dimension
    first = torch.nn.Linear(dimension, dimension, bias=False, device="cpu", dtype=torch.bfloat16)
    second = torch.nn.Linear(dimension, dimension, bias=False, device="cpu", dtype=torch.bfloat16)
    source_first = first.weight.detach().clone()
    source_second = second.weight.detach().clone()
    model = torch.nn.Sequential(first, second).to("cuda")
    library = load_native_vmm(args.library)
    # CUDA VMM rounds each region to allocation granularity (2 MiB on this
    # device), so leave enough budget for one rounded group.
    one_group_bytes = source_first.numel() * source_first.element_size()
    arena = VmmWeightArena(library, resident_budget_bytes=max(one_group_bytes * 4, 4 * 2**20))
    adapter = VmmModuleAdapter(arena)
    adapter.attach_module_parameters(model[0], "block0")
    adapter.attach_module_parameters(model[1], "block1")
    input_tensor = torch.randn(1, dimension, device="cuda", dtype=torch.bfloat16)
    output = model(input_tensor)
    torch.cuda.synchronize()
    if output.shape != input_tensor.shape or not torch.isfinite(output).all():
        raise AssertionError("adapter produced an invalid output")
    arena.collect_completed_events(wait=True)
    adapter.close()
    arena.close()
    print("VMM module adapter probe: PASS")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  layers: 2, dimension: {dimension}")
    print(f"  resident budget: {arena.resident_budget_bytes / 2**20:.2f} MiB")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    run(parse_args())
