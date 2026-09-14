"""Validate recursive transformer-block style attachment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def run(library: Path, dimension: int) -> None:
    import torch

    from ddo_vmm_allocator.native import load_native_vmm
    from ddo_vmm_allocator.torch_module_adapter import VmmModuleAdapter
    from ddo_vmm_allocator.weight_arena import VmmWeightArena

    class BlockStack(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(
                [torch.nn.Linear(dimension, dimension, bias=False) for _ in range(2)]
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            for block in self.transformer_blocks:
                value = block(value)
            return value

    model = BlockStack().to(device="cuda", dtype=torch.bfloat16)
    native = load_native_vmm(library)
    arena = VmmWeightArena(native, resident_budget_bytes=16 * 2**20)
    adapter = VmmModuleAdapter(arena)
    attached = adapter.attach_module_tree(model, r"transformer_blocks\.\d+$")
    output = model(torch.randn(1, dimension, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    if not torch.isfinite(output).all() or len(attached) != 2:
        raise AssertionError("recursive adapter validation failed")
    arena.collect_completed_events(wait=True)
    adapter.close()
    arena.close()
    print("VMM module tree probe: PASS")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  attached groups: {len(attached)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=256)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    run(args.library, args.dimension)
