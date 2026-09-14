"""Attach one real Diffusers transformer block to the DDO VMM arena.

This probe intentionally stops before a model forward.  It validates checkpoint
loading, parameter names, BF16 shapes and persistent views without changing the
DDO production runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--pattern", default=r"transformer_blocks\.0$")
    parser.add_argument("--resident-budget-gb", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()

    import torch
    from diffusers import AutoModel

    from ddo_vmm_allocator.native import load_native_vmm
    from ddo_vmm_allocator.torch_module_adapter import VmmModuleAdapter
    from ddo_vmm_allocator.weight_arena import VmmWeightArena

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = AutoModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    pattern = re.compile(args.pattern)
    matches = [name for name, _module in model.named_modules() if pattern.search(name)]
    if not matches:
        raise RuntimeError(f"no transformer modules matched {args.pattern!r}")
    library = load_native_vmm(args.library)
    arena = VmmWeightArena(library, resident_budget_bytes=int(args.resident_budget_gb * 2**30))
    adapter = VmmModuleAdapter(arena)
    attached = adapter.attach_module_tree(model, args.pattern)
    print("Diffusers transformer VMM attach probe: PASS")
    print(f"  model: {args.model_path}")
    print(f"  matched modules: {len(matches)}")
    print(f"  attached weight groups: {len(attached)}")
    print(f"  resident budget: {args.resident_budget_gb:.2f} GiB")
    adapter.close()
    arena.close()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
