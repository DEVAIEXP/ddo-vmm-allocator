"""Optional integration helpers for Diffusers-style transformer modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .torch_module_adapter import VmmModuleAdapter
from .weight_arena import VmmWeightArena


@dataclass(frozen=True)
class TransformerAttachReport:
    pattern: str
    attached_groups: int
    adapter: VmmModuleAdapter


def attach_transformer_blocks(
    transformer: Any,
    arena: VmmWeightArena,
    *,
    module_pattern: str = r"transformer_blocks\.\d+$",
    recursive: bool = True,
) -> TransformerAttachReport:
    """Attach matching transformer blocks to an explicit VMM arena.

    The function only relies on the standard ``torch.nn.Module`` traversal API,
    so Diffusers remains an optional caller dependency.  It deliberately does
    not install a global PyTorch allocator or remove existing Diffusers hooks.
    """
    adapter = VmmModuleAdapter(arena)
    names = adapter.attach_module_tree(transformer, module_pattern, recursive=recursive)
    return TransformerAttachReport(
        pattern=module_pattern,
        attached_groups=len(names),
        adapter=adapter,
    )
