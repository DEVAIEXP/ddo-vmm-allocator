"""Structural discovery contracts with CPU tensors; never initialize CUDA."""

from types import SimpleNamespace
import re

import pytest
import torch

from ddo_vmm_allocator.model_arena_adapter import (
    _discover_linear_layout, ModelArenaModuleAdapter, ModelArenaWeightStore,
)


def model_with_list(name, count=12):
    root = torch.nn.Module()
    root.add_module(name, torch.nn.ModuleList([
        torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 3))
        for _ in range(count)
    ]))
    root.add_module("output", torch.nn.Linear(3, 2))
    return root


@pytest.mark.parametrize("pattern", [r"transformer_blocks", r"^transformer_blocks\.2$", r"\.0$", r".*", r"^$", r"missing"])
def test_selection_and_alignment_match_previous_contract(pattern):
    root = model_with_list("transformer_blocks")
    matcher = re.compile(pattern)
    matched = {n for n, _ in root.named_modules() if n and matcher.search(n)}
    expected = [(n, m) for n, m in root.named_modules() if isinstance(m, torch.nn.Linear)
                and (matcher.search(n) or any(n == p or n.startswith(p + ".") for p in matched))]
    payload = 0
    for _, module in expected:
        for parameter in module.parameters(recurse=False):
            alignment = max(parameter.element_size(), 256)
            payload = ((payload + alignment - 1) // alignment) * alignment
            payload += parameter.numel() * parameter.element_size()
    selected, _, actual_payload = _discover_linear_layout(root, pattern)
    assert selected == tuple(expected)
    assert actual_payload == payload


def test_renamed_lists_preserve_numeric_order_and_unit_geometry():
    original = _discover_linear_layout(model_with_list("transformer_blocks"), r".*")
    renamed = _discover_linear_layout(model_with_list("layers"), r".*")
    assert original[2] == renamed[2]
    units = list(dict.fromkeys(unit for unit, _ in renamed[1].values()))
    assert units == [f"layers.{i}" for i in range(12)]
    assert "output" not in renamed[1]


def test_nested_unet_like_structure_uses_enclosing_list_item_not_class_name():
    root = torch.nn.Module()
    block = model_with_list("attention_layers", count=2)
    block.add_module("conv", torch.nn.Conv2d(4, 4, 3))
    root.add_module("down_blocks", torch.nn.ModuleList([block]))
    selected, units, _ = _discover_linear_layout(root, r".*")
    assert len(selected) == 5
    assert {name for name, _ in units.values()} == {"down_blocks.0"}
    assert all(module is block for _, module in units.values())
    assert not any("conv" in name for name, _ in selected)


def test_sequential_and_root_linear_remain_individual_units():
    assert _discover_linear_layout(torch.nn.Sequential(torch.nn.Linear(2, 2)), r".*")[1] == {}
    selected, units, size = _discover_linear_layout(torch.nn.Linear(2, 2), r"^$")
    assert selected[0][0] == "" and not units and size > 0


def test_fixed_geometry_queried_once_but_residency_not_cached():
    class Native:
        reads = 0

        @property
        def info(self):
            self.reads += 1
            return SimpleNamespace(address=4096, page_size=512, virtual_size=8192,
                                   resident_size=self.reads)

    store = ModelArenaWeightStore.__new__(ModelArenaWeightStore)
    store.native = Native()
    for _ in range(10):
        assert (store.address, store.page_bytes, store.virtual_bytes) == (4096, 512, 8192)
    assert store.native.reads == 1
    assert store.native.info.resident_size == 2


def test_prepared_layout_attaches_without_rediscovery_or_leaf_hooks(monkeypatch):
    root = model_with_list("layers", count=3)
    layout = _discover_linear_layout(root, r".*")
    store = SimpleNamespace(virtual_bytes=layout[2], page_bytes=512,
                            _file_slices=None, _host_registry=None,
                            _source_stats={"cpu_source_bindings": 0, "cpu_source_bytes": 0},
                            _streaming_enabled=False,
                            stable_tensor=lambda **kw: (kw["source"].clone(), None))
    adapter = ModelArenaModuleAdapter(store)
    installed = []
    monkeypatch.setattr(adapter, "_register_execution_unit", installed.append)
    monkeypatch.setattr(root, "named_modules", lambda: pytest.fail("layout rediscovered"))
    report = adapter.attach_linear_modules(root, r".*", _layout=layout)
    assert report.execution_units == 4
    assert [unit.module_name for unit in installed] == ["layers.0", "layers.1", "layers.2", "output"]
    assert [len(unit.tensors) for unit in installed] == [4, 4, 4, 2]
    assert report.payload_bytes == layout[2]
    assert installed[0].prefetch_next is installed[1]
    assert installed[2].prefetch_next is None
    for handle in adapter._handles:
        handle.remove()
