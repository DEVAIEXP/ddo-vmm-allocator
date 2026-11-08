"""Experimental execution-unit level contracts with CPU tensors; never initialize CUDA."""

from types import SimpleNamespace

import pytest
import torch

from ddo_vmm_allocator import model_arena_adapter as adapter_module
from ddo_vmm_allocator.model_arena_adapter import (
    _discover_linear_layout, _ObservedExecutionQueue, ModelArenaModuleAdapter,
    configure_experimental_execution_unit_level, experimental_execution_unit_level,
)


@pytest.fixture(autouse=True)
def restore_level():
    previous = experimental_execution_unit_level()
    yield
    configure_experimental_execution_unit_level(previous)


def block():
    item = torch.nn.Module()
    item.add_module("attn", torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)))
    item.add_module("ff", torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Linear(16, 4)))
    item.add_module("proj", torch.nn.Linear(4, 4))  # Linear that is a direct child.
    item.add_module("norm", torch.nn.LayerNorm(4))   # Not selected.
    return item


def model(count=3):
    root = torch.nn.Module()
    root.add_module("layers", torch.nn.ModuleList([block() for _ in range(count)]))
    root.add_module("output", torch.nn.Linear(4, 2))
    return root


def store_for(layout):
    return SimpleNamespace(virtual_bytes=layout[2], page_bytes=512, _file_slices=None, _host_registry=None,
                           _source_stats={"cpu_source_bindings": 0, "cpu_source_bytes": 0},
                           _streaming_enabled=False, stable_tensor=lambda **kw: (kw["source"].clone(), None))


def unit_sizes(units):
    sizes = []
    for unit in units:
        size = 0
        for binding in unit.tensors:
            size = ((size + 511) & ~511) + binding.nbytes
        sizes.append(size)
    return sizes


def test_default_level_is_block_and_unknown_level_refused():
    assert experimental_execution_unit_level() == "block"
    with pytest.raises(ValueError):
        configure_experimental_execution_unit_level("linear")
    assert experimental_execution_unit_level() == "block"


def test_block_child_discovery_uses_direct_children_by_structure():
    root = model()
    block_layout = _discover_linear_layout(root, r".*")
    configure_experimental_execution_unit_level("block_child")
    child_layout = _discover_linear_layout(root, r".*")
    assert child_layout[0] == block_layout[0]          # Same selection.
    assert child_layout[2] == block_layout[2]          # Same payload and alignment.
    assert {u for u, _ in block_layout[1].values()} == {"layers.0", "layers.1", "layers.2"}
    expected = {f"layers.{i}.{c}" for i in range(3) for c in ("attn", "ff", "proj")}
    assert {u for u, _ in child_layout[1].values()} == expected
    by_name = dict(root.named_modules())
    assert all(module is by_name[unit] for unit, module in child_layout[1].values())
    assert "output" not in child_layout[1]             # Non-list modules stay individual.


def test_list_item_that_is_a_linear_stays_its_own_unit():
    root = torch.nn.Module()
    root.add_module("layers", torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(2)]))
    configure_experimental_execution_unit_level("block_child")
    units = _discover_linear_layout(root, r".*")[1]
    assert {u for u, _ in units.values()} == {"layers.0", "layers.1"}


def attach(root, layout, monkeypatch):
    adapter = ModelArenaModuleAdapter(store_for(layout))
    installed = []
    monkeypatch.setattr(adapter, "_register_execution_unit", installed.append)
    report = adapter.attach_linear_modules(root, r".*", _layout=layout)
    return adapter, installed, report


def test_block_child_attachment_shrinks_the_largest_unit(monkeypatch):
    root = model()
    block_adapter, block_units, block_report = attach(root, _discover_linear_layout(root, r".*"), monkeypatch)
    for handle in block_adapter._handles:
        handle.remove()
    assert block_report.execution_units == 4 and block_adapter._execution_queue.learn is False
    configure_experimental_execution_unit_level("block_child")
    root2 = model()
    child_adapter, child_units, child_report = attach(root2, _discover_linear_layout(root2, r".*"), monkeypatch)
    try:
        assert child_report.execution_units == 3 * 3 + 1
        assert [u.module_name for u in child_units[:3]] == ["layers.0.attn", "layers.0.ff", "layers.0.proj"]
        assert child_units[-1].module_name == "output"
        assert child_report.payload_bytes == block_report.payload_bytes
        assert max(unit_sizes(child_units)) < max(unit_sizes(block_units))
        assert sum(len(u.tensors) for u in child_units) == sum(len(u.tensors) for u in block_units)
        assert child_adapter._execution_queue.learn is True
    finally:
        for handle in child_adapter._handles:
            handle.remove()


def states(count):
    return [SimpleNamespace(name=str(i)) for i in range(count)]


def test_learned_order_replaces_enumeration_order():
    a, b, c = states(3)
    queue = _ObservedExecutionQueue([a, b, c], learn=True)
    queue.begin_pass()                                  # Learning pass.
    assert [queue.consume(s) for s in (c, a, b)] == [None, None, None]
    assert queue.valid
    queue.begin_pass()
    assert queue.learned and not queue.learn and queue.states == [c, a, b]
    assert queue.consume(c) is a and queue.consume(a) is b and queue.consume(b) is None
    queue.begin_pass()
    assert queue.consume(a) is None and not queue.valid  # Deviation disables look-ahead.


@pytest.mark.parametrize("sequence", [(0, 1, 1, 2), (0, 2)])
def test_repeated_or_incomplete_pass_keeps_enumeration_order(sequence):
    units = states(3)
    queue = _ObservedExecutionQueue(list(units), learn=True)
    queue.begin_pass()
    for index in sequence:
        assert queue.consume(units[index]) is None
    queue.begin_pass()
    assert queue.learn_failed and not queue.learned and not queue.learn
    assert queue.states == units
    assert queue.consume(units[0]) is units[1]          # Existing behavior afterwards.


def test_block_level_queue_never_learns():
    a, b = states(2)
    queue = _ObservedExecutionQueue([a, b])
    queue.begin_pass()
    assert queue.consume(a) is b                        # Look-ahead from the first pass, as before.
    queue.begin_pass()
    assert queue.states == [a, b] and not queue.learned


def test_auxiliary_states_outside_the_queue_are_ignored_while_learning():
    a, b = states(2)
    other = SimpleNamespace(name="aux")
    queue = _ObservedExecutionQueue([a, b], learn=True)
    queue.begin_pass()
    assert queue.consume(other) is None and queue.consume(a) is None and queue.consume(b) is None
    queue.begin_pass()
    assert queue.learned and queue.states == [a, b]


def test_no_cuda_initialized():
    assert not torch.cuda.is_initialized()
    assert adapter_module.experimental_execution_unit_level() in ("block", "block_child")
