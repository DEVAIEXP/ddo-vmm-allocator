"""CPU contracts for transient setup mapping; no CUDA queries or allocation."""

from types import SimpleNamespace

import pytest
import torch

from ddo_vmm_allocator import model_arena_adapter as module


def make_store(monkeypatch, *, refuse_span=False, refuse_all=False):
    calls = []

    def prepare(offset, size):
        calls.append(("prepare", offset, size))
        if refuse_all or (refuse_span and size > 16):
            raise module.NativeVmmError("view prepare refused")

    store = module.ModelArenaWeightStore.__new__(module.ModelArenaWeightStore)
    store.native = SimpleNamespace(
        prepare_view_range=prepare,
        release_view_range=lambda offset, size: calls.append(("release", offset, size)),
    )
    store._runtime_stats = {}
    store._buffers = []
    store._backing = "device"
    store._arena_layout = (4096, 512, 8192)

    def view(address, size, **kwargs):
        calls.append(("view", address - 4096, size))
        return torch.zeros(size, dtype=torch.uint8), object()

    monkeypatch.setattr(module, "cuda_uint8_tensor_from_address", view)
    return store, calls


def tensor(store, offset):
    return store.stable_tensor(offset=offset, source=torch.ones(4))


def test_one_prepare_release_per_unit_and_same_tensor_offsets(monkeypatch):
    store, calls = make_store(monkeypatch)
    with store.view_setup(((0, 272), (512, 16))):
        for offset in (0, 256, 512):
            stable, signature = tensor(store, offset)
            assert stable.shape == (4,) and signature is None
    assert calls == [("prepare", 0, 272), ("view", 0, 16), ("view", 256, 16),
                     ("release", 0, 272), ("prepare", 512, 16),
                     ("view", 512, 16), ("release", 512, 16)]
    assert store._runtime_stats["view_prepare_calls"] == 2
    assert store._runtime_stats["view_release_calls"] == 2
    assert store._runtime_stats["view_create_calls"] == 3
    assert store._view_setup_state is None


def test_view_failure_releases_unit_before_propagating(monkeypatch):
    store, calls = make_store(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("DLPack creation failed")

    monkeypatch.setattr(module, "cuda_uint8_tensor_from_address", fail)
    with pytest.raises(RuntimeError, match="DLPack"):
        with store.view_setup(((0, 272),)):
            tensor(store, 0)
    assert calls == [("prepare", 0, 272), ("release", 0, 272)]
    assert store._view_setup_state is None


def test_refused_span_preserves_individual_tensor_path(monkeypatch):
    store, calls = make_store(monkeypatch, refuse_span=True)
    with store.view_setup(((0, 272),)):
        tensor(store, 0)
        tensor(store, 256)
    assert calls == [("prepare", 0, 272), ("prepare", 0, 16), ("view", 0, 16),
                     ("release", 0, 16), ("prepare", 256, 16),
                     ("view", 256, 16), ("release", 256, 16)]


def test_refused_individual_view_preserves_cpu_source(monkeypatch):
    store, calls = make_store(monkeypatch, refuse_all=True)
    source = torch.ones(4)
    with store.view_setup(((0, 16),)):
        stable, signature = store.stable_tensor(offset=0, source=source)
    assert stable is source and signature is None
    assert not any(call[0] in {"view", "release"} for call in calls)


def test_release_failure_prevents_successful_attach(monkeypatch):
    store, _ = make_store(monkeypatch)

    def fail(*args):
        raise module.NativeVmmError("release failed")

    store.native.release_view_range = fail
    with pytest.raises(module.NativeVmmError, match="release failed"):
        with store.view_setup(((0, 16),)):
            tensor(store, 0)


def test_layout_mismatch_releases_existing_mapping(monkeypatch):
    store, calls = make_store(monkeypatch)
    with pytest.raises(module.ModelArenaError, match="layout"):
        with store.view_setup(((0, 16), (256, 16))):
            tensor(store, 0)
            tensor(store, 512)
    assert ("release", 0, 16) in calls
    assert not any(call[0] == "view" and call[1] == 512 for call in calls)


def test_nested_setup_is_rejected_without_releasing_outer_mapping(monkeypatch):
    store, calls = make_store(monkeypatch)
    with store.view_setup(((0, 16),)):
        tensor(store, 0)
        with pytest.raises(module.ModelArenaError, match="nested"):
            with store.view_setup(((0, 16),)):
                pass
        assert not any(call[0] == "release" for call in calls)
    assert calls[-1] == ("release", 0, 16)


def test_outside_scope_retains_existing_prepare_release(monkeypatch):
    store, calls = make_store(monkeypatch)
    tensor(store, 0)
    assert calls == [("prepare", 0, 16), ("view", 0, 16), ("release", 0, 16)]


def test_structural_spans_preserve_aligned_payload_without_tensor_owners():
    root = torch.nn.Module()
    root.add_module("layers", torch.nn.ModuleList([
        torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 3))
        for _ in range(2)
    ]))
    selected, units, payload = module._discover_linear_layout(root, r".*")
    spans = module._layout_view_spans(selected, units)
    assert len(spans) == 2
    assert sum(spans[-1]) == payload
    assert all(isinstance(value, int) for span in spans for value in span)


def test_setup_time_scopes_include_failed_calls(monkeypatch):
    store, _ = make_store(monkeypatch)
    ticks = iter((1.0, 1.2, 2.0, 2.3, 3.0, 3.4))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))

    def fail(*args, **kwargs):
        raise RuntimeError("view failed")

    monkeypatch.setattr(module, "cuda_uint8_tensor_from_address", fail)
    with pytest.raises(RuntimeError, match="view failed"):
        with store.view_setup(((0, 16),)):
            tensor(store, 0)
    assert store._runtime_stats["view_prepare_seconds"] == pytest.approx(0.2)
    assert store._runtime_stats["view_create_seconds"] == pytest.approx(0.3)
    assert store._runtime_stats["view_release_seconds"] == pytest.approx(0.4)
