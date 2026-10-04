"""Platform admission uses the same native bridge; no CUDA or file reads."""
from types import SimpleNamespace

import pytest

from ddo_vmm_allocator import model_arena_adapter as adapter


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_supported_platforms_use_identical_native_geometry(monkeypatch, platform):
    monkeypatch.setattr(adapter, "sys", SimpleNamespace(platform=platform))
    calls = []
    ring = SimpleNamespace(source_stats={"workers": 2})

    def create_ring(size, slots, *, device_ordinal):
        calls.append((size, slots, device_ordinal))
        return ring

    transport = adapter._NativeShardTransport(
        SimpleNamespace(create_host_ring=create_ring),
        slot_size_bytes=128 * 2**20, slots=2, device_ordinal=0,
    )
    assert calls == [(64 * 2**20, 4, 0)]
    assert transport.ring is ring
    assert transport.cache is None


def test_other_platforms_refuse_before_native_allocation(monkeypatch):
    monkeypatch.setattr(adapter, "sys", SimpleNamespace(platform="darwin"))
    with pytest.raises(adapter.ModelArenaError, match="native Linux"):
        adapter._NativeShardTransport(SimpleNamespace(), slot_size_bytes=1024, slots=1, device_ordinal=0)
