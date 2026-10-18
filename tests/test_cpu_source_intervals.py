"""CPU-only retained-source metadata; no library, CUDA allocation or transfers."""

from types import SimpleNamespace

import pytest
import torch

from ddo_vmm_allocator.model_arena_adapter import ModelArenaModuleAdapter, ModelArenaError


def make_adapter(bindings):
    adapter = ModelArenaModuleAdapter(SimpleNamespace())
    adapter._bindings = [SimpleNamespace(tensors=bindings)]
    return adapter


def test_cpu_sources_observed_without_retaining_or_copying_tensors(monkeypatch):
    source = torch.arange(8, dtype=torch.bfloat16)
    binding = SimpleNamespace(file_slice=None, source=source, nbytes=16)
    adapter = make_adapter([binding, SimpleNamespace(file_slice=object())])
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA forbidden"))
        patch.setattr(torch.Tensor, "clone", lambda *args, **kwargs: pytest.fail("copy forbidden"))
        result = adapter.cpu_source_intervals()
    assert result == {"schema_version": 1, "binding_count": 1,
                      "intervals": [(source.data_ptr(), 16)]}
    assert binding.source is source


@pytest.mark.parametrize("source,nbytes", [(torch.ones(4), 99),
                                          (torch.ones(4, 4).t(), 64),
                                          (torch.ones(0), 0), (object(), 4)])
def test_unknown_source_layout_refused(source, nbytes):
    adapter = make_adapter([SimpleNamespace(file_slice=None, source=source, nbytes=nbytes)])
    with pytest.raises(ModelArenaError):
        adapter.cpu_source_intervals()
