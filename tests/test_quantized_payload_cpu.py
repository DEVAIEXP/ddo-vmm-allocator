"""Optional CPU-only composite weight contracts; no native allocator needed."""

from dataclasses import fields
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torchao")
from torchao.quantization import Float8WeightOnlyConfig, quantize_

# Isolate this CPU contract from the existing package's eager adapter imports.
_module_spec = importlib.util.spec_from_file_location(
    "ddo_quantized_payload_cpu_gate",
    Path(__file__).parents[1] / "src/ddo_vmm_allocator/quantized_payload.py",
)
_module = importlib.util.module_from_spec(_module_spec)
sys.modules[_module_spec.name] = _module
_module_spec.loader.exec_module(_module)
describe_float8_weight_only_cpu = _module.describe_float8_weight_only_cpu


@pytest.fixture(autouse=True)
def no_cuda():
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
        yield
    assert not torch.cuda.is_initialized()


def make_weight():
    model = torch.nn.Linear(32, 16, bias=False, dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    quantize_(model, Float8WeightOnlyConfig(set_inductor_config=False))
    return model.weight


def test_reconstruction_preserves_payload_identity_and_backend_output():
    weight = make_weight()
    spec = describe_float8_weight_only_cpu(weight)
    assert spec.physical_bytes == 576
    assert spec.logical_dtype == torch.bfloat16
    # The contract has no loader tensors or a closure over the original weight.
    assert not any(isinstance(getattr(spec, field.name), torch.Tensor) for field in fields(spec))
    payloads = {"qdata": weight.qdata.clone(), "scale": weight.scale.clone()}
    restored = spec.reconstruct_cpu(payloads)
    assert restored.qdata is payloads["qdata"] and restored.scale is payloads["scale"]
    inputs = torch.randn(4, 32, dtype=torch.bfloat16)
    torch.testing.assert_close(torch.nn.functional.linear(inputs, restored),
                               torch.nn.functional.linear(inputs, weight), rtol=0, atol=0)


@pytest.mark.parametrize("change", ["missing", "extra", "dtype", "shape", "stride", "gradient"])
def test_invalid_reconstruction_refused_without_mutating_source(change):
    weight = make_weight()
    spec = describe_float8_weight_only_cpu(weight)
    pointer = weight.qdata.data_ptr()
    payloads = {"qdata": weight.qdata, "scale": weight.scale}
    if change == "missing":
        payloads.pop("scale")
    elif change == "extra":
        payloads["other"] = torch.ones(1)
    elif change == "dtype":
        payloads["qdata"] = weight.qdata.to(torch.float32)
    elif change == "shape":
        payloads["scale"] = weight.scale.flatten()
    elif change == "stride":
        payloads["qdata"] = torch.empty((32, 16), dtype=weight.qdata.dtype).t()
    else:
        payloads["scale"] = weight.scale.clone().requires_grad_(True)
    with pytest.raises(ValueError):
        spec.reconstruct_cpu(payloads)
    assert weight.qdata.data_ptr() == pointer


def test_shared_storage_refused():
    weight = make_weight()
    spec = describe_float8_weight_only_cpu(weight)
    backing = torch.zeros(576, dtype=torch.uint8)
    payloads = {"qdata": backing[:512].view(torch.float8_e4m3fn).view(16, 32),
                "scale": backing[512:].view(torch.float32).view(16, 1)}
    with pytest.raises(ValueError, match="Shared"):
        spec.reconstruct_cpu(payloads)


@pytest.mark.parametrize("change", ["plain", "activation", "block", "scale", "metadata", "kernel"])
def test_unknown_recipes_fail_closed(change):
    weight = make_weight()
    if change == "plain":
        weight = torch.ones(16, 32)
    elif change == "activation":
        weight.act_quant_kwargs = object()
    elif change == "block":
        weight.block_size = [16, 32]
    elif change == "scale":
        weight.scale = weight.scale.to(torch.bfloat16)
    elif change == "kernel":
        weight.kernel_preference = object()
    else:
        original = weight.__tensor_flatten__
        def unknown_metadata():
            names, metadata = original()
            return names, dict(metadata, unknown=True)
        weight.__tensor_flatten__ = unknown_metadata
    with pytest.raises(ValueError):
        describe_float8_weight_only_cpu(weight)


def test_unverified_backend_version_refused():
    weight = make_weight()
    with patch.object(_module, "version", return_value="0.19.0"):
        with pytest.raises(ValueError, match="0.18.0"):
            describe_float8_weight_only_cpu(weight)
