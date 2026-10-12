"""Small native convolution/decode-method gate; explicit CUDA approval required."""

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import tempfile
import weakref


def run(library):
    os.environ["DDO_VMM_LIBRARY_PATH"] = str(library)
    os.environ["DDO_VMM_TORCH_CACHE_MIB"] = "512"
    import torch
    from ddo_vmm_allocator.torch_allocator import install_pytorch_vmm_allocator

    install_pytorch_vmm_allocator(library)  # Before integration imports or CUDA allocation.

    from safetensors.torch import load_file, save_file
    from diffusers_dynamic_offloader import (
        DynamicOffloadConfig, enable_offload, from_pretrained_with_dynamic_offload,
        remove_dynamic_offload,
    )

    class DecodeComponent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Conv2d(2, 2, 1, dtype=torch.bfloat16)
            self.blocks = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Conv2d(2, 2, 3, padding=1, dtype=torch.bfloat16),
                    torch.nn.GroupNorm(1, 2, dtype=torch.bfloat16),
                )
            ])
            self.interrupt = False

        def forward(self, value):
            raise AssertionError("The root forward must not be substituted for decode")

        def decode(self, value):
            value = self.projection(value)
            for block in self.blocks:
                value = block(value)
            if self.interrupt:
                raise KeyboardInterrupt("intentional decode cancellation")
            return value

    torch.manual_seed(42)
    config = DynamicOffloadConfig(
        execution_mode="vmm_fast_runtime", vmm_native_transport=True,
        vmm_component_tensor_ownership=True, vmm_component_buffer_limit_mb=0,
        vmm_resident_weight_budget_gb=0.07, vmm_staging_buffer_gb=2 / 1024,
        vmm_page_mb=2, vmm_source_cache_gb=64 / 1024,
        vmm_source_registered_gb=64 / 1024,
    )
    with tempfile.TemporaryDirectory(prefix="ddo-convolution-probe-") as directory:
        path = Path(directory) / "model.safetensors"
        original = DecodeComponent().eval()
        save_file({name: value.detach() for name, value in original.named_parameters()}, str(path))
        reference = copy.deepcopy(original).cuda()
        samples = [torch.randn(1, 2, size, size, dtype=torch.bfloat16, device="cuda") for size in (4, 8)]
        with torch.inference_mode():
            expected = [reference.decode(sample).cpu() for sample in samples]
        del reference, original
        gc.collect()

        def loader(folder):
            module = DecodeComponent().eval()
            module.load_state_dict(load_file(str(Path(folder) / "model.safetensors")))
            return module

        loaded = from_pretrained_with_dynamic_offload(
            directory, model_loader=loader, dynamic_offload_config=config, apply_dynamic=False,
        )
        module = loaded.module
        source = weakref.ref(module.projection.weight)
        forward = type(module.projection).forward
        result = enable_offload(
            module, component="vae", route="dynamic_offload", config=config,
            execution_device="cuda", use_environment=False,
        )
        try:
            gc.collect()
            assert source() is None, "Loader CPU weights retained"
            assert type(module.projection).forward is forward
            for _ in range(2):
                for sample, target in zip(samples, expected, strict=True):
                    with torch.inference_mode():
                        actual = module.decode(sample).cpu()
                    torch.testing.assert_close(actual, target, rtol=0, atol=0)
            report = result.metrics_snapshot(collect_completed=True).as_dict()
            stats = report["runtime"]["vmm"]
            assert stats["fast_cpu_source_bindings"] == 0
            assert stats["fast_file_backed_bindings"] == 6
            for key in ("fast_fault_failures", "fast_temporary_tensors",
                        "fast_source_cache_admission_denials", "fast_source_cache_bypasses"):
                assert stats[key] == 0, (key, stats[key])
            assert report["runtime"]["completion_collection"]["tickets_after"] == 0
            module.interrupt = True
            try:
                with torch.inference_mode():
                    module.decode(samples[-1])
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("Cancellation was not exercised")
        finally:
            remove_dynamic_offload(module)
        assert all(parameter.device.type == "meta" for parameter in module.parameters())
        assert all(not child._forward_hooks and not child._forward_pre_hooks for child in module.modules())
        print(json.dumps({"status": "PASS", "numerical_comparison": "exact", "report": report}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    run(parser.parse_args().library.resolve(strict=True))
