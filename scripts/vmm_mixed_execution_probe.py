"""Explicit small CUDA gate for mixed file/CPU execution-buffer payloads."""

import argparse
from array import array
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


def run(library):
    # Test this checkout's adapter against an explicitly selected unchanged DLL.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    os.environ["DDO_VMM_TORCH_CACHE_MIB"] = "64"
    import torch
    from ddo_vmm_allocator.native import load_native_vmm
    from ddo_vmm_allocator.torch_allocator import install_pytorch_vmm_allocator
    from ddo_vmm_allocator.model_arena_adapter import (
        ModelArenaWeightStore, NativeExecutionBuffer, NativeExecutionBuffers, _TensorBinding,
    )

    if torch.cuda.is_initialized():
        raise RuntimeError("This gate requires a fresh process before CUDA initialization")
    install_pytorch_vmm_allocator(library)
    native = load_native_vmm(library)
    mib = 1024**2
    # The pool carves two 2 MiB slots out of this ceiling and requires headroom.
    # The parent arena stays unmapped in this gate; only the execution slots run.
    model_budget, slot_bytes = 6*mib, 2*mib
    with tempfile.TemporaryDirectory(prefix="ddo-mixed-execution-", ignore_cleanup_errors=True) as directory:
        print(f"Temporary source directory: {directory}", flush=True)
        path = Path(directory) / "base.bin"
        with path.open("wb") as output:
            array("f", [0.125] * (64 * 64)).tofile(output)
        base = torch.full((64, 64), 0.125)
        a, b = torch.full((4, 64), 0.125), torch.full((64, 4), 0.25)
        value = torch.ones(2, 64, device="cuda")
        expected_base = torch.nn.functional.linear(torch.ones(2, 64), base)
        expected_mixed = expected_base + 0.8 * torch.nn.functional.linear(
            torch.nn.functional.linear(torch.ones(2, 64), a), b)
        store = ModelArenaWeightStore(
            native, payload_bytes=model_budget, resident_budget_bytes=model_budget,
            native_transport_slot_size_bytes=mib, native_transport_slots=2,
            native_source_cache_slot_bytes=mib, native_source_cache_slots=2,
            native_source_cache_process_limit_bytes=2*mib,
            native_source_cache_model_windows=True,
            native_source_cache_registered_limit_bytes=2*mib,
        )

        def binding(source, label, file_backed=False):
            descriptor = SimpleNamespace(path=path, offset_bytes=0) if file_backed else None
            return _TensorBinding(None, "weight", source, descriptor, source, 0,
                                  source.numel()*source.element_size(), label=label)

        mixed = [binding(base, "base", True), binding(a, "adapter.A"), binding(b, "adapter.B")]
        controls = [[mixed[0]], mixed, [binding(base, "host.base")]]
        checks = 0
        for buffer_type in (NativeExecutionBuffer, NativeExecutionBuffers):
            buffer = buffer_type(store, slot_bytes)
            for bindings in controls:
                pending = []
                for _ in range(3):
                    result = buffer.upload(bindings)
                    views, ready = (result.views, result.ready) if isinstance(buffer, NativeExecutionBuffers) else result
                    store.wait_for_compute(ready)
                    output = torch.nn.functional.linear(value, views[0][1])
                    if len(views) == 3:
                        output = output + 0.8 * torch.nn.functional.linear(
                            torch.nn.functional.linear(value, views[1][1]), views[2][1])
                    pending.append(output)
                    stream = torch.cuda.current_stream()
                    if isinstance(buffer, NativeExecutionBuffers):
                        buffer.finish_compute(result, stream)
                    else:
                        buffer.finish_compute(stream)
                expected = expected_mixed if len(bindings) == 3 else expected_base
                for output in pending:
                    torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)
                    checks += 1
            torch.cuda.current_stream().synchronize()
            buffer.close()
            print(f"{buffer_type.__name__}: PASS ({checks} cumulative checks)", flush=True)
        store.close()
    print(json.dumps({"status": "PASS", "checks": checks,
        "scope": "mixed-source execution buffers, not full-model LoRA acceptance",
        "library": str(library), "adapter": __import__(
            "ddo_vmm_allocator.model_arena_adapter", fromlist=["__file__"]).__file__}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--run-cuda", action="store_true")
    args = parser.parse_args()
    if not args.run_cuda:
        parser.error("Pass --run-cuda only after approving this hardware gate")
    run(args.library.resolve(strict=True))
