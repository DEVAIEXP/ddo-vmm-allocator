# Allocator guidance

## Documentation Language

- Write all new or updated repository documentation in English, including README files, AGENTS.md, documentation pages, handoffs, plans, benchmark commentary, and explanatory examples or docstrings.
- Preserve verbatim quotations, raw logs, benchmark prompts, and required license or attribution text in their original form; write surrounding explanations in English. Do not change recorded evidence merely to translate it.
- This requirement applies to repository documentation, not to conversations with the user, which may remain in the user's preferred language.

## Scope

This subproject provides the native CUDA Driver VMM library and the PyTorch
adapter consumed by DDO. This is an independent implementation; do not copy
code from external projects.

## Target architecture

- `native/`: a model-sized virtual range, on-demand page materialization,
  per-generation signatures, unpinning and native LRU eviction.
- `src/ddo_vmm_allocator/model_arena_adapter.py`: receives the actual weight
  size, creates the single range and installs fault/copy hooks on dense modules.
- `weight_arena.py` and `torch_module_adapter.py`: the legacy `linear_runtime`
  path remains supported for validated presets and experiments; do not remove it.
- The native path uses `backing=device`; under WDDM the driver may place physical
  pages in shared memory under pressure, without switching to PyTorch's allocator.
- Persistent sources and the registered cache belong to C++ in this submodule.
  Sources retain descriptors and native bytes, not loader CPU tensors. Do not
  introduce a parallel safetensors cache or Python group heuristics: residency,
  admission and pressure belong to the native runtime.
- Host sources and GPU arenas have distinct lifetimes. A retained handle keeps
  the file alive after its original owner closes. Closing an owner requires
  serialization with calls on that same handle; retained owners allow concurrent
  reads. The future cache may recycle bytes only after I/O and every referencing
  DMA have completed.
- In the persistent buffer (config v5 / ABI v8), host commitment and CUDA
  registration have separate ceilings and lifetimes. Unregistration preserves
  bytes; subsequent DMA can register them again without rereading the file.
  Only host pressure/clear removes content. Registration policy and its events
  belong to C++.
- The persistent path uses two native execution slots charged against the
  existing GPU ceiling and immutable batch descriptors. Recycling requires the
  last consumer's event; cancellation requires the upload event. Partial failure
  does not release ownership without proven completion. Python submits one unit,
  rather than managing requests/DMAs per window; views may be reused.

## Safety rules

1. Never deallocate/remap a region while a stream may reference it.
2. Materialization failure must fall back to a temporary CUDA buffer and restore
   parameters only after the relevant stream synchronization/event.
3. Native pressure eviction must consider only unpinned pages; the virtual range
   remains fully faultable after eviction.
4. Every lifecycle change needs a small CUDA test before any LTX benchmark.

## Required tests

```powershell
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build\Release\ddo_vmm.dll'
& 'E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe' -m pytest tests -q
```

Present the command, hypothesis and approval criterion and wait for confirmation
before any CUDA test or benchmark. `tests/test_native_source.py` tests only
files/ownership, without initializing CUDA. Use DDO's `.venv-vmm` environment for
both projects. Rebuild the DLL when changing `native/`; record actual gate
results rather than treating historical test counts as the current baseline.

## Integration

To resume on Linux, read `../docs/linux-handoff.md` and DDO's Linux plan.
`scripts/build_linux.py --mode cpu` builds isolated contracts without CUDA;
`--mode library` only builds the experimental library. Neither authorizes CUDA
tests. Use DDO's `.venv-linux` and `build-linux`, preserving Windows environments
and binaries. A POSIX source does not imply portable cache/DMA support.

Current Linux checkpoint: `../docs/native-linux-transport-gate.md`. Transport,
mmap storage/cache/batches compiled; CPU tests and the probe/seven approved CUDA
contracts passed. The corrected library also passed three CUDA cache contracts
and one real staged generation with native encoder/transformer; see
`../docs/native-linux-staged-gate.md`. New CUDA tests and model executions still
require separate confirmation. Preserve commit/cgroup v2/process limits, with
no automatic growth or WDDM emulation. Unregistration must preserve bytes;
ownership/events remain native.

Hosts use DDO's public API, which consumes `attach_model_linear_modules`.
During source gates, preserve the active 4 GiB budget and approved staging;
the host cache has a separate ceiling and does not increase the GPU budget.
`linear_runtime` remains the stable route and must not silently redirect to the
native arena. `vmm_runtime` is the protected NVMe baseline.

## Regression prohibition

Do not remove fallback, multiple-block or host-tensor tests. Do not run the
complete model if CUDA tests fail.
