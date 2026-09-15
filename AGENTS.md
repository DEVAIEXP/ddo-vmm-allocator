# Allocator guidance

## Scope

This subproject provides the native CUDA Driver VMM library and the PyTorch
adapter used by the LTX host. This is an independent implementation; do not copy
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
- Do not introduce a parallel safetensors cache or Python group heuristics:
  residency and pressure belong to the native arena.

## Safety rules

1. Never deallocate/remap a region while a stream may reference it.
2. Materialization failure must fall back to a temporary CUDA buffer and restore
   parameters only after the relevant stream synchronization/event.
3. Native pressure eviction must consider only unpinned pages; the virtual
   range remains fully faultable after eviction.
4. Every lifecycle change needs a small CUDA test before any LTX benchmark.

## Required tests

```powershell
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build\Release\ddo_vmm.dll'
& 'E:\ProjetosIA\ltx2.3-image-diffusers\.venv\Scripts\python.exe' -m pytest tests -q
```

The current baseline is `4 passed`. Rebuild the DLL using the user's MSVC prompt
when changing `native/`.

## Integration

The DDO runner must use `attach_model_linear_modules` and pass the residency
budget explicitly (or `None` to leave pressure to the driver).
`linear_runtime` remains the compatibility route and must not silently
redirect to the native arena.

## Regression prohibition

Do not remove fallback, multiple-block or host-tensor tests. Do not run the
complete model if CUDA tests fail.
