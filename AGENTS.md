# Allocator guidance

## Scope

This subproject provides the native CUDA Driver VMM library and the PyTorch
adapter used by the LTX host. This is an independent implementation; do not copy
code from external projects.

## Already validated architecture

- `native/`: virtual reservation, `cuMemCreate`, map/unmap, eviction and DXGI
  guards.
- `src/ddo_vmm_allocator/weight_arena.py`: residency budget and packed groups.
- `src/ddo_vmm_allocator/torch_module_adapter.py`: per-block hooks with isolated
  state and safe temporary fallback.
- The recommended path uses `backing=device` for physical pages resident in VRAM.
- Reuse persistent host transport from generic DDO; do not implement a second
  safetensors cache.

## Safety rules

1. Never deallocate/remap a region while a stream may reference it.
2. Materialization failure must fall back to a temporary CUDA buffer and restore
   parameters only after the relevant stream synchronization/event.
3. Native pressure eviction is disabled when the DXGI guard is configured;
   the Python policy must choose a safe eviction.
4. Every lifecycle change needs a small CUDA test before any LTX benchmark.

## Required tests

```powershell
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build\Release\ddo_vmm.dll'
& 'E:\ProjetosIA\ltx2.3-image-diffusers\.venv\Scripts\python.exe' -m pytest tests -q
```

The current baseline is `4 passed`. Rebuild the DLL using the user's MSVC prompt
when changing `native/`.

## Next authorized task

Connect the existing registered host transport from generic DDO to
`VmmWeightArena`, retaining `backing=device`. Do not use `backing=host`
as a production solution: the test showed almost no shared WDDM usage
and high cost. Do not change the runner to create a new cache.

## Regression prohibition

Do not remove fallback, multiple-block or host-tensor tests. Do not run the
complete model if CUDA tests fail.
