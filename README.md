# DDO VMM Allocator

An Apache-2.0 DDO CUDA Virtual Memory Management (VMM) allocator project for PyTorch model-weight paging.

This repository is intentionally separate from `diffusers-dynamic-offloader`.  Its purpose is to provide a reusable, testable allocator runtime which DDO may later consume as an optional backend.

## Goals

- Stable CUDA virtual addresses for model-weight tensors.
- Explicit physical backing, mapping, residency and eviction through the CUDA Driver VMM API.
- A policy layer for demand loading that is independent of Diffusers and model architecture.
- Windows and Linux support for the CUDA VMM core.
- A future, small PyTorch allocator adapter isolated from the portable VMM core.

## Non-goals

- No code, names, layouts, constants or implementation details are taken from GPL projects.
- No use of `cudaMallocManaged` as a weight-compute backend.
- No promise that CUDA VMM alone changes PyTorch's own allocator behavior.  That requires the later native adapter.

## Architecture

```
CUDA driver VMM                 portable Python policy
cuMemAddressReserve        ->  virtual address arena
cuMemCreate                ->  physical-page pool
cuMemMap/cuMemSetAccess    ->  residency manager
cuMemUnmap/cuMemRelease    ->  eviction manager

future native PyTorch adapter -> allocator integration and WDDM interaction
```

The core API will be implemented through `cuda-python` (`cuda.bindings`), which is Apache-2.0.  The project will first validate the VMM lifecycle with isolated probes before exposing a PyTorch allocator integration.

## Platform support

The VMM lifecycle is designed for CUDA on Windows and Linux.  Both platforms support the CUDA Driver VMM API.  File transport and native allocator loading are platform-specific adapters; they must not leak into the core paging policy.

## Development status

October 10, 2026 delivery update: an opt-in wheel build can assemble an existing,
explicitly hash-selected AMD64 DLL under `ddo_vmm_allocator/native/win32/`.
The installed DDO bootstrap discovers it automatically; normal adapter builds
remain pure Python and never compile/download a DLL at execution time. The local
Windows artifact passed packaging and isolated CPU discovery checks, not a new
CUDA execution gate. FP8 payload/ownership changes are Python-only; selected
Windows DLL and Linux .so hashes remain unchanged. See the parent checkout's
`docs/native-wheel-delivery.md` for assembly, installation and acceptance details.
The older core-only/future-adapter wording below is historical, not current scope.

Current native Linux checkpoint: source/ring/persistent-storage/cache/batches
compile into `build-linux/libddo_vmm.so` with CPU contracts passed; the approved
allocator probe and seven small CUDA contracts also passed. The corrected library
then passed three cache contracts and one selected native encoder/transformer
staged generation (1.050970s/step); [manifest/limits](../docs/native-linux-staged-gate.md).
Linux remains experimental outside that bounded workload. Use the parent DDO
`.venv-linux` / optional `vmm-dev-linux` profile and its
[build/evidence guide](../docs/native-linux-transport-gate.md). Preserve Windows
builds. The older core-probe description below is historical, not current scope.

The portable C++ library now contains a first CUDA Driver VMM probe: it creates a device-local physical allocation, maps it into a reserved virtual address, grants access, writes through the mapping, and tears it down.  It still does not install a PyTorch allocator.  That separation is deliberate: a Python-only package cannot safely replace PyTorch's native caching allocator.

## Windows build and probe

Open the x64 MSVC developer prompt, then run:

```powershell
cd E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator
cmake -S . -B build -DDDO_VMM_BUILD_PROBES=ON
cmake --build build --config Release
.\build\Release\ddo_vmm_driver_probe.exe 64
```

If CMake selects a single-config generator such as Ninja, the probe path is instead:

```powershell
.\build\ddo_vmm_driver_probe.exe 64
```

The probe writes a known byte pattern through the mapped virtual address.  It is intentionally small and independent of PyTorch.

## Python bridge probe

With the host Python environment, call the DLL through the explicit opaque-handle API:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m ddo_vmm_allocator.native --library .\build\Release\ddo_vmm.dll --megabytes 64
```

The bridge owns the VMM region with a context manager, validates an
evict/rematerialize cycle at the same virtual address, and deliberately does
not expose an unmapped raw address as a PyTorch tensor yet.

## Experimental PyTorch allocator probe

The native DLL also exports the two function pointers accepted by PyTorch's
`CUDAPluggableAllocator`.  This is an isolated validation of allocator ABI and
VMM pointer usability, not the final paging allocator.  It creates one VMM
region per PyTorch allocation and must be run in a new Python process:

```powershell
$env:PYTHONPATH = "$PWD\src"
python .\scripts\vmm_torch_allocator_probe.py --library .\build\Release\ddo_vmm.dll --megabytes 64
```

Do not use this prototype for inference: it intentionally has no caching,
stream-recording, pressure policy, or model-weight paging yet.

Set a bounded cache in MiB to reuse free VMM mappings in the isolated probe:

```powershell
python .\scripts\vmm_torch_allocator_probe.py --library .\build\Release\ddo_vmm.dll --megabytes 64 --cache-mib 128
```

The cache is exact-process state.  On a new-allocation failure it is drained
before one retry, so idle cached regions cannot turn a potential allocation
into an immediate OOM. Stream-aware reuse in the global PyTorch allocator
remains pending; model-weight paging is handled separately by the weight arena.

## Weight-residency arena

`VmmWeightArena` is the framework-neutral resident-weight policy layer: every group owns
a stable virtual address, but only groups inside its configured resident budget
hold physical VRAM backing.  It evicts only groups that the caller has unpinned.

```powershell
$env:PYTHONPATH = "$PWD\src"
python .\scripts\vmm_weight_arena_probe.py `
    --library .\build\Release\ddo_vmm.dll `
    --group-megabytes 32 `
    --resident-megabytes 64
```

The first arena probe is intentionally framework-free. A caller must not
unpin a group until all CUDA work using it has completed; the tensor adapter
provides `release_after_current_stream()` and `collect_completed_events()` for
asynchronous CUDA completion.

## Persistent PyTorch tensor probe

The tensor adapter creates one CUDA tensor while a VMM region is mapped.  It
then evicts and rematerializes backing without recreating the tensor object or
changing its device pointer:

```powershell
$env:PYTHONPATH = "$PWD\src"
python .\scripts\vmm_persistent_tensor_probe.py `
    --library .\build\Release\ddo_vmm.dll `
    --megabytes 32
```

Data is intentionally discarded by eviction; model code must repopulate a
weight group before submitting kernels that use its persistent tensor view.

The native API also exposes an experimental `host` backing kind for Driver VMM
experiments.  CUDA reports those mappings as CPU DLPack tensors, so they are
not valid PyTorch CUDA weight views and are intentionally not used by the
transformer adapter.  The model path uses device-backed VMM allocations; a
large materialized CUDA virtual set is what allows WDDM to page backing under
Windows.

## PyTorch module adapter probe

`VmmModuleAdapter` attaches a CPU-backed parameter to a module with explicit
pre-forward population and post-forward CUDA-event release.  This is the first
framework integration layer; it is not the global PyTorch allocator and does
not depend on any GPL implementation:

```powershell
$env:PYTHONPATH = "$PWD\src"
python .\scripts\vmm_module_adapter_probe.py `
    --library .\build\Release\ddo_vmm.dll `
    --dimension 512
```

The probe uses two BF16 linear layers and a one-group resident budget.  The
second layer forces a safe eviction/reload of the first layer while preserving
the virtual tensor addresses.  Only after this probe passes should a model
specific transformer adapter be added.

For recursive transformer-style blocks, the tree adapter can be checked with:

```powershell
python .\scripts\vmm_module_tree_probe.py `
    --library .\build\Release\ddo_vmm.dll `
    --dimension 256
```

The first real-model attach check is deliberately forward-free:

```powershell
python .\scripts\vmm_diffusers_transformer_attach_probe.py `
    --model-path elismasilva/ltx2.3-image-distilled-1.1 `
    --library .\build\Release\ddo_vmm.dll `
    --pattern 'transformer_blocks\.0$' `
    --resident-budget-gb 1
```

## Tensor weight arena probe

The tensor-aware arena binds CPU weights, reloads them whenever their VMM group
is materialized, and delays unpinning with a CUDA event:

```powershell
$env:PYTHONPATH = "$PWD\src"
python .\scripts\vmm_tensor_weight_arena_probe.py `
    --library .\build\Release\ddo_vmm.dll `
    --megabytes 16
```

## Fresh-process runner launcher

The launcher installs the same experimental allocator before importing the
target runner.  It changes only its child process and requires an explicit
acknowledgement:

```powershell
python .\scripts\run_with_vmm_allocator.py `
    --library .\build\Release\ddo_vmm.dll `
    --python E:\ProjetosIA\ltx2.3-image-diffusers\.venv\Scripts\python.exe `
    --experimental-allow-global-allocator `
    E:\ProjetosIA\ltx2.3-image-diffusers\run_dynamic_modular_vmm_distilled.py --help
```

Use this launcher only with a small, isolated probe until the allocator gains
stream-aware reuse and an explicit pressure/eviction policy.
