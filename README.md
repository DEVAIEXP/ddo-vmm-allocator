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

The bridge owns the VMM region with a context manager and deliberately does not expose the raw virtual address as a PyTorch tensor yet.
