# Maintainer builds
Requires a real C++20 compiler, CMake >=3.25 and reviewed CUDA toolkit headers/
driver link support. The default native target uses CUDA Driver API; optional
experimental execution kernels add separate CUDA compilation requirements.
Use new candidate directories, never overwrite accepted Windows/Linux/WSL builds.
An illustrative compilation-only Linux span-pool configuration:
```bash
cmake -S . -B build-linux-candidate \
  -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/path/to/reviewed/cuda-toolkit \
  -DDDO_VMM_BUILD_PROBES=OFF -DDDO_VMM_NATIVE_SPAN_POOL=ON \
  -DDDO_VMM_NATIVE_JOINED_SPANS=OFF -DDDO_VMM_NATIVE_SOURCE_DISPATCHER=OFF \
  -DDDO_VMM_SOURCE_TEST_HOOKS=OFF -DDDO_VMM_ALLOCATION_DIAGNOSTICS=OFF \
  -DDDO_VMM_BUILD_EXECUTION_OPS=OFF -DDDO_VMM_SEGMENTED_DRIVER=ON \
  -DDDO_VMM_AUTO_SEGMENTED=ON -DDDO_VMM_TIMING_COUNTERS=ON
cmake --build build-linux-candidate --parallel 2
```
`DDO_VMM_SEGMENTED_DRIVER` and `DDO_VMM_AUTO_SEGMENTED` require the span pool;
automatic selection also requires the segmented driver. They build the cold
configuration exports that `install_pytorch_vmm_allocator(segmented_auto=...,
growable_reserve_bytes=...)` and the DDO bootstrap defaults require; a library
built without them is refused when that configuration is requested.
Windows uses the reviewed MSVC/CMake configuration; selected joined-span artifacts
have different features, so do not assume this Linux recipe reproduces them.
`DDO_VMM_MAPPING_TRACE=ON` is a Windows-only opt-in diagnostic build feature.
It does not enable recording automatically; use a separate candidate directory.
Its internal ABI is not a supported Python residency-control interface.

The optional checked PyTorch bridge is a separate CMake project in
`native/torch_bridge` (MSVC Windows x64). Configure a new candidate directory
with `-DDDO_VMM_BUILD_TORCH_BRIDGE=ON` and explicit `TORCH_ROOT` (the target
environment's `site-packages/torch`), `PYTHON_ROOT`, `PYTHON_LIBRARY` and
`CUDA_ROOT` (headers only). It always builds `checked_allocation_cpu`, a CPU
policy test without Torch or CUDA. Record the Torch version/git revision,
CPython tag, toolchain and module SHA256 in its identity JSON.

Build helpers have fixed paths: inspect before running. Record source revision/
dirty state, features, compiler/toolkit, architecture, exports, dependencies and hash.
Do not bundle host NVIDIA drivers. WSL cannot necessarily load a newer native-Linux
glibc/C++ binary; build for the supported ABI without upgrading the user's system
runtime to accommodate an artifact. No CUDA test is authorized by compilation.
