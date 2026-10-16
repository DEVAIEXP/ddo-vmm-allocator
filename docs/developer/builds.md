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
  -DDDO_VMM_BUILD_EXECUTION_OPS=OFF
cmake --build build-linux-candidate --parallel 2
```
Windows uses the reviewed MSVC/CMake configuration; selected joined-span artifacts
have different features, so do not assume this Linux recipe reproduces them.
Build helpers have fixed paths: inspect before running. Record source revision/
dirty state, features, compiler/toolkit, architecture, exports, dependencies and hash.
Do not bundle host NVIDIA drivers. WSL cannot necessarily load a newer native-Linux
glibc/C++ binary; build for the supported ABI without upgrading the user's system
runtime to accommodate an artifact. No CUDA test is authorized by compilation.
