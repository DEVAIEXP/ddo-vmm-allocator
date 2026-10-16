# Installation boundary
The Python adapter and native library are separate delivery concerns. Ordinary DDO
linear users do not need this allocator. Native users need compatible CUDA-enabled
PyTorch, the system NVIDIA driver and a reviewed native DLL/.so.
A compatible packaged wheel allows DDO bootstrap to discover the library without
a user-supplied path. Reviewed local Windows and Ubuntu 22.04-baseline Linux wheels
passed separate selected installed allocator/trim gates on Windows, native Linux
and WSL. This is not full-model or arbitrary-platform acceptance, nor a claim that
these prereleases have been published to GitHub or PyPI.
Editable/source Python installation does not compile the library. No build/download
at import or inference is supported. For explicit source builds see
[maintainer builds](../developer/builds.md); do not overwrite a validated artifact.
DDO owns native setup before CUDA and public component lifecycle integration.
Selected Windows/Linux/WSL compatibility is not general platform/model certification.

For checksum verification, clean environments, compatible Torch installation and
the paired DDO `vmm` dependencies, follow the
[DDO installation procedure](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/user/installation.md).
Select the wheel matching the OS/architecture; do not reuse a host-built newer-glibc
library on an older distribution. Native wheels need the system NVIDIA driver,
not an end-user compiler/toolkit. Video/audio model dependencies belong to examples.
