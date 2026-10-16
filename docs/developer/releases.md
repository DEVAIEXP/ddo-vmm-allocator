# Native package delivery
Experimental prereleases are planned before stable distribution. Do not advertise
an unuploaded package or claim arbitrary-model/platform compatibility.
Current Hatch build hook assembles an existing Windows AMD64 DLL only when
`DDO_VMM_WHEEL_LIBRARY` and `DDO_VMM_WHEEL_SHA256` explicitly select verified bytes.
It records a manifest and uses a platform wheel tag. Plain wheels contain no native
binary. Hash-selected Linux x86-64 ELF assembly and installed CPU discovery now
exist; older-distro ABI portability and installed CUDA delivery remain work.
Reuse one explicit CMake recipe locally and in CI; record compiler/toolkit/features,
revision, ABI requirements and hash. Build candidates first, validate the actual
wheel, and publish unchanged tested bytes only after explicit approval.
Keep system CUDA driver libraries external. Earn manylinux tags through dependency/
symbol validation rather than retagging a newer distro's binary. Avoid unnecessary
Python/Torch-specific wheel multiplication unless the actual native ABI requires it.
Maintain reviewed DDO adapter/allocator API pairings. Tags, push, registry upload
and promotion are explicit release actions, never implicit documentation tasks.

## Linux candidate assembly
The same paired `DDO_VMM_WHEEL_LIBRARY` / `DDO_VMM_WHEEL_SHA256` selection accepts
ELF64 little-endian x86-64 shared objects. It snapshots the verified bytes under
ddo_vmm_allocator/native/linux/libddo_vmm.so with artifact.json and tags the wheel
py3-none-linux_x86_64. `DDO_VMM_WHEEL_ABI_REPORT` optionally embeds an audit tied
to that binary/platform; mismatched evidence is refused. Windows defaults/tags
and plain Python wheels remain unchanged. The hook never compiles or loads CUDA.

The parent DDO maintainer command generates a paired prerelease from isolated
source snapshots, compiles the native CPU contracts/library, audits ELF and checks
the installed wheels. See its [release recipe](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/developer/releases.md).
Standalone assembly remains possible with `python -m build --wheel` in an
appropriate build-tools environment and the explicit artifact/hash selection.
No parent-local handoff is required. Host-built wheels are not manylinux claims.

Initial host-Linux rc1 development assembly passed CPU checks. The library was
byte-identical to the accepted cgroup candidate; the local ABI requires GLIBC 2.38,
GLIBCXX 3.4.31 and CXXABI 1.3.15 symbols. It is not an Ubuntu/WSL 22.04 package.
CUDA wheel acceptance and publication are separate steps; nothing was uploaded.
