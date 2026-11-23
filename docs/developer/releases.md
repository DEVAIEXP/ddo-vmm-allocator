# Native package delivery

## GitHub release drafts

After an accepted version is published on PyPI, run **Actions → Prepare reviewed
PyPI release draft** from `main`, supplying the version and full allocator source
SHA used to build those wheels. The workflow uses DDO's shared draft tooling and
the allocator repository's scoped `GITHUB_TOKEN`; it needs no PyPI credential.
It verifies the existing PyPI wheel hashes and metadata, generates categorized
notes, and attaches the unchanged wheels, checksums and manifest to a draft.
It does not build binaries, execute CUDA, publish the draft or move existing tags.

Review source identity and platform acceptance before publishing the draft in
GitHub Releases. That explicit publication creates `v<version>` at the selected
allocator commit. Publish DDO's corresponding draft separately using its own
build revision and matching allocator gitlink. Existing drafts/tags are refused;
inspect partial drafts after failures rather than rerunning blindly.
See the [shared release procedure](https://devaiexp.github.io/diffusers-dynamic-offloader/docs/developer/releases/#github-release-drafts).

## Candidate assembly
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

A Windows wheel can also include the optional checked PyTorch allocation bridge:
set both `DDO_VMM_WHEEL_TORCH_BRIDGE` (the built `.pyd`) and
`DDO_VMM_WHEEL_TORCH_BRIDGE_IDENTITY` (its JSON identity: abi, torch_version,
torch_git, python, platform, module, sha256). The hook verifies the module name,
SHA256 and PE architecture, snapshots both into `native/win32/` and never builds
or loads them. Without these variables Windows wheels are unchanged. The module
is specific to one PyTorch build and CPython ABI; other environments keep the
plain allocator registration at runtime.

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
