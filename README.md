# DDO VMM allocator
Native CUDA Driver VMM and Python adapters consumed by
[DDO](https://github.com/DEVAIEXP/diffusers-dynamic-offloader).
This repository owns native arenas, source/cache transport, event lifetimes,
pressure/admission, the PyTorch pluggable allocator and accounting. It contains
no model-specific runners or weights.
## Documentation
- [User installation](docs/user/installation.md)
- [Developer architecture and contracts](docs/developer/contracts.md)
- [Maintainer builds](docs/developer/builds.md)
- [Tests and validation](docs/developer/validation.md)
- [Experimental package delivery](docs/developer/releases.md)
- [Agent instructions](AGENTS.md)
## Current scope
Selected Windows, native Linux and WSL workloads passed bounded contracts.
Native VMM remains experimental; arbitrary models/drivers, macOS and driver-loss
recovery are not promised. Stable DDO linear/NVMe paths must remain untouched.
Optional Windows/Linux wheels can embed hash-selected DLL/ELF artifacts; Linux
selected installed allocator/trim gates passed on Windows, native Linux and WSL,
not broad ABI, full-model or arbitrary CUDA delivery acceptance.
Editable Python installation does not compile native code.
No binary is built/downloaded during import or inference.
For local native wheels, follow the [installation guide](docs/user/installation.md)
and the linked DDO wheel-pair procedure. Package assembly and GitHub publication
are separate; do not assume a prerelease is available remotely.
Historical probe scaffolding and investigations are preserved locally in ignored
`.internal/documentation/`; current contracts do not depend on those files.
Model benchmarks belong to [DDO examples](https://github.com/DEVAIEXP/ddo-examples).
## License
Apache-2.0. Preserve existing copyright/license/attribution notices in source and
distribution. Driver/toolkit components retain their own applicable terms.
