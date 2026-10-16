# Allocator repository guidance
## Scope and starting points
This independent repository owns native CUDA Driver VMM and Python adapters,
consumed through DDO. It must remain architecture/model agnostic.
Read README.md and docs/developer/{contracts,builds,validation,releases}.md.
If attached as DDO's submodule, read the parent AGENTS.md too. Standalone work
must not require parent-relative historical handoffs or ignored local records.
## Language and evidence
Write all new/updated documentation, comments and explanatory docstrings in English.
Conversation may use the user's language. Preserve verbatim logs, prompts, quotes
and required license/attribution text; never alter evidence to translate or sanitize it.
Use docs/user for current integration, docs/developer for contributor contracts,
and ignored .internal/documentation for handoffs/plans/intermediate investigations.
README must index public docs. Never link public instructions to required ignored files.
Keep a separate backup of local internal records; .gitignore is not a backup and
does not remove previously tracked content from Git history. No implicit history rewrite.
## Code map and contracts
- native/src and native/include: C ABI, arena/pool, source/ring/cache/batches,
  pressure/admission, events and accounting.
- src/ddo_vmm_allocator: native bindings, source/tensor ownership and adapters.
- scripts: explicit build/diagnostic/probe tools, never automatic runtime tuning.
- tests and native/tests: CPU and explicitly approved hardware contracts.
Native backing is device memory; driver WDDM pressure behavior is not emulated on
Linux. Legacy adapters remain compatibility paths and must not be removed.
Sources retain immutable identity/descriptors and positioned Windows/POSIX reads.
Keep owners across async I/O; serialize close with calls on the same handle.
Persistent host storage is native bytes, not loader tensors. Storage commitment,
registration and GPU residency ceilings remain independent and fixed.
Unregister preserves cached bytes. Recycling requires last DMA/compute completion.
Immutable batches/tickets retain source/arena ownership through partial failure.
Pressure evicts only unpinned/idle backing. Never unmap/free a stream-referenced
region. Temporary CUDA fallback restores state only after the correct event.
Failed CUDA completion must retain unsafe ownership, not pretend driver recovery.
Linux observes overcommit mode/cgroup/process/physical bounds conservatively.
Canonical cgroup mount identity must not weaken limits. WSL registration observations
are workload-specific and cannot automatically become defaults or raised ceilings.
## Builds and delivery
Use a real compiler and explicitly reviewed CMake features. Build directories must
be isolated candidates; build helpers have fixed output paths and need inspection.
Record source revision/dirty state, toolchain, ABI, features, exports and SHA256.
CMake uses Driver API; optional execution kernels are separate experiments.
Windows/Linux/WSL are separate artifact/validation identities. Do not retag an
incompatible ELF as manylinux or upgrade a user's glibc to load our binary.
No compiler/native download at import or inference. Driver libraries remain
system prerequisites. Assemble only explicitly selected/hash-verified artifacts.
Windows and host-Linux hash-selected wheel assembly exist. Installed CUDA and
older-distro portability remain separate gates; host tags are not manylinux claims.
Public native adapter dependencies remain optional for ordinary DDO users.
## Validation and integrations
Run targeted CPU contracts without CUDA; never count a skip as hardware acceptance.
Before lifecycle changes reach model benchmarks, require approved small native
exact-payload/ownership tests. Preserve fallback/multi-block/host-tensor coverage.
DDO controls fresh-child bootstrap, generic configuration and public metrics.
Keep host scheduling/residency heuristics out of Python runners. Quantized payload
transport must retain backend math; optional execution kernels are not compatibility.
When authorized, commit this repo first and update the DDO gitlink separately.
## Execution, validation and release authority
Read this file and relevant public docs before acting. Inspect git status and preserve
user changes. Use apply_patch for manual edits; scope searches/edits to the task.
Before CUDA, benchmarks, benchmark parameter changes, compatibility removal or
history cleanup, show exact command, hypothesis, fixed limits and criteria and
wait for user approval. CPU tests/builds do not authorize GPU execution.
Stop at the first failure/anomaly. No silent retries, budget growth, driver resets,
OS memory-policy changes, binary promotion or full Cartesian test matrix.
Never claim general support from selected acceptance. Treat missing fields as
unavailable, not zero; preserve raw result identity and rejected experiments.
Experimental prereleases precede stable release. Tags, push, Hub/registry upload,
CI publication and Git-history rewrites require explicit permission.
Do not remove required licenses/attribution or suggest that hiding names establishes
independent implementation/license compatibility. Do not commit user scratch files,
weights, environments, build products, outputs or internal records.
