# Validation

## Allocation recovery diagnostics

The segmented pool keeps recovered allocation-pressure notices quiet by default.
Enable `DDO_SHOW_RECOVERY_WARNS=1` before bootstrap to display them when using a
binary with `ddo_vmm_set_recovery_warns_v1`. Older binaries report that the control
is unavailable. Final allocation failures remain visible in either mode, and
capacity receipts remain available independently of console visibility. An initial
Driver or materialization refusal can be recovered by the capacity transaction.
Its final summary distinguishes `status=recovered` (the allocation succeeded and
execution can continue) from `status=failed` (the allocation was not satisfied).
The preceding detailed record preserves the original Driver statuses and page
counts. A recovered allocation does not certify the rest of the inference.

## Execution backing integration: selected Windows lifecycle acceptance

The optional execution-slot reclaim integration adds CPU contracts for pending
completion, partial failure, pressure thresholds and registry/close lifetime.
The selected Windows integrated DLL SHA256
`41a16e8608ad5e4d950624fd73e55348e2c52b122028d6c7f4a09e7f58906dd1`
passed an explicit small reclaim/recreation gate: two 4 MiB buffers, 8 MiB
physical backing, 24 MiB VA, one 4 MiB release/recreation, two streams and up to
three events. Physical usage fell from 8 to 4 MiB and returned to 8 MiB.
Leased and pending slots refused reclamation; both virtual addresses and the
control payload were preserved, the replacement upload was exact, and close
completed with exit zero and empty stderr.

The same DLL also passed the registered allocation-pressure callback gate with
a synthetic UINT64_MAX headroom margin, without exhausting memory. Four calls
covered disabled, leased, pending and completed states. Receipts recorded three
enabled queries/triggers, two busy decisions, one 4 MiB release and one 4 MiB
recreation, with no unavailable observations or failures. Exact payloads, stable
addresses and safe close passed. This tests callback behavior under a forced
decision, not selection of a useful production threshold or Torch allocation
scheduling in a model.

These gates do not establish model correctness, performance, physical local-VRAM
placement, installed-wheel or Linux acceptance. Prior isolated experimental model
results must not be attributed to this binary. Next gates cover Linux lifecycle
and approved model/installed-package validation.
Windows DXGI automatic pressure behavior is not a Linux policy.

## CPU contract coverage

Native CPU tests cover immutable sources, host storage and Linux admission.
Python CPU tests cover adapters, versioned ABI handling and packaging/discovery.
The allocation-attempt CPU contract injects success, out-of-memory and other
driver failures. It checks exact returned results, eviction counts, guarded
refusal, exhausted recovery and stopping when an OOM recovery encounters a
different failure. The failure-trace contract independently checks copied error
text. These contracts do not execute CUDA or validate a model/installed artifact.
The host-admission CPU contract checks injected RAM/commit observations, exact
fits, configured reserves, overflow boundaries and unavailable observations. On
Windows it checks that error 1455 is copied before a later OS error overwrites it.
Selected Windows compilation and these CPU contracts do not validate actual
host-ring CUDA registration, model execution or Linux execution.
The budget-admission CPU contract checks overflow-safe budget arithmetic and
injected allowed, insufficient and unavailable observations. It checks that an
unavailable observation stops recovery without another eviction, including after
an insufficient observation. It does not query DXGI or prove physical placement.
Selected Windows native admission candidate SHA256
`7555676c83ce14716565bc5def481829551869642ea28011959df64ebb2c893f`
passed a bounded gate: a 2 MiB VMM payload verified before eviction and after
new upload at the same address, one 64 KiB registered host slot, 64 KiB CPU
verification buffer, and deterministic refusal under an impossible system reserve.
The process exited 0 with empty stderr and observed zero final resident backing.
Event/stream release was checked; void region/ring destruction only has returned-call
receipts, not checked destruction status. The gate explicitly retained its CUDA
context through resource teardown. Its predecessor crashed during close and is
rejected. Query-unavailable behavior remains CPU-injected evidence. This is not
model, performance, local-VRAM placement or installed-wheel acceptance.
The same bounded gate also passed for the composed experimental pressure variant
SHA256 `65b3b94ee3bcf19b3334a42b892a3cbaa3a090fb6de86ed2b42a2a5f970143a7`.
That artifact preserves joined spans and the existing experimental pressure API;
eight CPU contracts passed before the GPU gate. This selected gate does not
establish model acceptance or validate the complete pressure policy under load.
A selected Windows two-step diagnostic using that composed artifact subsequently
exited 0, completed both 48-block transformer forwards and scheduler steps, and
reported checked model/compute closure. Both complete video/audio latent outputs
were finite and bitwise equal to the retained reference. Residual CPU source
detachment reduced observed private commit from 51,377,942,528 to 13,360,537,600
bytes. The requested standby purge was skipped because the process token lacked
the required privilege; the user authorized completion as a diagnostic only.
Its 426.76-second observed two-step interval is not matched performance evidence.
This does not establish broad model, placement or installed-wheel support.
The optional Windows mapping observer has selected CPU-only validation: its
injected-call contract preserves driver results, refuses stop during an active
call, retains bounded records with an explicit dropped count, and rejects invalid
snapshots. Its disabled stub returns unavailable without modifying output.
Both enabled and disabled library configurations compiled in isolated Windows
candidate directories. No GPU capture has validated this observer. Mapping API
timestamps do not establish consumer completion, allocation role or local-VRAM
placement; owner/generation attribution remains unavailable in this first layer.
An additional CPU-injected owner-observation contract checks canonical lookup,
generation changes, unknown interior addresses, invalid extents, unchanged output
on error and explicit unknown role/version/completion. A separate bounded Python
producer has CPU coverage for re-querying identity, finite storage, query failure
and thread/device/stream refusal. These tests do not validate actual CUDA pool
lookup during model execution or supply a complete model-access trace.
Hardware contracts cover exact payloads, event/slot ownership, registration recycling,
fallback and idle trim. Before CUDA, require exact command/limits/criteria approval.
Never interpret a platform skip as a pass or extend a gate after failure.
Selected Linux cgroup-root correction passed native CPU, isolated compilation,
fresh-process idle trim and three source-cache contracts. Six diagnostic search
CPU contracts passed; the search imports no CUDA in its parent.
The tested corrected Linux library SHA256 is
`5a49e0a6acd672186af116ee31fe1409bd33c605b2846bfc5f9d4ae42d79907c`.
Selected WSL corrected candidate SHA256 is
`5915cca767c89ab7606c04f2001c9eddffa762386c852cc9f517708a2d03584c`.
Acceptance belongs to each artifact/environment; changing build features or compiler
requires new bounded evidence. Source-checkout tests are not installed-wheel tests.
See [DDO validation](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/developer/validation.md)
for selected technical acceptance and examples for model conditions/results.

## Selected installed-wheel WSL execution backing validation

The rc3 development Linux wheel library SHA256
`71183e987bb8b1d6a252c71ab938589ee89ac13f1e48339cb184731569d55535`
passed an explicit execution-buffer reclaim/recreation gate in WSL2 Ubuntu 22.04.
The user executed the gate; preserved evidence reports observed exit 0, empty
stderr and the complete expected protocol. Two 4 MiB buffers remained within
8 MiB physical/24 MiB virtual payload addresses, with required 2 MiB pages,
64 KiB host verification, two streams and at most three events. One completed
idle buffer released 4 MiB, reducing physical backing from 8 to 4 MiB; acquisition
recreated it at the same virtual address and restored the total to 8 MiB.
Leased and pending buffers refused reclaim. The control payload and new upload
were exact, and checked pool close completed. The gate allowed 25 s natively/
40 s in the controller, with no retry or automatic termination.

The standalone Linux probe SHA256 is
`3c7e79862264d5fd97af10b5d7da32693d3a7470daa5913fd8cda0719c727c75`.
It was built separately in the GPU-free Ubuntu 22.04/CUDA 13.2 Docker toolchain;
the installed wheel library was not rebuilt for the gate. Preparation passed
36 protocol/preflight and six simulated controller cases without CUDA.
This validates explicit reclaim for this installed artifact on WSL, not native
Linux, model performance, physical VRAM placement or automatic pressure policy.
The optional automatic DXGI trigger remains Windows-only and disabled by default.

The same installed rc3 ELF subsequently completed selected LTX text-prompt image
workloads on native Linux and WSL, with exact retained reference images. These
model checks used automatic execution-slot reclaim disabled and do not extend
the explicit small-gate acceptance to a Linux automatic pressure policy.
Platform-specific budgets, warnings and measurements belong to the
[model validation conditions](https://github.com/DEVAIEXP/ddo-examples/blob/main/docs/benchmarks/ltx23-image-details.md#installed-rc3-functional-validation);
the [DDO validation summary](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/developer/validation.md#local-rc3-linux-wheel-pair)
records the local wheel identities. These development artifacts have not been
published or certified for arbitrary models, full video or performance.
