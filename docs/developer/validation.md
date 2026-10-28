# Validation
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
