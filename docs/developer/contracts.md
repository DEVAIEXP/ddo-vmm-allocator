# Native contracts and ownership
The implementation is model-agnostic and uses CUDA Driver API C ABI handles.
Python adapters bind physical weight payloads without replacing backend math.
Retained sources own descriptors and immutable identity metadata; an independently
retained owner survives close of the original. Serialize operations with close on
the same owner. Positioned Windows/POSIX reads must never publish partial payloads.
Native persistent cache storage contains bytes, not loader CPU tensors. Separate
storage and CUDA registration ceilings; unregistering retains bytes. Host pressure
or explicit safe clear may remove unleased content. Source tickets/batches retain
owners until outstanding reads/DMAs complete; clear/close must refuse live ownership.
Arena virtual addresses stay stable. Physical pages are materialized, signed,
unpinned and evicted only after last-consumer events permit reuse. Registered
rings/reusable slots and native immutable batches must not be recycled early.
Partial submission failures retain ownership until completion is proven.
Physical allocation failure preserves the CUDA driver result. Only explicit
CUDA out-of-memory may enter the existing allocation-pressure recovery path;
other driver/context failures return CUDA_ERROR without capacity eviction or a
new model-arena capacity boundary. This classification does not establish that
CUDA out-of-memory was caused by host commit exhaustion or permit driver recovery.
Temporary fallback remains event-safe and observable; do not force-free a region
after poisoned CUDA completion or promise driver-loss recovery.
Execution buffers may combine file-backed weights and retained CPU tensors.
Only file-backed bindings enter immutable source batches; CPU payloads copy into
the existing destination views on the same upload stream. A completion event
covers both sources before consumption. Buffer reuse still requires the native
last-consumer fence, including after partial upload failure. File-only batches
retain their original transport path. This contract does not establish full-model
LoRA or quantization support.
At an exclusive pre-execution boundary, `ModelArenaModuleAdapter.cpu_source_intervals()`
reports integer address/byte ranges for every retained CPU payload binding.
It does not copy, release or retain additional tensors, and refuses unsupported
layouts. Callers must verify completeness against native binding counts and
reject overlapping mutations. This snapshot does not prove DMA completion or
permit concurrent rebinding, execution or close.
Admission observes physical/commit/cgroup/process bounds and fails conservatively
on unknown metadata. The canonical cgroup mount-root fix preserves those bounds.
Eager host-ring storage checks Windows/Linux RAM and commit headroom even without
an active cache ceiling, retaining the configured reserve when present. Unknown
or insufficient observations refuse allocation. Platform allocation failures copy
the original Win32/errno code before cleanup; this text does not itself establish
causality or provide a typed Python exception. Observation cannot reserve memory
against concurrent allocations by other processes.
An enabled Windows DXGI budget guard refuses materialization when its observation
fails. That failure returns CUDA_ERROR without pressure eviction or a new arena
capacity boundary. Observed insufficient headroom retains the existing pressure
path. Budget/reserve checks use subtraction after bounds checks, avoiding unsigned
overflow. Existing disabled-guard and model-arena WDDM placement choices remain.
Ceilings never grow automatically. WSL registration constraints are independent
from storage and GPU residency; WDDM behavior is not emulated on Linux.
Profiles/reports are consumed through DDO's public interface, not host internals.
