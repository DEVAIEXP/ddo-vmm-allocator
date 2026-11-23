# Native contracts and ownership

## Recovery notification control

The optional `ddo_vmm_set_recovery_warns_v1(0|1)` export changes a process-wide,
atomic console-notification flag without CUDA calls, waits or allocation changes.
New builds default to quiet successful recovery. Opt-in enables transient
materialization refusal details and successful capacity-transaction notices.
Failed transaction reports and final allocation failures remain visible.
Capacity receipts and counters are recorded regardless of console verbosity.
Invalid flag values are refused without changing the previous value. Libraries
without this export preserve their existing behavior; bindings report control
as unavailable rather than claiming that old binaries were silenced.

## Optional execution-slot backing reclaim

Execution pools expose additive `reclaim_idle`, `configure_reclaim` and
`reclaim_stats` C ABI functions. Reclaim preserves the virtual address and releases
physical pages only from an unleased, materialized slot after a CPU-visible
successful last-consumer event query. A pending event refuses reclamation without
synchronizing. The next acquisition materializes backing through the ordinary
fault path; the adapter must upload the weights again before consumption.

The registry serializes pool lifetime against allocation callbacks, with lock
order registry, pool, arena. Allocation callbacks use nonblocking lookup/locks.
Completion or partial unmap/release failure poisons the slot and retains ownership;
acquisition and close refuse unsafe reuse. Successfully released bytes remain
accounted even if a later page fails. This is not driver-loss recovery.

Automatic reclaim is disabled by default. Windows opt-in uses DXGI local-budget
headroom and a configured margin, releasing at most one eligible idle slot per
allocation request. Missing observations refuse the request. Busy slots are never
forced free. Linux refuses this automatic policy; explicit native reclamation is
a separate platform validation target. No physical local-VRAM placement is
guaranteed by virtual-address stability or resident byte counters.

Stats schema 1 has 16 uint64 fields: schema, count, enabled, margin bytes, queries,
triggers, busy decisions, reclaim operations, released bytes, recreations,
recreated bytes, unavailable observations, failures and three reserved zeros.
Older artifacts without these optional exports report unavailable statistics and
refuse explicit enablement. There is no activation eviction or codec dependency.

## General ownership

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

## Windows execution-slot pressure during allocator reuse

When execution-backing reclaim is explicitly enabled, the Windows native Torch
allocator also checks current pressure on existing-backing allocation paths.
It enters before the Torch pool/cache locks, after the existing request/context
validation where applicable. Registry and execution-pool lookup remain try-lock
operations. Disabled policy returns before registry lookup or clock observation.

Each eligible execution pool limits reuse checks to one per 250 ms. Existing
growth checks remain immediate and postpone the next reuse check. Reuse assesses
current DXGI budget/usage against the configured margin; it does not pretend that
reusing existing backing requires another physical allocation. Unknown budget
observations retain the existing conservative error behavior.

Only observed pressure permits an idle-slot reclaim attempt. The existing lease,
completion and poisoned-ownership checks remain authoritative: pending or live
buffers are retained, at most one completed idle slot is released per check, and
backing is recreated only upon subsequent acquisition. No timer thread, extra
CUDA synchronization, periodic forced recreation or Python scheduling is added.
The WDDM reuse hook is compiled out on Linux/WSL. Public C ABI and default
enablement are unchanged. Compilation and CPU cadence tests do not establish
GPU lifecycle safety or model performance; those require separate validation.

## Internal mapping observation

The optional Windows `DDO_VMM_MAPPING_TRACE` build feature is off by default.
Its internal versioned ABI records map, unmap and release call arguments,
original driver results and QPC call brackets. It adds no CUDA queries, events,
waits, retries or residency decisions. Recording starts only through explicit
CPU setup with a row capacity and byte limit; rows are 128 bytes. Start is
one-shot. Stop refuses in-flight observed calls; snapshots are CPU copies and
do not prove GPU completion. A full buffer retains prior rows and reports
dropped records. Loss or clock failure prevents complete attribution.

Process creation identity and a bracketed UTC/QPC anchor are exported. Records
append in return order; sequence numbers identify admission order. Failed
unmap/release calls remain attempts, never evidence of completed retirement.
Zero owner/generation/role/context and an unknown device remain unavailable;
this initial observer does not classify model weights or activations. Consumers
must separately establish adapter, logical-owner epochs, access roles and clock
alignment before joining WDDM evidence. A CUDA allocation handle is not a VidMm
handle. These records alone do not report local/shared placement or lifetime
percentages. Collection must begin before the mappings of interest, otherwise
initial state remains unknown. Unsupported/disabled builds return unavailable.

With both mapping observation and the native span pool enabled, the internal
owner observation query returns copied canonical allocation metadata under the
pool lock: backing address, generation, lease, requested extent, offset, device
and stored context. It performs no CUDA calls, holds no additional owners and
does not reap fences or change pool state. Queries require the canonical storage
start, not an interior view pointer. Missing owners remain explicit; unknown
completion, role and content version are UINT64_MAX. A successful query is only
a point-in-time identity observation, never permission to unmap or reuse bytes.
The disabled/non-pool implementation returns unavailable without touching output.
