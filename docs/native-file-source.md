# Native immutable file source

This is the file/ownership foundation for a future registered source cache.
It does not allocate CUDA memory, register host pages, create an I/O pool or
enable caching. Windows is currently the implemented platform.

```python
from ddo_vmm_allocator.native import load_native_vmm

library = load_native_vmm(library_path)
with library.open_source_file(checkpoint_path) as source:
    info = source.info
    destination = bytearray(16)
    source.read_into(destination, offset_bytes=payload_offset)
```

The example is an allocator contract/diagnostic. Host runners continue to use
DDO's public loading and offload APIs, rather than managing these handles.

`source.info` reports immutable volume/file identity, byte size, last-write
timestamp and generation. Opening and reading use the same positioned Windows
file primitives as the existing native ring. The handle denies data writes and
deletion while open. Metadata is revalidated before/after reads; attribute-only
changes can still occur on Windows and are rejected as a changed source.

`source.retain()` creates an independently owned handle to the same generation.
Closing the original does not invalidate retained owners. Operations on one
Python owner serialize against its close; distinct retained owners support
concurrent positioned reads. Explicit close/context managers are the primary
lifecycle, with a best-effort finalizer as a secondary safeguard. No tensors,
file mappings or model payloads are retained by this wrapper.

The C ABI uses opaque handles and a versioned `ddo_vmm_source_file_info` with
caller-initialized `struct_size` and `version=1`. Source errors are obtained
from `ddo_vmm_source_last_error()` on the calling thread, independently of the
existing CUDA error channel. New status values distinguish I/O failure, changed
source, unsupported platform and internal failure; previous status values and
symbols keep their meaning. C++ exceptions are contained at the source ABI.

A C caller must keep the supplied destination valid for the entire synchronous
read and must serialize closing an owner with all operations on that owner.
Retain a separate owner before dispatching work elsewhere. Failed reads leave
destination contents invalid: callers must never publish a partially filled
range. Zero-length reads and ranges outside the file are rejected.

CPU-only verification, from the allocator directory after building Release:

```powershell
& ..\.venv-vmm\Scripts\python.exe -m pytest tests/test_native_source.py tests/test_platform.py -q -p no:cacheprovider
```

These tests intentionally do not use the CUDA marker or initialize CUDA.
Registered-memory, transfer, cancellation and event-protected cache contracts
will be separate hardware gates requiring user confirmation.

## Experimental retained-slot cache

`library.create_source_cache(slot_bytes=..., slots=...)` now creates a bounded
cache for hardware contract validation. It requires an initialized primary CUDA
context. Allocation now grows one registered slot per admission, capped at
128 MiB; creation starts with zero registered bytes. This progressive revision
passed its small CUDA contract gate but must not be enabled as a production
route. Automatic pressure reduction remains implementation work. No DDO
preset currently selects it.

`process_limit_bytes` caps the sum of registered cache slots and native eager
staging rings in this process. Existing staging rings retain their legacy
creation behavior; when they consume the cap, new cache admissions bypass.
The strictest live cache limit wins. New admissions check this limit and current
RAM/commit headroom; clear drains reads and releases registered slots and their
aggregate charges. Creating a cache with a cap below current registrations is
rejected without modifying other caches. This ledger does not include other
libraries and does not automatically evict under system pressure.

`cache.request(source, offset, size)` returns a ticket or `None` for native
bypass. The native cache retains exact/contained source ranges without rereading
them on hits. Partial overlaps, oversized ranges and full capacity bypass instead
of duplicating/continuously recycling bytes. Different aliases of the same file
share identity. Source filling uses the existing native ring worker pool.

A ticket permits one DMA attempt into a pinned model arena. Its completion event
can be waited on by a compute stream. Explicit ticket close waits for that DMA
only. Cancelling an unsubmitted ticket leaves any outstanding read owned by the
cache; cache clear/close drains it before closing the retained source. Cache
clear/close returns BUSY while tickets remain live. Callers still own the arena
pin through compute and must serialize close with calls on the same native owner.

The native snapshot includes `registered_bytes` (currently registered slots),
`retained_bytes` (admitted payload, possibly still filling), `live_tickets`,
`hits` (including coalesced pending reads), `misses`, `bypasses`,
`source_read_bytes` (requested reader bytes, not physical SSD traffic) and
`dma_bytes` (successfully submitted H2D with a recorded event). These are
micro-contract counters; the public DDO report integration remains pending.

Cache ABI v2 adds `capacity_bytes` (configured local ceiling),
`peak_registered_bytes`, `process_registered_bytes`, `process_limit_bytes` and
`admission_denials`. The C API continues accepting v1 config/stat prefixes;
v1 output writes only its declared prefix. The Python wrapper requires v3
capability instead of silently ignoring staging consumption. Cache ABI
v3 adds `process_staging_registered_bytes` and counts native eager rings during
admission. The C snapshot still supports the v1 and v2 output prefixes. This
revision passed its approved CUDA gate on 2026-09-29 (`14 passed, 11 deselected
in 6.50s`). The gate covers small buffers, not model-scale capacity or speed.

ABI v4 adds native pressure polling on cache requests. RAM and available commit
use a one-slot low margin and a two-slot recovery margin above configured
headroom. Ready, unleased tail entries may be released; live tickets protect
their bytes. `cache.trim(target_registered_bytes)` exercises that release path
without manufacturing host OOM. Snapshots add `pressure_evictions`,
`pressure_released_bytes`, `pressure_blocked_requests`, and
`pressure_suspended`. This revision compiled and passed 15 CPU source/platform
contracts; its approved CUDA gate passed (`15 passed, 11 deselected in 4.86s`).
Polling occurs on requests, not in
a background thread; real host-pressure behavior is not yet measured.

ABI v5 adds `ddo_vmm_source_ticket_dma_ready`, a nonblocking event query for
an uploaded ticket. It returns an explicit CUDA error on asynchronous failure.
This is needed to retire cache tickets during forward execution without a
host wait per tensor. Release compiled; CPU source/platform contracts passed
(`15 passed in 4.67s`); its approved CUDA gate passed (`15 passed,
11 deselected in 4.82s`).

The model-arena adapter now accepts an opt-in native cache alongside its native
ring. The cache chooses admission; bypassed ranges use the ring. Completed DMA
tickets are retired by the nonblocking query before subsequent batches, and
remaining tickets drain during close. Ring speculation is suppressed while the
cache is active so it cannot reread a cache hit. The approved small numerical
adapter CUDA gate passed (`2 passed, 9 deselected in 4.79s`): it checks
the cache hit without a second weight-source read and ring bypass for the bias.
The forced signature reset is test-only, not physical GPU eviction. The public
DDO fast runtime has an opt-in micro cache route; its default staging path is
unchanged.

The opt-in DDO micro route now creates the cache before its eager native ring.
While any cache is live, native ring creation checks the strictest process cap
and Windows RAM/commit headroom before registering host bytes. The 128 MiB
ceiling is unchanged. Release compiled and CPU-only contracts passed (`19
passed, 7 deselected in 4.59s`); the approved CUDA cap/rollback and adapter
regression gate passed (`2 passed, 22 deselected in 4.76s`).

ABI v6 separates the local 128 MiB cache capacity from the process cap.
Native config versions 1/2 retain their 128 MiB process ceiling; config
version 3 accepts a process cap up to 1 GiB to include larger staging rings.
The Python binding selects v3 only for an explicitly larger process cap and
requires ABI v6 in that case. Defaults and local cache allocation remain
unchanged. Release compiled; CPU-only contracts passed (`19 passed,
7 deselected in 4.58s`). The approved compatibility CUDA gate passed
(`17 passed, 11 deselected in 4.85s`); the approved 640 MiB staging plus
128 MiB retained-cache capacity gate passed (`1 passed in 4.62s`).

The larger capacity test lives in `tests/test_source_capacity.py` and is
skipped unless `DDO_VMM_CAPACITY_GATE=1`. It registers a 640 MiB eager ring,
fills two 64 MiB cache slots from a synthetic file, rejects additional staging
at the 768 MiB cap, and verifies clear/close refunds. Its observed ledger
reached 640 MiB staging + 128 MiB cache and returned to zero for both on close.
It does not load a model.

The six `tests/test_source_cache.py` contracts and three existing ring transport
regressions passed their approved CUDA gate on 2026-09-22 (`9 passed` in 7.79s).
This validates the small retained-slot contract, not full-model capacity or
inference performance. Subsequent CUDA gates still require user confirmation.
The subsequent progressive-allocation revision adds three tests (growth/release,
shared capacity and strictest live cap). Its approved gate passed on 2026-09-22:
`12 passed, 11 deselected in 4.64s`, including the previous nine contracts and
the v1 snapshot boundary check. This remains micro-contract validation only;
automatic pressure reduction and runtime integration are pending.
