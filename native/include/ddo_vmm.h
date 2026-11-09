#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(DDO_VMM_CPU_API)
#define DDO_VMM_API
#elif defined(_WIN32)
#if defined(DDO_VMM_BUILDING_LIBRARY)
#define DDO_VMM_API __declspec(dllexport)
#else
#define DDO_VMM_API __declspec(dllimport)
#endif
#else
#define DDO_VMM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ddo_vmm_region ddo_vmm_region;
typedef struct ddo_vmm_model_arena ddo_vmm_model_arena;
typedef struct ddo_vmm_execution_pool ddo_vmm_execution_pool;

/** Before first materialization, reserve execution-buffer capacity within the
 * existing residency ceiling and fix the admitted model prefix. No allocation
 * or ceiling increase occurs. Legacy callers are unchanged. */
DDO_VMM_API int ddo_vmm_model_arena_enable_streaming(
    ddo_vmm_model_arena* arena, size_t execution_buffer_bytes);
/** Native compute fence for a reusable execution arena. Record after compute;
 * wait queues a dependency on the upload stream, not a host/global wait. */
DDO_VMM_API int ddo_vmm_model_arena_record_compute(
    ddo_vmm_model_arena* arena, void* stream);
DDO_VMM_API int ddo_vmm_model_arena_wait_compute(
    ddo_vmm_model_arena* arena, void* stream);
/** Local fence drainage for explicit close. Failure retains the native owner. */
DDO_VMM_API int ddo_vmm_model_arena_drain_compute(ddo_vmm_model_arena* arena);
/** Native round-robin execution slots; capacity is carved from model ceiling.
 * acquire returns a leased, pinned arena after queuing its reuse dependency.
 * release records the caller's last-use stream before making that slot free.
 * All methods must be serialized against close; destroy refuses live leases. */
DDO_VMM_API int ddo_vmm_execution_pool_create(ddo_vmm_model_arena* model,
    size_t slot_bytes, size_t slot_count, ddo_vmm_execution_pool** output);
DDO_VMM_API int ddo_vmm_execution_pool_acquire(ddo_vmm_execution_pool* pool,
    void* upload_stream, size_t* slot, ddo_vmm_model_arena** arena);
DDO_VMM_API int ddo_vmm_execution_pool_release(ddo_vmm_execution_pool* pool,
    size_t slot, void* last_use_stream);
DDO_VMM_API int ddo_vmm_execution_pool_close(ddo_vmm_execution_pool** pool);
/** Explicit idle-slot reclaim. Requires a completed consumer fence; never waits. */
DDO_VMM_API int ddo_vmm_execution_pool_reclaim_idle(ddo_vmm_execution_pool* pool,
    size_t index, uint64_t* released_bytes);
/** Opt-in Windows DXGI policy. Unsupported platforms refuse enablement. */
DDO_VMM_API int ddo_vmm_execution_pool_configure_reclaim(ddo_vmm_execution_pool* pool,
    int enabled, uint64_t margin_bytes);
/** Version 1: 16 copied uint64_t values; counters describe backing, not placement. */
DDO_VMM_API int ddo_vmm_execution_pool_reclaim_stats(ddo_vmm_execution_pool* pool,
    uint64_t* values, size_t count);
typedef struct ddo_vmm_host_ring ddo_vmm_host_ring;
typedef struct ddo_vmm_source_file ddo_vmm_source_file;
typedef struct ddo_vmm_source_cache ddo_vmm_source_cache;
typedef struct ddo_vmm_source_ticket ddo_vmm_source_ticket;
typedef struct ddo_vmm_source_batch ddo_vmm_source_batch;

typedef enum ddo_vmm_status {
    DDO_VMM_SUCCESS = 0,
    DDO_VMM_INVALID_ARGUMENT = 1,
    DDO_VMM_OUT_OF_MEMORY = 2,
    DDO_VMM_CUDA_ERROR = 3,
    DDO_VMM_IO_ERROR = 4,
    DDO_VMM_SOURCE_CHANGED = 5,
    DDO_VMM_UNSUPPORTED = 6,
    DDO_VMM_INTERNAL_ERROR = 7,
    DDO_VMM_BYPASS = 8,
    DDO_VMM_BUSY = 9,
} ddo_vmm_status;

/** Immutable file-source ABI, independent of CUDA initialization. */
typedef struct ddo_vmm_source_file_info {
    uint32_t struct_size;
    uint32_t version;
    uint64_t volume_id;
    uint64_t file_id;
    uint64_t size_bytes;
    uint64_t last_write_time;
    uint64_t generation;
} ddo_vmm_source_file_info;

DDO_VMM_API uint32_t ddo_vmm_source_abi_version(void);
DDO_VMM_API const char* ddo_vmm_source_last_error(void);
/** Windows: deny write/delete sharing for the lifetime of all retained owners. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_file_open(
    const char* path_utf8, ddo_vmm_source_file** out_file
);
/** Produce a separately owned handle to the same immutable file generation. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_file_retain(
    const ddo_vmm_source_file* file, ddo_vmm_source_file** out_file
);
/** Caller initializes struct_size and version=1. Revalidates file metadata. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_file_get_info(
    const ddo_vmm_source_file* file, ddo_vmm_source_file_info* out_info
);
/** Synchronous positioned read into caller-owned storage; no CUDA calls.
 * Revalidates identity/size/time. On failure, destination contents are invalid.
 * Concurrent reads through retained handles are supported. The caller must
 * serialize close against calls on that SAME handle. Zero-size reads are invalid.
 */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_file_read(
    const ddo_vmm_source_file* file, uint64_t offset_bytes,
    void* destination, size_t size_bytes
);
/** Releases and nulls this owner; repeated close on the null handle is safe. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_file_close(ddo_vmm_source_file** file);

/** Experimental micro-contract only: progressive registered slots, <=128 MiB.
 * Requires the requested CUDA device's primary context to be current. No
 * production default uses this API until progressive allocation/pressure gates.
 */
typedef struct ddo_vmm_source_cache_config {
    uint32_t struct_size;
    uint32_t version;
    int32_t device_ordinal;
    uint32_t slot_count;
    uint64_t slot_bytes;
    uint64_t host_headroom_bytes;
    /** ABI v2 field, ABI v3 behavior: process-wide cache + native eager-ring cap.
     * Config versions 1/2 retain the 128 MiB ceiling; config version 3 requires
     * ABI v6 and permits a process cap up to 1 GiB. Local cache capacity still
     * cannot exceed 128 MiB. Config v4 (ABI v7) explicitly permits <=32 GiB
     * local cache and <=40 GiB process cap, with shared aligned file windows.
     * It commits/registers lazily and checks RAM/commit on every growth.
     * These are configuration ceilings, not approved allocation targets.
     * The strictest live cache limit wins. */
    uint64_t process_limit_bytes;
    /** Config v5 / ABI v8: separate local CUDA registration ceiling.
     * slot_bytes * slot_count remains the persistent storage ceiling.
     * Zero uses that same ceiling; smaller values recycle registrations,
     * never persistent contents. Versions 1-4 ignore this suffix. */
    uint64_t registered_limit_bytes;
} ddo_vmm_source_cache_config;
typedef struct ddo_vmm_source_cache_stats {
    uint32_t struct_size;
    uint32_t version;
    uint64_t registered_bytes;
    uint64_t retained_bytes;
    uint64_t live_tickets;
    uint64_t hits;
    uint64_t misses;
    uint64_t bypasses;
    uint64_t source_read_bytes;
    uint64_t dma_bytes;
    /* ABI v2 additions. ABI v1 prefix remains supported. */
    uint64_t capacity_bytes;
    uint64_t peak_registered_bytes;
    uint64_t process_registered_bytes;
    uint64_t process_limit_bytes;
    uint64_t admission_denials;
    /* ABI v3: process_registered_bytes remains cache-only. Admission considers
     * its sum with native eager rings; unrelated allocators are excluded. */
    uint64_t process_staging_registered_bytes;
    /* ABI v4: pressure policy and explicit trim observations. */
    uint64_t pressure_evictions;
    uint64_t pressure_released_bytes;
    uint64_t pressure_blocked_requests;
    uint64_t pressure_suspended;
    /* Stats v5 / ABI v8: persistent buffer accounting. */
    uint64_t committed_bytes;
    uint64_t peak_committed_bytes;
    uint64_t process_committed_bytes;
    uint64_t registered_limit_bytes;
    uint64_t registrations;
    uint64_t unregistrations;
    uint64_t registration_waits;
    uint64_t ring_fallback_bytes;
    uint64_t host_copy_bytes;
    /* Stats v6 / ABI v9: host wall time, not CUDA device-event DMA time.
     * Direct persistent registration/recycling only; teardown is excluded. */
    double registration_seconds;
    double unregistration_seconds;
    double registration_wait_seconds;
    /* Stats v7 / ABI v10: direct retained-source copy host scopes. */
    double source_ready_seconds;
    double dma_submit_seconds;
    double dma_event_seconds;
    double ticket_close_wait_seconds;
    /* Stats v8 / ABI v11: full native API scopes, nested, not device time. */
    double request_total_seconds;
    double copy_total_seconds;
    double close_total_seconds;
    double event_acquire_seconds;
    double event_retire_seconds;
    double registration_schedule_seconds;
    double batch_submit_seconds;
    double batch_reap_seconds;
    uint64_t events_created;
    uint64_t events_reused;
    uint64_t batch_submissions;
    uint64_t batch_bypass_bytes;
} ddo_vmm_source_cache_stats;
/** Immutable execution-unit descriptors, retained once. dest_offset is relative
 * to the destination base supplied at submit, allowing native slot rotation. */
typedef struct ddo_vmm_source_span {
    ddo_vmm_source_file* source;
    uint64_t file_offset;
    size_t size;
    size_t dest_offset;
} ddo_vmm_source_span;
DDO_VMM_API ddo_vmm_status ddo_vmm_source_batch_create(
    ddo_vmm_source_cache* cache, const ddo_vmm_source_span* spans, size_t count,
    ddo_vmm_host_ring* fallback_ring, size_t ring_slots, size_t ring_slot_bytes,
    ddo_vmm_source_batch** output);
/** Optional source-only preparation: no GPU allocation, DMA, new registration
 * or explicit read-completion wait. Existing admission/pressure guards apply.
 * Available only in the isolated native-source-dispatcher build. */
DDO_VMM_API int ddo_vmm_source_dispatcher_enabled(void);
/** Cold per-cache selection (0/1) before the first request. A persistent cache
 * in a dispatcher build selects it by default; 0 uses the direct ring reads of
 * builds without it. Enabling it elsewhere returns UNSUPPORTED; after a request
 * (until clear) BUSY. Changes no capacity, admission or ownership. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_cache_select_dispatcher(ddo_vmm_source_cache* cache, int enabled);
/** Optional CPU-only metadata snapshot: owned windows, peak since clear and
 * dispatcher limit. Counts include completed reads awaiting retirement. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_dispatcher_snapshot(
    ddo_vmm_source_cache* cache, size_t* owned, size_t* peak, size_t* limit);
/** CPU-only snapshot: transfer-owned, pooled and batch-completion events.
 * Excludes unsubmitted tickets and other allocators. No CUDA queries/waits.
 * Outputs unchanged on failure. Requires serialized cache lifecycle. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_owned_events_v1(
    ddo_vmm_source_cache* cache, size_t* transfers, size_t* pooled, size_t* batches);
#ifdef DDO_VMM_SOURCE_TEST_HOOKS
/** Test build only: fail one copy after N allowed copy attempts (-1 disables),
 * or one ticket close before completion proof. Never induce a driver error. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_test_failures(
    ddo_vmm_source_cache* cache, int64_t copies_before_failure, int close_once);
#endif
DDO_VMM_API ddo_vmm_status ddo_vmm_source_batch_prepare(ddo_vmm_source_batch* batch);
/** All windowing, request/copy and completion recycling happen natively.
 * Partial failure retains submitted leases; caller must not release destination
 * pins before its upload stream is proven drained. No global synchronization. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_batch_submit(
    ddo_vmm_source_batch* batch, ddo_vmm_model_arena* arena,
    size_t dest_base, void* stream);
DDO_VMM_API ddo_vmm_status ddo_vmm_source_batch_wait_on_stream(
    ddo_vmm_source_batch* batch, void* stream);
DDO_VMM_API ddo_vmm_status ddo_vmm_source_batch_close(ddo_vmm_source_batch** batch);
/** Retire completed batch-owned tickets without waiting for pending DMA.
 * Explicit phase-boundary collection; snapshot itself remains read-only. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_cache_collect_completed(ddo_vmm_source_cache* cache);
DDO_VMM_API uint32_t ddo_vmm_source_cache_abi_version(void);
DDO_VMM_API const char* ddo_vmm_source_cache_last_error(void);
DDO_VMM_API ddo_vmm_status ddo_vmm_source_cache_create(
    const ddo_vmm_source_cache_config* config, ddo_vmm_source_cache** out_cache);
/** BYPASS returns no ticket. Same/contained ranges reuse retained entries;
 * partially overlapping ranges bypass rather than duplicate their payload. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_cache_request(
    ddo_vmm_source_cache* cache, const ddo_vmm_source_file* source,
    uint64_t offset, size_t size, ddo_vmm_source_ticket** out_ticket);
/** Wait for this entry's source only; does not synchronize CUDA. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_ticket_wait(ddo_vmm_source_ticket* ticket);
/** One H2D per ticket. Caller holds the arena pin until compute completes. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_ticket_copy_async(
    ddo_vmm_source_ticket* ticket, ddo_vmm_model_arena* arena,
    size_t arena_offset, void* stream);
/** Registration refusal fallback using an existing registered ring slot.
 * Copies retained native bytes into that slot, not from the file or a tensor.
 * Caller reserves the slot and keeps ring/arena alive until ticket DMA ends. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_ticket_copy_via_ring_async(
    ddo_vmm_source_ticket* ticket, ddo_vmm_host_ring* ring, size_t slot,
    ddo_vmm_model_arena* arena, size_t arena_offset, void* stream);
DDO_VMM_API ddo_vmm_status ddo_vmm_source_ticket_wait_on_stream(
    ddo_vmm_source_ticket* ticket, void* stream);
/** Nonblocking DMA completion query. Requires a recorded upload event.
 * Returns CUDA_ERROR on asynchronous failure, never reports it as pending. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_ticket_dma_ready(
    ddo_vmm_source_ticket* ticket, int* out_ready);
/** Wait for this ticket's DMA, then release it. No device-wide synchronization. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_ticket_close(ddo_vmm_source_ticket** ticket);
/** Drop retained entries only when no ticket is live; drains outstanding reads. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_cache_clear(ddo_vmm_source_cache* cache);
/** Versions 1-4 release ready, unleased tail entries until registered_bytes <= target.
 * Config v5 unregisters eligible segments without discarding their bytes.
 * Returns BUSY if a live ticket or filling entry prevents reaching target.
 * Never waits for I/O or synchronizes the device. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_cache_trim(
    ddo_vmm_source_cache* cache, uint64_t target_registered_bytes);
DDO_VMM_API ddo_vmm_status ddo_vmm_source_cache_snapshot(
    ddo_vmm_source_cache* cache, ddo_vmm_source_cache_stats* stats);
/** Optional pressure observations v1: exactly 23 uint64 values.
 * Preserves the stats ABI and performs no OS/CUDA sampling. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_cache_pressure_snapshot_v1(
    ddo_vmm_source_cache* cache, uint64_t* values, size_t count);
/** BUSY leaves the handle alive when tickets still own it. Caller serializes
 * destruction against all calls on this handle, including ticket operations. */
DDO_VMM_API ddo_vmm_status ddo_vmm_source_cache_close(ddo_vmm_source_cache** cache);

typedef enum ddo_vmm_backing_kind {
    DDO_VMM_BACKING_DEVICE = 0,
    DDO_VMM_BACKING_HOST = 1,
} ddo_vmm_backing_kind;

typedef struct ddo_vmm_windows_memory_info {
    uint64_t local_budget_bytes;
    uint64_t local_usage_bytes;
    uint64_t nonlocal_budget_bytes;
    uint64_t nonlocal_usage_bytes;
    uint64_t available_physical_bytes;
    uint64_t available_commit_bytes;
} ddo_vmm_windows_memory_info;

/**
 * Reserve a stable CUDA virtual address, create device-local physical backing,
 * map it, and grant read/write access to `device_ordinal`.
 *
 * The returned virtual address is valid until ddo_vmm_region_destroy().
 * `size_bytes` is rounded up to CUDA's minimum VMM allocation granularity.
 */
DDO_VMM_API ddo_vmm_status ddo_vmm_region_create(
    int device_ordinal,
    size_t size_bytes,
    ddo_vmm_region** out_region
);

DDO_VMM_API ddo_vmm_status ddo_vmm_region_create_ex(
    int device_ordinal,
    size_t size_bytes,
    ddo_vmm_backing_kind backing_kind,
    ddo_vmm_region** out_region
);

/** Unmap, release physical backing, and return the virtual address reservation. */
DDO_VMM_API void ddo_vmm_region_destroy(ddo_vmm_region* region);

/** Release physical backing while preserving the region's reserved virtual address. */
DDO_VMM_API ddo_vmm_status ddo_vmm_region_evict(ddo_vmm_region* region);

/** Recreate device-local backing and map it at the region's original virtual address. */
DDO_VMM_API ddo_vmm_status ddo_vmm_region_materialize(ddo_vmm_region* region);

/**
 * Recreate backing for a byte range inside the reserved virtual address.
 *
 * `offset_bytes` and `size_bytes` are rounded outward to CUDA's VMM
 * allocation granularity.  The returned resident-byte accounting therefore
 * may be larger than the requested payload range.
 */
DDO_VMM_API ddo_vmm_status ddo_vmm_region_materialize_range(
    ddo_vmm_region* region,
    size_t offset_bytes,
    size_t size_bytes
);

/** Return resident physical bytes currently mapped into this virtual region. */
DDO_VMM_API size_t ddo_vmm_region_resident_size(const ddo_vmm_region* region);

DDO_VMM_API uintptr_t ddo_vmm_region_address(const ddo_vmm_region* region);
DDO_VMM_API size_t ddo_vmm_region_requested_size(const ddo_vmm_region* region);
DDO_VMM_API size_t ddo_vmm_region_mapped_size(const ddo_vmm_region* region);
DDO_VMM_API int ddo_vmm_region_is_materialized(const ddo_vmm_region* region);
DDO_VMM_API size_t ddo_vmm_device_granularity(int device_ordinal);

/**
 * Reserve one stable virtual address span for a complete model-weight address
 * space. Physical pages are not committed until ddo_vmm_model_arena_fault_range().
 *
 * resident_budget_bytes == 0 lets the CUDA driver/WDDM decide residency until
 * allocation pressure is reported by cuMemCreate. page_size_bytes == 0 uses
 * the CUDA minimum VMM allocation granularity for the selected backing kind.
 */
DDO_VMM_API ddo_vmm_status ddo_vmm_model_arena_create(
    int device_ordinal,
    size_t virtual_size_bytes,
    size_t resident_budget_bytes,
    size_t page_size_bytes,
    ddo_vmm_backing_kind backing_kind,
    ddo_vmm_model_arena** out_arena
);

DDO_VMM_API void ddo_vmm_model_arena_destroy(ddo_vmm_model_arena* arena);

/**
 * Materialize and pin the pages covering a byte range. out_signature receives
 * a stable content-generation signature for the current resident pages. If the
 * same range faults again without any page being recycled, the signature stays
 * unchanged and callers can skip copying immutable source bytes.
 */
DDO_VMM_API ddo_vmm_status ddo_vmm_model_arena_fault_range(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes,
    uint64_t* out_signature
);
/**
 * Temporarily map a small reusable physical page across a virtual range so a
 * CUDA DLPack view can be constructed. Call release_view_range immediately
 * afterwards; this mapping is never model-weight residency.
 */
/** Read-only residency/generation observation. Does not fault, pin, allocate,
 * evict or touch CUDA context. Consumers must still fault/recheck before DMA. */
DDO_VMM_API ddo_vmm_status ddo_vmm_model_arena_peek_range_signature(
    ddo_vmm_model_arena* arena, size_t offset_bytes, size_t size_bytes,
    uint64_t* out_signature, int* out_resident
);
DDO_VMM_API ddo_vmm_status ddo_vmm_model_arena_prepare_view_range(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes
);
DDO_VMM_API ddo_vmm_status ddo_vmm_model_arena_release_view_range(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes
);
DDO_VMM_API ddo_vmm_status ddo_vmm_model_arena_copy_from_host(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes,
    const void* host_ptr
);
DDO_VMM_API ddo_vmm_status ddo_vmm_model_arena_copy_from_host_async(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes,
    const void* host_ptr,
    void* cuda_stream
);
/** Release one pin acquired by fault_range for each page covering the range. */
DDO_VMM_API ddo_vmm_status ddo_vmm_model_arena_unpin_range(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes
);

DDO_VMM_API uintptr_t ddo_vmm_model_arena_address(const ddo_vmm_model_arena* arena);
DDO_VMM_API int ddo_vmm_model_arena_device_ordinal(const ddo_vmm_model_arena* arena);
DDO_VMM_API size_t ddo_vmm_model_arena_virtual_size(const ddo_vmm_model_arena* arena);
DDO_VMM_API size_t ddo_vmm_model_arena_page_size(const ddo_vmm_model_arena* arena);
DDO_VMM_API size_t ddo_vmm_model_arena_resident_size(const ddo_vmm_model_arena* arena);
DDO_VMM_API size_t ddo_vmm_model_arena_fault_ceiling(const ddo_vmm_model_arena* arena);
DDO_VMM_API uint64_t ddo_vmm_model_arena_materialized_pages(const ddo_vmm_model_arena* arena);
DDO_VMM_API uint64_t ddo_vmm_model_arena_reused_pages(const ddo_vmm_model_arena* arena);
DDO_VMM_API uint64_t ddo_vmm_model_arena_evicted_pages(const ddo_vmm_model_arena* arena);

/** Number of resident logical pages in the contiguous low-address prefix. */
DDO_VMM_API size_t ddo_vmm_model_arena_resident_prefix_pages(const ddo_vmm_model_arena* arena);

/** Number of native capacity evictions selected from the high-address tail. */
DDO_VMM_API uint64_t ddo_vmm_model_arena_tail_evictions(const ddo_vmm_model_arena* arena);

/** Number of model pages released by native global CUDA allocation pressure. */
DDO_VMM_API uint64_t ddo_vmm_model_arena_pressure_evictions(const ddo_vmm_model_arena* arena);

/**
 * Allocate a bounded ring of CUDA-registered host slots for native shard
 * transport. Slots are intentionally raw byte ranges; I/O tickets and DMA
 * events are layered on this ownership contract.
 */
DDO_VMM_API ddo_vmm_status ddo_vmm_host_ring_create(
    int device_ordinal,
    size_t slot_size_bytes,
    size_t slot_count,
    ddo_vmm_host_ring** out_ring
);
/** Close requires callers to stop submitting new operations on this handle.
 * Existing source waiters retire before storage and synchronization are freed. */
DDO_VMM_API void ddo_vmm_host_ring_destroy(ddo_vmm_host_ring* ring);
DDO_VMM_API size_t ddo_vmm_host_ring_slot_size(const ddo_vmm_host_ring* ring);
DDO_VMM_API size_t ddo_vmm_host_ring_slot_count(const ddo_vmm_host_ring* ring);
/** Native source transport configuration and lifetime counters. */
DDO_VMM_API size_t ddo_vmm_host_ring_source_worker_count(const ddo_vmm_host_ring* ring);
DDO_VMM_API uint64_t ddo_vmm_host_ring_source_read_chunks(const ddo_vmm_host_ring* ring);
DDO_VMM_API uint64_t ddo_vmm_host_ring_source_read_bytes(const ddo_vmm_host_ring* ring);
DDO_VMM_API uint64_t ddo_vmm_host_ring_source_read_failures(const ddo_vmm_host_ring* ring);
DDO_VMM_API uintptr_t ddo_vmm_host_ring_slot_address(const ddo_vmm_host_ring* ring, size_t slot_index);
DDO_VMM_API int ddo_vmm_host_ring_slot_ready(ddo_vmm_host_ring* ring, size_t slot_index);
/** Block only until this slot's prior DMA completes, making its bytes reusable. */
DDO_VMM_API ddo_vmm_status ddo_vmm_host_ring_wait_slot(ddo_vmm_host_ring* ring, size_t slot_index);
/** Queue one slot range into a faulted model-arena range on `cuda_stream`.
 * The caller retains both the arena pin and the slot until its own CUDA event
 * has completed; this function never synchronizes the stream. */
DDO_VMM_API ddo_vmm_status ddo_vmm_host_ring_copy_to_arena_async(
    ddo_vmm_host_ring* ring,
    size_t slot_index,
    size_t slot_offset_bytes,
    ddo_vmm_model_arena* arena,
    size_t arena_offset_bytes,
    size_t size_bytes,
    void* cuda_stream
);
/** Windows-only native shard operation: read a file range into one registered
 * host slot, then enqueue its DMA to a faulted model-arena range.  The slot's
 * completion event protects reuse; this call never synchronizes CUDA. */
DDO_VMM_API ddo_vmm_status ddo_vmm_host_ring_read_file_to_arena_async(
    ddo_vmm_host_ring* ring,
    size_t slot_index,
    const char* path_utf8,
    uint64_t file_offset_bytes,
    ddo_vmm_model_arena* arena,
    size_t arena_offset_bytes,
    size_t size_bytes,
    void* cuda_stream
);
DDO_VMM_API ddo_vmm_status ddo_vmm_host_ring_read_file_async(
    ddo_vmm_host_ring* ring, size_t slot_index, const char* path_utf8,
    uint64_t file_offset_bytes, size_t size_bytes
);
/** Dispatcher builds wait without the global admission lock. Concurrent
 * waiters retain the slot until all have observed the shared completion. */
DDO_VMM_API ddo_vmm_status ddo_vmm_host_ring_wait_file_read(
    ddo_vmm_host_ring* ring, size_t slot_index
);
/**
 * Wait for source reads in a ticket, then enqueue all of its H2D ranges on one
 * CUDA stream. The final slot's completion event represents the whole ticket.
 */
DDO_VMM_API ddo_vmm_status ddo_vmm_host_ring_copy_ticket_to_arena_async(
    ddo_vmm_host_ring* ring,
    const size_t* slot_indices,
    ddo_vmm_model_arena* arena,
    const size_t* arena_offsets_bytes,
    const size_t* sizes_bytes,
    size_t range_count,
    void* cuda_stream
);
/** Queue several packed slot ranges, then record one completion event. */
DDO_VMM_API ddo_vmm_status ddo_vmm_host_ring_copy_batch_to_arena_async(
    ddo_vmm_host_ring* ring,
    size_t slot_index,
    const size_t* slot_offsets_bytes,
    ddo_vmm_model_arena* arena,
    const size_t* arena_offsets_bytes,
    const size_t* sizes_bytes,
    size_t range_count,
    void* cuda_stream
);
DDO_VMM_API ddo_vmm_status ddo_vmm_host_ring_wait_slot_async(
    ddo_vmm_host_ring* ring,
    size_t slot_index,
    void* cuda_stream
);

/** Register model-weight regions that may be evicted under allocator pressure. */
DDO_VMM_API void ddo_vmm_pressure_register(ddo_vmm_region* region);
DDO_VMM_API void ddo_vmm_pressure_unregister(ddo_vmm_region* region);
DDO_VMM_API void ddo_vmm_pressure_set_pinned(ddo_vmm_region* region, int pinned);
DDO_VMM_API void ddo_vmm_pressure_touch(ddo_vmm_region* region);
/** Evict one eligible native VMM mapping for a device; returns 1 on success. */
DDO_VMM_API int ddo_vmm_pressure_evict_one(int device_ordinal);
/**
 * Reserve WDDM headroom for an ordinary CUDA allocation. When a configured
 * guard needs room, this reclaims eligible native VMM pages first.
 */
DDO_VMM_API ddo_vmm_status ddo_vmm_pressure_prepare_cuda_allocation(
    int device_ordinal, size_t required_bytes
);
DDO_VMM_API ddo_vmm_status ddo_vmm_windows_memory_query(
    int device_ordinal, ddo_vmm_windows_memory_info* out_info
);
DDO_VMM_API ddo_vmm_status ddo_vmm_dxgi_budget_guard_set(
    int device_ordinal,
    uint64_t local_headroom_bytes,
    uint64_t nonlocal_headroom_bytes,
    uint64_t system_headroom_bytes
);
DDO_VMM_API const char* ddo_vmm_last_error(void);
/** Optional process-lifetime release observations, no CUDA calls. Exactly six
 * uint64 values: sync calls, sync ns, failed proofs, cache eviction attempts,
 * attempted eviction bytes, eviction sync ns. Concurrent snapshots are approximate.
 */
DDO_VMM_API int ddo_vmm_torch_release_stats_v1(uint64_t* values, size_t count);

/** Optional atomic live-record snapshot (6 uint64_t values): current requested
 * bytes, minimum mapped bytes, actual mapped bytes, granularity padding bytes,
 * oversized reuse bytes, allocation count. Excludes cache/retiring/model arenas.
 * No CUDA calls or waits. Invalid arguments/query leave output unchanged.
 */
DDO_VMM_API int ddo_vmm_torch_allocation_footprint_v1(uint64_t* values, size_t count);

/** Optional native pool schema v1: exactly 15 uint64_t values. Enabled,
 * mapped, requested, leased, available, pending, quarantined bytes; live and
 * reserved lease counts; backing count; whole-idle backing bytes; eviction
 * attempts, host wait nanoseconds, failures, whole-idle backing count.
 * Physical backing counted once; excludes model arena and execution slots.
 * Return 2 means pool unavailable in this build, not zero consumption.
 * Read-only ledger snapshot, no CUDA calls. Failure leaves output untouched.
 */
DDO_VMM_API int ddo_vmm_torch_pool_stats_v1(uint64_t* values, size_t count);

/** Explicit process-wide phase-boundary trim of whole-idle native pool backing.
 * Does not free live leases, partial backing, model arenas or quarantined bytes.
 * Proves event completion before checked physical release; may wait.
 * Return 0: operation completed (not a promise of zero VRAM); 1: failure with
 * unsafe bytes retained; 2: unavailable in this build. No automatic invocation.
 */
DDO_VMM_API int ddo_vmm_torch_trim_idle_v1(void);

/** Diagnostic-only process scope. Enable returns 1 if already owned, 2 if unsupported.
 * Snapshot: schema, enabled, seven (calls, host ns, failed calls) triples in
 * create/map/access/unmap/release/reserve/address-free order, pool reuse calls,
 * pool new backings, observed pool mapped-byte high-water mark. Count must be 26.
 * Reads do not invoke CUDA. Concurrent updates are not an atomic allocation census.
 */
DDO_VMM_API int ddo_vmm_allocation_diagnostics_enable_v1(int enabled);
DDO_VMM_API int ddo_vmm_allocation_diagnostics_snapshot_v1(uint64_t* values, size_t count);
/* Optional v2: count 36, same v1 fields with schema=2; indices 26..31 are
 * triples for access-retry context sync and enclosing fault host calls.
 * Fault failed-count field is unavailable/reserved zero, not success evidence.
 * Indices 32..35 count access success/not-ready/out-of-memory/other returns.
 * Nested timers overlap. No additional synchronization is introduced.
 */
DDO_VMM_API int ddo_vmm_allocation_diagnostics_snapshot_v2(uint64_t* values, size_t count);

#ifdef __cplusplus
}
#endif
