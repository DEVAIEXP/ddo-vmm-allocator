"""Ownership bindings for the experimental native retained-slot cache."""
from __future__ import annotations

import ctypes as c
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from threading import RLock
import time

from .native import NativeVmmError
from .source import NativeSourceError


def _close_context_owner(owner, original_error):
    """Keep the operation's first error visible when protected close fails."""
    _run_cleanup_preserving_error(owner.close, type(owner).__name__, original_error)


def _run_cleanup_preserving_error(cleanup, description, original_error):
    """Attempt safety cleanup without replacing an already active failure."""
    if original_error is None:
        cleanup()
        return
    try:
        cleanup()
    except BaseException as cleanup_error:
        note = (
            f"DDO {description} cleanup failed: "
            f"{type(cleanup_error).__name__}: {cleanup_error}. "
            "Protection/cleanup did not complete; retain the owner for explicit cleanup."
        )
        add_note = getattr(original_error, "add_note", None)
        if callable(add_note):
            add_note(note)
        else:  # Python 3.10: retain details without replacing the first error.
            original_error.__notes__ = [*getattr(original_error, "__notes__", []), note]


class _Config(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("version", c.c_uint32),
                ("device_ordinal", c.c_int32), ("slot_count", c.c_uint32),
                ("slot_bytes", c.c_uint64), ("host_headroom_bytes", c.c_uint64),
                ("process_limit_bytes", c.c_uint64), ("registered_limit_bytes", c.c_uint64)]


class _Stats(c.Structure):
    _fields_ = [("struct_size", c.c_uint32), ("version", c.c_uint32)] + [
        (name, c.c_uint64) for name in (
            "registered_bytes", "retained_bytes", "live_tickets", "hits", "misses",
            "bypasses", "source_read_bytes", "dma_bytes", "capacity_bytes",
            "peak_registered_bytes", "process_registered_bytes", "process_limit_bytes", "admission_denials",
            "process_staging_registered_bytes", "pressure_evictions",
            "pressure_released_bytes", "pressure_blocked_requests", "pressure_suspended",
            "committed_bytes", "peak_committed_bytes", "process_committed_bytes",
            "registered_limit_bytes", "registrations", "unregistrations", "registration_waits",
            "ring_fallback_bytes", "host_copy_bytes")] + [
        (name, c.c_double) for name in (
            "registration_seconds", "unregistration_seconds", "registration_wait_seconds",
            "source_ready_seconds", "dma_submit_seconds", "dma_event_seconds", "ticket_close_wait_seconds",
            "request_total_seconds", "copy_total_seconds", "close_total_seconds", "event_acquire_seconds",
            "event_retire_seconds", "registration_schedule_seconds", "batch_submit_seconds", "batch_reap_seconds")] + [
        (name, c.c_uint64) for name in ("events_created", "events_reused", "batch_submissions", "batch_bypass_bytes")]


class _Span(c.Structure):
    _fields_ = [("source", c.c_void_p), ("file_offset", c.c_uint64),
                ("size", c.c_size_t), ("dest_offset", c.c_size_t)]


def _stream(stream):
    return c.c_void_p(0 if stream is None else int(getattr(stream, "cuda_stream", stream)))


@contextmanager
def _measure_batch_scope(owner, name):
    """Host wall scope, including failures; never waits for CUDA completion."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        with getattr(owner, "_lock", nullcontext()):
            metrics = getattr(owner, "_batch_timings", None)
            if metrics is None:
                metrics = owner._batch_timings = {}
            metrics[name + "_seconds"] = metrics.get(name + "_seconds", 0.0) + elapsed
            metrics[name + "_calls"] = metrics.get(name + "_calls", 0) + 1


class NativeSourceCache:
    def __init__(self, owner, *, slot_bytes, slots, device=0, host_headroom_bytes=2**30,
                 process_limit_bytes=128 * 2**20, model_windows=False,
                 persistent_buffer=False, registered_limit_bytes=0):
        if not isinstance(slot_bytes, int) or slot_bytes <= 0 or slot_bytes > 64 * 2**20 or slot_bytes % 65536:
            raise ValueError("slot_bytes must be 64 KiB aligned and <=64 MiB")
        capacity_limit = 32 * 2**30 if model_windows or persistent_buffer else 128 * 2**20
        if not isinstance(slots, int) or slots <= 0 or slots >= 2**32 or slots * slot_bytes > capacity_limit:
            raise ValueError("cache capacity exceeds its model/micro configuration ceiling")
        if not isinstance(device, int) or not 0 <= device < 2**31:
            raise ValueError("device must be a non-negative int32")
        if not isinstance(host_headroom_bytes, int) or not 0 <= host_headroom_bytes < 2**64:
            raise ValueError("headroom must be an unsigned 64-bit integer")
        if not isinstance(process_limit_bytes, int) or not 0 < process_limit_bytes <= (40 * 2**30 if model_windows or persistent_buffer else 2**30):
            raise ValueError("process cache limit exceeds its model/micro configuration ceiling")
        if not isinstance(registered_limit_bytes, int) or registered_limit_bytes < 0 or (
            registered_limit_bytes and (registered_limit_bytes % slot_bytes or registered_limit_bytes > slots * slot_bytes)
        ):
            raise ValueError("registration ceiling must cover whole segments within storage capacity")
        if registered_limit_bytes and not persistent_buffer:
            raise ValueError("a separate registration ceiling requires persistent_buffer")
        self.model_windows = bool(model_windows or persistent_buffer)
        self.persistent_buffer = bool(persistent_buffer)
        self.slot_bytes = slot_bytes
        self._owner = owner
        self._dll = owner._library
        self._lock = RLock()
        self._handle = c.c_void_p()
        self.source_dispatcher = False
        specs = {
            "source_cache_create": [c.POINTER(_Config), c.POINTER(c.c_void_p)],
            "source_cache_request": [c.c_void_p, c.c_void_p, c.c_uint64, c.c_size_t, c.POINTER(c.c_void_p)],
            "source_cache_snapshot": [c.c_void_p, c.POINTER(_Stats)],
            "source_cache_clear": [c.c_void_p], "source_cache_close": [c.POINTER(c.c_void_p)],
            "source_cache_trim": [c.c_void_p, c.c_uint64],
            "source_ticket_wait": [c.c_void_p],
            "source_ticket_copy_async": [c.c_void_p, c.c_void_p, c.c_size_t, c.c_void_p],
            "source_ticket_wait_on_stream": [c.c_void_p, c.c_void_p],
            "source_ticket_dma_ready": [c.c_void_p, c.POINTER(c.c_int)],
            "source_ticket_close": [c.POINTER(c.c_void_p)],
        }
        try:
            version = self._dll.ddo_vmm_source_cache_abi_version
            version.argtypes, version.restype = [], c.c_uint32
            abi_version = version()
            if abi_version not in (5, 6, 7, 8, 9, 10, 11):
                raise NativeVmmError("ticket polling source cache requires ABI v5")
            if process_limit_bytes > 128 * 2**20 and abi_version < 6:
                raise NativeVmmError("source cache process cap above 128 MiB requires ABI v6")
            if model_windows and abi_version < 7:
                raise NativeVmmError("shared model source windows require ABI v7")
            if persistent_buffer and abi_version < 8:
                raise NativeVmmError("persistent storage with independent CUDA registration requires ABI v8; rebuild allocator")
            self._stats_version = 8 if abi_version >= 11 else 7 if abi_version >= 10 else 6 if abi_version >= 9 else 5 if abi_version >= 8 else 4
            self._abi_version = abi_version
            if abi_version >= 11:
                specs["source_batch_create"] = [c.c_void_p, c.POINTER(_Span), c.c_size_t,
                    c.c_void_p, c.c_size_t, c.c_size_t, c.POINTER(c.c_void_p)]
                specs["source_batch_submit"] = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_void_p]
                specs["source_batch_wait_on_stream"] = [c.c_void_p, c.c_void_p]
                specs["source_batch_close"] = [c.POINTER(c.c_void_p)]
            if abi_version >= 8:
                specs["source_ticket_copy_via_ring_async"] = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_void_p, c.c_size_t, c.c_void_p]
            for name, args in specs.items():
                fn = getattr(self._dll, "ddo_vmm_" + name)
                fn.argtypes, fn.restype = args, c.c_int
            self._dll.ddo_vmm_source_cache_last_error.argtypes = []
            self._dll.ddo_vmm_source_cache_last_error.restype = c.c_char_p
            capability = getattr(self._dll, "ddo_vmm_source_dispatcher_enabled", None)
            prepare = getattr(self._dll, "ddo_vmm_source_batch_prepare", None)
            if persistent_buffer and capability is not None and prepare is not None:
                capability.argtypes, capability.restype = [], c.c_int
                prepare.argtypes, prepare.restype = [c.c_void_p], c.c_int
                self.source_dispatcher = bool(capability())
        except AttributeError as error:
            raise NativeVmmError("DLL lacks the experimental source cache; rebuild allocator") from error
        config_version = 5 if persistent_buffer else 4 if model_windows else 3 if process_limit_bytes > 128 * 2**20 else 2
        config = _Config(c.sizeof(_Config), config_version, device, slots, slot_bytes, host_headroom_bytes,
                         process_limit_bytes, registered_limit_bytes)
        self._check(self._dll.ddo_vmm_source_cache_create(c.byref(config), c.byref(self._handle)))

    def _check(self, status):
        if status:
            error = self._dll.ddo_vmm_source_cache_last_error()
            raise NativeSourceError(status, error.decode("utf-8", errors="replace") if error else f"cache status {status}")

    def _require_open(self):
        if not self._handle.value:
            raise NativeVmmError("source cache is closed")

    def request(self, source, offset, size):
        if not isinstance(offset, int) or not 0 <= offset < 2**64 or not isinstance(size, int) or not 0 < size < 2**64:
            raise ValueError("source range requires uint64 offset and positive size")
        with self._lock, source._lock:
            self._require_open()
            source._require_open()
            ticket = NativeSourceTicket(self)
            status = self._dll.ddo_vmm_source_cache_request(self._handle, source._handle, offset, size, c.byref(ticket._handle))
            if status == 8:
                return None  # Explicit bypass; no Python admission policy.
            self._check(status)
            return ticket

    def snapshot(self):
        with self._lock:
            self._require_open()
            stats = _Stats(struct_size=c.sizeof(_Stats), version=self._stats_version)
            self._check(self._dll.ddo_vmm_source_cache_snapshot(self._handle, c.byref(stats)))
            result = {name: getattr(stats, name) for name, _ in _Stats._fields_
                      if getattr(_Stats, name).offset < stats.struct_size}
            result.update(getattr(self, "_batch_timings", {}))
            event_snapshot = getattr(self._dll, "ddo_vmm_source_owned_events_v1", None)
            if event_snapshot is not None:
                event_snapshot.argtypes = [c.c_void_p] + [c.POINTER(c.c_size_t)] * 3
                event_snapshot.restype = c.c_int
                transfers, pooled, batches = c.c_size_t(), c.c_size_t(), c.c_size_t()
                self._check(event_snapshot(self._handle, c.byref(transfers), c.byref(pooled), c.byref(batches)))
                result.update(transfer_owned_events=transfers.value,
                              pooled_events=pooled.value, batch_completion_events=batches.value)
            if getattr(self, "source_dispatcher", False):
                result["source_dispatcher_enabled"] = 1
                dispatcher_snapshot = getattr(self._dll, "ddo_vmm_source_dispatcher_snapshot", None)
                if dispatcher_snapshot is not None:
                    dispatcher_snapshot.argtypes = [c.c_void_p] + [c.POINTER(c.c_size_t)] * 3
                    dispatcher_snapshot.restype = c.c_int
                    owned, peak, limit = c.c_size_t(), c.c_size_t(), c.c_size_t()
                    self._check(dispatcher_snapshot(self._handle, c.byref(owned), c.byref(peak), c.byref(limit)))
                    result.update(source_dispatcher_owned_windows=owned.value,
                                  source_dispatcher_peak_windows=peak.value,
                                  source_dispatcher_window_limit=limit.value)
            return result

    def create_batch(self, spans, ring, *, ring_slots, ring_slot_bytes):
        """Prepare descriptors once; native submission owns per-window tickets."""
        if self._abi_version < 11:
            raise NativeVmmError("native source batches require ABI v11; rebuild allocator")
        return NativeSourceBatch(self, spans, ring, ring_slots, ring_slot_bytes)

    def collect_completed(self):
        """Native, nonblocking retirement of batch-owned completed tickets only."""
        with self._lock:
            self._require_open()
            fn = getattr(self._dll, "ddo_vmm_source_cache_collect_completed", None)
            if fn is None:
                raise NativeVmmError("completed batch collection requires a rebuilt allocator")
            fn.argtypes, fn.restype = [c.c_void_p], c.c_int
            self._check(fn(self._handle))

    def clear(self):
        with self._lock:
            self._require_open()
            self._check(self._dll.ddo_vmm_source_cache_clear(self._handle))

    def trim(self, target_registered_bytes=0):
        if not isinstance(target_registered_bytes, int) or not 0 <= target_registered_bytes < 2**64:
            raise ValueError("trim target must be an unsigned 64-bit integer")
        with self._lock:
            self._require_open()
            self._check(self._dll.ddo_vmm_source_cache_trim(self._handle, target_registered_bytes))

    def close(self):
        with self._lock:
            self._check(self._dll.ddo_vmm_source_cache_close(c.byref(self._handle)))

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, _type, error, _traceback):
        _close_context_owner(self, error)

    def __del__(self):
        # Explicit close reports failures and leaves protection intact.
        with suppress(Exception):
            self.close()


class NativeSourceBatch:
    def __init__(self, cache, spans, ring, ring_slots, ring_slot_bytes):
        self._cache, self._ring = cache, ring
        self._handle = c.c_void_p()
        self._destinations = []
        if not spans:
            raise ValueError("native batch requires immutable source spans")
        with _measure_batch_scope(cache, "batch_bridge_create_total"), cache._lock, ExitStack() as source_locks:
            cache._require_open()
            # Serialize retention against owner close; ordering also avoids
            # inversions when distinct caches prepare overlapping source sets.
            for source in sorted({id(span[0]): span[0] for span in spans}.values(), key=id):
                source_locks.enter_context(source._lock)
            descriptors = []
            for source, offset, size, destination in spans:
                if (not all(isinstance(value, int) for value in (offset, size, destination))
                        or not 0 <= offset < 2**64 or not 0 < size < 2**64
                        or not 0 <= destination < 2**64):
                    raise ValueError("invalid source batch span")
                source._require_open()
                descriptors.append(_Span(source._handle, offset, size, destination))
            array = (_Span * len(descriptors))(*descriptors)
            ring_handle = ring._require_handle()
            output = c.byref(self._handle)
            with _measure_batch_scope(cache, "batch_bridge_create_ffi"):
                status = cache._dll.ddo_vmm_source_batch_create(cache._handle, array,
                    len(array), ring_handle, ring_slots, ring_slot_bytes, output)
            cache._check(status)

    def prepare_source(self):
        """Enqueue native source-only reads; never submit DMA or wait for I/O."""
        if not self._cache.source_dispatcher:
            return False
        with _measure_batch_scope(self._cache, "batch_bridge_source_prepare_total"), self._cache._lock:
            self._cache._require_open()
            if not self._handle.value:
                raise NativeVmmError("native source batch is closed")
            self._cache._check(self._cache._dll.ddo_vmm_source_batch_prepare(self._handle))
        return True

    def submit(self, arena, stream, dest_base=0):
        if not isinstance(dest_base, int) or not 0 <= dest_base < 2**64:
            raise ValueError("batch destination base must be uint64")
        with _measure_batch_scope(self._cache, "batch_bridge_submit_total"):
            locked = time.perf_counter()
            with self._cache._lock:
                wait = time.perf_counter() - locked
                metrics = getattr(self._cache, "_batch_timings", None)
                if metrics is None:
                    metrics = self._cache._batch_timings = {}
                metrics["batch_bridge_submit_lock_wait_seconds"] = metrics.get("batch_bridge_submit_lock_wait_seconds", 0.0) + wait
                with _measure_batch_scope(self._cache, "batch_bridge_submit_prepare"):
                    if not self._handle.value:
                        raise NativeVmmError("native source batch is closed")
                    # Keep all potential destinations alive, even on partial failure.
                    if not any(arena is previous for previous in self._destinations):
                        self._destinations.append(arena)
                    destination, upload = arena._require_handle(), _stream(stream)
                with _measure_batch_scope(self._cache, "batch_bridge_submit_ffi"):
                    status = self._cache._dll.ddo_vmm_source_batch_submit(
                        self._handle, destination, dest_base, upload)
                with _measure_batch_scope(self._cache, "batch_bridge_submit_result"):
                    self._cache._check(status)

    def close(self):
        with self._cache._lock:
            self._cache._check(self._cache._dll.ddo_vmm_source_batch_close(c.byref(self._handle)))
            self._destinations.clear()

    def wait_on_stream(self, stream):
        with self._cache._lock:
            self._cache._check(self._cache._dll.ddo_vmm_source_batch_wait_on_stream(
                self._handle, _stream(stream)))


class NativeSourceTicket:
    def __init__(self, cache):
        self._cache = cache
        self._handle = c.c_void_p()
        self._arena = None
        self._ring = None

    def _require_open(self):
        if not self._handle.value:
            raise NativeVmmError("source ticket is closed")

    def wait(self):
        with self._cache._lock:
            self._require_open()
            self._cache._check(self._cache._dll.ddo_vmm_source_ticket_wait(self._handle))

    def copy_to_arena_async(self, arena, offset=0, stream=None):
        if not isinstance(offset, int) or not 0 <= offset < 2**64:
            raise ValueError("arena offset must be uint64")
        with self._cache._lock:
            self._require_open()
            if self._arena is not None:
                raise NativeVmmError("a source ticket permits only one DMA attempt")
            self._arena = arena  # Keep the destination owner alive through DMA.
            status = self._cache._dll.ddo_vmm_source_ticket_copy_async(
                self._handle, arena._require_handle(), offset, _stream(stream))
            if status == 8:
                self._arena = None  # Registration refusal did not submit DMA.
            self._cache._check(status)

    def copy_via_ring_async(self, ring, slot, arena, offset=0, stream=None):
        """Registration-refusal fallback, entirely native, without rereading."""
        if not isinstance(offset, int) or not 0 <= offset < 2**64 or not isinstance(slot, int) or slot < 0:
            raise ValueError("fallback requires a uint64 offset and non-negative ring slot")
        with self._cache._lock:
            self._require_open()
            if self._arena is not None:
                raise NativeVmmError("a submitted source ticket cannot be sent through fallback")
            self._arena, self._ring = arena, ring
            self._cache._check(self._cache._dll.ddo_vmm_source_ticket_copy_via_ring_async(
                self._handle, ring._require_handle(), slot, arena._require_handle(), offset, _stream(stream)))

    def wait_on_stream(self, stream=None):
        with self._cache._lock:
            self._require_open()
            self._cache._check(self._cache._dll.ddo_vmm_source_ticket_wait_on_stream(self._handle, _stream(stream)))

    def dma_ready(self):
        with self._cache._lock:
            self._require_open()
            ready = c.c_int()
            self._cache._check(self._cache._dll.ddo_vmm_source_ticket_dma_ready(self._handle, c.byref(ready)))
            return bool(ready.value)

    def close(self):
        with self._cache._lock:
            self._cache._check(self._cache._dll.ddo_vmm_source_ticket_close(c.byref(self._handle)))
            self._arena = None
            self._ring = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, _type, error, _traceback):
        _close_context_owner(self, error)

    def __del__(self):
        with suppress(Exception):
            self.close()
