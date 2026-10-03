"""Bounded native CUDA window observations, never a residency controller."""
import ctypes
import math

from .native import NativeVmmError

KINDS = {1: "upload_batch_window", 2: "compute_upload_dependency",
         3: "upload_slot_dependency", 4: "execution_unit_window", 5: "step_marker"}


def union_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def summarize_windows(samples, *, start=None, end=None):
    windows = {}
    for kind in KINDS.values():
        if kind == "step_marker":
            continue
        spans = [(max(s["start_seconds"], start) if start is not None else s["start_seconds"],
                  min(s["end_seconds"], end) if end is not None else s["end_seconds"])
                 for s in samples if s["status"] == "complete" and s["kind"] == kind]
        windows[kind] = union_intervals(spans)
    a, b = windows["upload_batch_window"], windows["execution_unit_window"]
    i = j = 0
    intersection = 0.
    while i < len(a) and j < len(b):
        intersection += max(0., min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return dict(scope="window_geometry_not_engine_activity",
                union_seconds={k: sum(y - x for x, y in spans) for k, spans in windows.items()},
                upload_execution_window_intersection_seconds=intersection)


class Sample(ctypes.Structure):
    _fields_ = [(k, ctypes.c_uint64) for k in ("sequence", "unit", "stream", "bytes")] + [
        ("kind", ctypes.c_uint32), ("state", ctypes.c_uint32),
        ("start_ms", ctypes.c_float), ("end_ms", ctypes.c_float)]


class Info(ctypes.Structure):
    _fields_ = [(k, ctypes.c_uint64) for k in ("capacity", "count", "dropped", "failures", "pending")] + [
        ("sealed", ctypes.c_uint32), ("version", ctypes.c_uint32)]


def decode_snapshot(info, rows):
    if info.version != 1 or info.count > len(rows) or info.count > info.capacity:
        raise NativeVmmError("invalid native timeline snapshot schema/capacity")
    samples = []
    for row in rows[:info.count]:
        value = dict(sequence=row.sequence, unit=row.unit, stream=row.stream,
                     bytes=row.bytes or None, kind=KINDS.get(row.kind, "unknown"),
                     status={0: "pending", 1: "complete", 2: "failed"}.get(row.state, "failed"))
        if row.state == 1:
            start, end = row.start_ms / 1000, row.end_ms / 1000
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                value["status"] = "failed"
            else:
                value.update(start_seconds=start, end_seconds=end)
        samples.append(value)
    failed = info.failures or any(s["status"] == "failed" for s in samples)
    status = ("failed" if failed else "truncated" if info.dropped else
              "pending" if info.pending or not info.sealed or
              any(s["status"] == "pending" for s in samples) else "complete")
    report = dict(schema_version=1, clock="cuda_event", unit="seconds",
                scope="native_stream_windows_not_engine_activity", status=status,
                capacity=info.capacity, dropped=info.dropped, failures=info.failures,
                pending=info.pending, samples=samples)
    if status == "complete":
        report["window_summary"] = summarize_windows(samples)
        markers = {s["unit"]: s["start_seconds"] for s in samples
                   if s["kind"] == "step_marker" and s["status"] == "complete"}
        if 1 in markers and 2**64 - 1 in markers:
            report["warm_window_summary"] = summarize_windows(
                samples, start=markers[1], end=markers[2**64 - 1])
    return report


class NativeTimeline:
    """Events remain native-owned until completion is proven without a wait."""
    def __init__(self, owner, *, capacity, stream):
        if not isinstance(capacity, int) or not 1 <= capacity <= 8192:
            raise ValueError("timeline capacity must be in 1..8192")
        self._owner = owner
        self._handle = ctypes.c_void_p()
        self.capacity = capacity
        self._sealed = False
        self._saved = None
        dll = owner._library
        specs = {
            "open": ([ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            "mark_begin": ([ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p], ctypes.c_uint64),
            "mark_end": ([ctypes.c_uint64, ctypes.c_void_p], None),
            "seal": ([ctypes.c_void_p], ctypes.c_int),
            "snapshot": ([ctypes.c_void_p, ctypes.POINTER(Info), ctypes.POINTER(Sample), ctypes.c_size_t], ctypes.c_int),
            "close": ([ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
        }
        self._functions = {}
        for name, (args, result) in specs.items():
            fn = getattr(dll, f"ddo_vmm_timeline_{name}_v1", None)
            if fn is None:
                raise NativeVmmError("native timeline requires a rebuilt allocator DLL")
            fn.argtypes, fn.restype = args, result
            self._functions[name] = fn
        if self._functions["open"](capacity, stream, ctypes.byref(self._handle)):
            raise NativeVmmError("native timeline initialization failed")

    def begin_unit(self, unit, stream):
        return self._functions["mark_begin"](4, unit, 0, stream)

    def end_unit(self, token, stream):
        self._functions["mark_end"](token, stream)

    def mark_boundary(self, index, stream):
        token = self._functions["mark_begin"](5, index, 0, stream)
        self._functions["mark_end"](token, stream)

    def seal(self):
        if self._handle.value and not self._sealed:
            if self._functions["seal"](self._handle):
                raise NativeVmmError("native timeline seal failed")
            self._sealed = True

    def snapshot(self):
        if self._saved is not None:
            return self._saved
        info, rows = Info(), (Sample * self.capacity)()
        if self._functions["snapshot"](self._handle, ctypes.byref(info), rows, self.capacity):
            raise NativeVmmError("native timeline snapshot failed")
        report = decode_snapshot(info, rows)
        report["execution_units"] = getattr(self, "units", {})
        if self._sealed and not info.pending and report["status"] not in ("failed", "pending"):
            report["event_cleanup"] = "released" if self._functions["close"](ctypes.byref(self._handle)) == 0 else "retained"
            if not self._handle.value:
                self._saved = report
        return report

    def close(self):
        """Nonblocking; return False while native owner must retain events."""
        if not self._handle.value:
            return True
        self.seal()
        # Preserve the final report before releasing its native events. A pending
        # snapshot is not a promise that the GPU will still be pending on the
        # next query, so do not issue an independent close after that snapshot.
        self.snapshot()
        return not self._handle.value
