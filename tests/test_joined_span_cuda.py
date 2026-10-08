"""Small direct joined-span gate; run only with explicit CUDA approval."""
import ctypes as c
import os
import sys

import pytest


@pytest.mark.vmm
@pytest.mark.parametrize("cross_stream", [False, True])
def test_joined_span_waits_for_both_predecessors(monkeypatch, cross_stream):
    if sys.platform != "win32" or not os.environ.get("DDO_VMM_LIBRARY_PATH"):
        pytest.skip("requires Windows and an explicitly selected candidate DLL")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setenv("DDO_VMM_TORCH_CACHE_MIB", "4")
    dll = c.CDLL(os.environ["DDO_VMM_LIBRARY_PATH"])
    allocate = dll.ddo_vmm_torch_malloc
    allocate.argtypes = [c.c_size_t, c.c_int, c.c_void_p]
    allocate.restype = c.c_void_p
    release = dll.ddo_vmm_torch_free
    release.argtypes = [c.c_void_p, c.c_size_t, c.c_int, c.c_void_p]
    release.restype = None
    stats = dll.ddo_vmm_torch_pool_stats_v1
    stats.argtypes = [c.POINTER(c.c_uint64), c.c_size_t]
    stats.restype = c.c_int
    driver = c.WinDLL("nvcuda.dll")
    fill = driver.cuMemsetD32Async
    fill.argtypes = [c.c_uint64, c.c_uint, c.c_size_t, c.c_void_p]
    fill.restype = c.c_int
    read = driver.cuMemcpyDtoH_v2
    read.argtypes = [c.c_void_p, c.c_uint64, c.c_size_t]
    read.restype = c.c_int
    device = torch.cuda.current_device()
    producer_a, producer_b = torch.cuda.Stream(device), torch.cuda.Stream(device)
    consumer = torch.cuda.Stream(device) if cross_stream else producer_a
    size, half = 2 * 1024**2, 1024**2
    owned = {}

    def acquire(count, stream):
        pointer = allocate(count, device, stream.cuda_stream)
        assert pointer
        owned[pointer] = (count, stream)
        return pointer

    def retire(pointer):
        count, stream = owned.pop(pointer)
        release(pointer, count, device, stream.cuda_stream)

    def snapshot():
        values = (c.c_uint64 * 15)()
        assert stats(values, 15) == 0
        return list(values)

    try:
        seed = acquire(size, producer_a)
        retire(seed)
        left = acquire(half, producer_a)
        right = acquire(half, producer_b)
        assert left == seed and right == seed + half, "must split one backing"
        done = []
        for pointer, stream in ((left, producer_a), (right, producer_b)):
            with torch.cuda.stream(stream):
                torch.cuda._sleep(500_000_000)
                assert fill(pointer, 3, half // 4, stream.cuda_stream) == 0
                event = torch.cuda.Event()
                event.record(stream)
                done.append(event)
            retire(pointer)
        assert all(not event.query() for event in done), "must exercise pending predecessors"
        before = snapshot()
        joined = acquire(size, consumer)
        assert joined == seed, "must join the original two spans, not grow backing"
        after = snapshot()
        assert after[1] == before[1] and after[9] == before[9]
        assert after[6] == before[6] == 0 and after[13] == before[13] == 0
        assert fill(joined, 7, size // 4, consumer.cuda_stream) == 0
        consumer.synchronize()
        producer_a.synchronize()
        producer_b.synchronize()
        result = (c.c_uint * (size // 4))()
        assert read(result, joined, size) == 0
        assert all(value == 7 for value in result), "a predecessor overwrote joined storage"
        retire(joined)
        assert snapshot()[3] == 0 and snapshot()[6] == 0
    finally:
        producer_a.synchronize()
        producer_b.synchronize()
        consumer.synchronize()
        for pointer in list(owned):
            retire(pointer)
