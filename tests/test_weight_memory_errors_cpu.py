"""Native OUT_OF_MEMORY for model weights becomes a friendly OOM; other failures are unchanged (CPU)."""
from types import SimpleNamespace

import pytest

from ddo_vmm_allocator import native


class Library:
    def __init__(self, status):
        self.status = status
        self._library = self

    def ddo_vmm_execution_pool_acquire(self, *args):
        return self.status

    def ddo_vmm_model_arena_fault_range(self, *args):
        return self.status

    def _last_error(self):
        return "model arena admission boundary reached by CUDA allocation pressure: cuMemCreate failed"


def pool(status):
    instance = object.__new__(native.NativeExecutionPool)
    instance._owner, instance._handle, instance._arenas = Library(status), SimpleNamespace(value=1), {}
    instance.capacity_bytes = 304 * 2**20
    return instance


def arena(status):
    instance = object.__new__(native.NativeModelArena)
    instance._owner, instance._handle = Library(status), object()
    return instance


def calls():
    stream = SimpleNamespace(cuda_stream=0)
    return [lambda status: pool(status).acquire(stream), lambda status: arena(status).fault_range(0, 2 * 2**20)]


@pytest.mark.parametrize("index, purpose, mib", [(0, "execution slot", "304.0"), (1, "resident weight range", "2.0")])
def test_native_out_of_memory_is_a_friendly_oom_and_still_a_native_error(index, purpose, mib):
    with pytest.raises(native.NativeVmmOutOfMemoryError) as caught:
        calls()[index](2)
    error = caught.value
    assert isinstance(error, native.NativeVmmError)
    message = str(error)
    assert f"could not obtain {mib} MiB of GPU memory for model weights ({purpose})" in message
    assert "Other applications may be using GPU memory" in message
    assert "cuMemCreate failed" in message  # Native detail preserved.
    torch = pytest.importorskip("torch")
    assert isinstance(error, torch.OutOfMemoryError)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("status", [1, 3, 4, 5])
def test_other_native_failures_keep_the_plain_error(index, status):
    with pytest.raises(native.NativeVmmError) as caught:
        calls()[index](status)
    assert not isinstance(caught.value, native.NativeVmmOutOfMemoryError)
    torch = pytest.importorskip("torch")
    assert not isinstance(caught.value, torch.OutOfMemoryError)


def test_existing_native_error_handlers_still_catch_the_oom():
    try:
        calls()[1](2)
    except native.NativeVmmError as error:  # Adapter fallbacks catch this type.
        assert isinstance(error, native.NativeVmmOutOfMemoryError)
    else:
        raise AssertionError("expected an error")
