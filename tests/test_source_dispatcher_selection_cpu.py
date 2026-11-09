"""CPU contracts for the per-run source dispatcher selection (fake DLL, no CUDA)."""
from types import SimpleNamespace

import pytest

from ddo_vmm_allocator import source_cache
from ddo_vmm_allocator.source import NativeSourceError
from ddo_vmm_allocator.native import NativeVmmError


class Export:
    def __init__(self, calls, name, result=0):
        self.calls, self.name, self.result = calls, name, result

    def __call__(self, *args):
        self.calls.append(self.name)
        return self.result() if callable(self.result) else self.result


class Dll:
    def __init__(self, *, dispatcher_build=True, select_export=True, select_status=0):
        self.calls = []
        results = {"ddo_vmm_source_cache_abi_version": 11,
                   "ddo_vmm_source_dispatcher_enabled": int(dispatcher_build),
                   "ddo_vmm_source_cache_select_dispatcher": select_status,
                   "ddo_vmm_source_cache_last_error": b"refused"}
        self._missing = set() if select_export else {"ddo_vmm_source_cache_select_dispatcher"}
        self._exports = {}
        self._results = results

    def __getattr__(self, name):
        if name.startswith("_") or name in self._missing:
            raise AttributeError(name)
        return self._exports.setdefault(name, Export(self.calls, name, self._results.get(name, 0)))


@pytest.fixture(autouse=True)
def restore_selection():
    previous = source_cache.source_dispatcher_selected()
    yield
    source_cache.configure_source_dispatcher(previous)


def cache(dll, **kwargs):
    owner = SimpleNamespace(_library=dll)
    return source_cache.NativeSourceCache(owner, slot_bytes=64 * 2**20, slots=4,
                                          process_limit_bytes=256 * 2**20, persistent_buffer=True, **kwargs)


def test_default_selection_keeps_the_dispatcher_of_a_dispatcher_build():
    dll = Dll()
    assert cache(dll).source_dispatcher is True
    assert "ddo_vmm_source_cache_select_dispatcher" not in dll.calls


def test_disabled_selection_turns_it_off_after_create_before_any_request():
    source_cache.configure_source_dispatcher(False)
    dll = Dll()
    instance = cache(dll)
    assert instance.source_dispatcher is False
    create = dll.calls.index("ddo_vmm_source_cache_create")
    select = dll.calls.index("ddo_vmm_source_cache_select_dispatcher")
    assert create < select and "ddo_vmm_source_cache_request" not in dll.calls


def test_disabled_selection_is_a_noop_without_the_dispatcher_build():
    source_cache.configure_source_dispatcher(False)
    dll = Dll(dispatcher_build=False, select_export=False)
    assert cache(dll).source_dispatcher is False
    assert "ddo_vmm_source_cache_select_dispatcher" not in dll.calls


def test_dispatcher_build_without_selection_export_refuses_to_disable():
    source_cache.configure_source_dispatcher(False)
    dll = Dll(select_export=False)
    with pytest.raises(NativeVmmError, match="cannot disable"):
        cache(dll)
    assert "ddo_vmm_source_cache_create" not in dll.calls


def test_native_refusal_closes_the_cache():
    source_cache.configure_source_dispatcher(False)
    dll = Dll(select_status=4)
    with pytest.raises(NativeSourceError):
        cache(dll)
    assert dll.calls[-1] == "ddo_vmm_source_cache_close"


def test_selection_requires_a_bool():
    with pytest.raises(TypeError):
        source_cache.configure_source_dispatcher(1)
