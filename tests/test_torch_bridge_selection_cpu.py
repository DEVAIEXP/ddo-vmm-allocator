"""CPU contracts for the optional checked Torch bridge selection (no CUDA, no real bridge)."""
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from ddo_vmm_allocator import _library, torch_allocator

TORCH_VERSION, TORCH_GIT = "2.12.1+cu132", "7269437d655783a26cba32aa88195b741ff496aa"


class Export:
    def __init__(self, calls, name):
        self.calls, self.name = calls, name

    def __call__(self, *args):
        self.calls.append((self.name, args))
        return 0


class Library:
    def __init__(self, calls, missing=("ddo_vmm_set_recovery_warns_v1",)):
        self._calls, self._missing, self._exports = calls, set(missing), {}

    def __getattr__(self, name):
        if name.startswith("_") or name in self._missing:
            raise AttributeError(name)
        return self._exports.setdefault(name, Export(self._calls, name))


@pytest.fixture
def setup(monkeypatch, tmp_path):
    library_path = tmp_path / "ddo_vmm.dll"
    library_path.touch()
    calls = []
    addresses = {"ddo_vmm_torch_malloc": 0x1000, "ddo_vmm_torch_free": 0x2000, "ddo_exp_segmented_last_refusal": 0x3000}
    library = Library(calls)
    monkeypatch.setattr(_library.platform, "system", lambda: "Windows")
    monkeypatch.setattr(_library.ctypes, "WinDLL", lambda value: library, raising=False)
    import ctypes
    real_cast = ctypes.cast

    def cast(value, kind):
        if isinstance(value, Export):
            return SimpleNamespace(value=addresses[value.name])
        return real_cast(value, kind)

    monkeypatch.setattr(ctypes, "cast", cast)
    memory = SimpleNamespace(
        CUDAPluggableAllocator=lambda *args: calls.append(("pluggable", args)) or "allocator",
        change_current_allocator=lambda allocator: calls.append(("install", (allocator,))))
    cuda = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False, memory=memory)
    torch = SimpleNamespace(cuda=cuda, __version__=TORCH_VERSION, version=SimpleNamespace(git_version=TORCH_GIT))
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name, value in (("_installed_allocator", None), ("_configured_library", None),
                        ("_allocation_boundary", None),
                        ("_pool_configuration", {"segmented_auto": False, "growable_reserve_bytes": 0})):
        monkeypatch.setattr(torch_allocator, name, value)
    for name in (torch_allocator.SEGMENTED_AUTO_ENVIRONMENT, torch_allocator.GROWABLE_RESERVE_ENVIRONMENT,
                 "DDO_SHOW_RECOVERY_WARNS", torch_allocator.TORCH_BRIDGE_ENVIRONMENT):
        monkeypatch.delenv(name, raising=False)
    bridge = SimpleNamespace(BRIDGE_ABI=1, install=lambda *args: calls.append(("bridge_install", args)))
    loaded = []

    def write_bridge(**overrides):
        module = library_path.parent / "ddo_vmm_torch_bridge.cp312-win_amd64.pyd"
        module.write_bytes(b"bridge bytes")
        identity = {"abi": 1, "torch_version": TORCH_VERSION, "torch_git": TORCH_GIT,
                    "python": f"cp{sys.version_info.major}{sys.version_info.minor}", "platform": sys.platform,
                    "module": module.name, "sha256": hashlib.sha256(b"bridge bytes").hexdigest()}
        identity.update(overrides)
        (library_path.parent / torch_allocator.TORCH_BRIDGE_IDENTITY).write_text(json.dumps(identity))

    def exec_module(module):
        loaded.append(module)
        module.BRIDGE_ABI = bridge.BRIDGE_ABI
        module.install = bridge.install

    import importlib.util as util
    real_spec = util.spec_from_file_location
    monkeypatch.setattr(util, "spec_from_file_location", lambda name, path: SimpleNamespace(
        name=name, loader=SimpleNamespace(exec_module=exec_module, create_module=lambda spec: None),
        origin=str(path), submodule_search_locations=None) if name == "ddo_vmm_torch_bridge" else real_spec(name, path))
    monkeypatch.setattr(util, "module_from_spec", lambda spec: SimpleNamespace())

    def run(**kwargs):
        return torch_allocator.install_pytorch_vmm_allocator(library_path, **kwargs)

    return SimpleNamespace(run=run, calls=calls, write_bridge=write_bridge, bridge=bridge, loaded=loaded,
                           torch=torch)


def names(calls):
    return [name for name, _ in calls]


def test_without_identity_the_plain_pluggable_path_is_unchanged(setup):
    assert setup.run() == "allocator"
    assert names(setup.calls) == ["pluggable", "install"]
    assert torch_allocator.installed_allocation_boundary() == "unguarded"


def test_matching_identity_registers_the_same_exports_through_the_bridge(setup):
    setup.write_bridge()
    setup.run(segmented_auto=True, growable_reserve_bytes=8 * 1024**3)
    # Pool configuration still happens first, on the same loaded library.
    assert names(setup.calls) == ["ddo_exp_segmented_configure_auto", "ddo_exp_segmented_configure_growable",
                                  "bridge_install"]
    assert setup.calls[-1] == ("bridge_install", (0x1000, 0x2000, 0x3000))
    assert torch_allocator.installed_allocation_boundary() == "checked"
    assert torch_allocator._configured_library is not None
    assert torch_allocator.installed_pool_configuration()["segmented_auto"] is True


@pytest.mark.parametrize("field, value", [("torch_version", "2.13.0+cu132"), ("torch_git", "other"),
                                          ("python", "cp311"), ("platform", "linux"), ("abi", 2)])
def test_identity_mismatch_keeps_the_plain_path(setup, field, value):
    setup.write_bridge(**{field: value})
    setup.run()
    assert names(setup.calls) == ["pluggable", "install"] and setup.loaded == []
    assert torch_allocator.installed_allocation_boundary() == "unguarded"


def test_disabled_bridge_keeps_the_plain_path(setup, monkeypatch):
    setup.write_bridge()
    monkeypatch.setenv(torch_allocator.TORCH_BRIDGE_ENVIRONMENT, "0")
    setup.run()
    assert names(setup.calls) == ["pluggable", "install"] and setup.loaded == []


@pytest.mark.parametrize("overrides", [None, {"torch_git": "other"}])
def test_required_bridge_refuses_when_unavailable(setup, monkeypatch, overrides):
    if overrides is not None:
        setup.write_bridge(**overrides)
    monkeypatch.setenv(torch_allocator.TORCH_BRIDGE_ENVIRONMENT, "1")
    with pytest.raises(torch_allocator.TorchAllocatorInstallError):
        setup.run()
    assert "install" not in names(setup.calls) and "bridge_install" not in names(setup.calls)
    assert torch_allocator._installed_allocator is None


def test_damaged_bridge_is_refused_before_native_configuration(setup):
    setup.write_bridge(sha256="0" * 64)
    with pytest.raises(torch_allocator.TorchAllocatorInstallError, match="hash mismatch"):
        setup.run(segmented_auto=True, growable_reserve_bytes=1024**3)
    assert setup.calls == []


def test_invalid_bridge_mode_is_refused(setup, monkeypatch):
    monkeypatch.setenv(torch_allocator.TORCH_BRIDGE_ENVIRONMENT, "yes")
    with pytest.raises(torch_allocator.TorchAllocatorInstallError, match="auto, 0 or 1"):
        setup.run()
    assert setup.calls == []


def test_missing_last_refusal_export_still_installs_the_bridge(setup, monkeypatch):
    setup.write_bridge()
    library = Library(setup.calls, missing=("ddo_vmm_set_recovery_warns_v1", "ddo_exp_segmented_last_refusal"))
    monkeypatch.setattr(_library.ctypes, "WinDLL", lambda value: library, raising=False)
    setup.run()
    assert setup.calls[-1] == ("bridge_install", (0x1000, 0x2000, 0))
