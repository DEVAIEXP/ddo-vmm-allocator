"""CPU contracts for the installer's cold pool configuration (no CUDA)."""
from types import SimpleNamespace
import sys

import pytest

from ddo_vmm_allocator import _library, torch_allocator


class Export:
    def __init__(self, calls, name, status=0):
        self.calls, self.name, self.status = calls, name, status

    def __call__(self, *args):
        self.calls.append((self.name, args))
        return self.status


class Library:
    def __init__(self, calls, *, missing=(), statuses=None):
        self._calls, self._missing, self._statuses = calls, set(missing), statuses or {}
        self._exports = {}

    def __getattr__(self, name):
        if name.startswith("_") or name in self._missing:
            raise AttributeError(name)
        return self._exports.setdefault(name, Export(self._calls, name, self._statuses.get(name, 0)))


@pytest.fixture
def install(monkeypatch, tmp_path):
    path = tmp_path / "selected.dll"
    path.touch()
    calls = []

    def run(library=None, **kwargs):
        selected = library or Library(calls, missing=("ddo_vmm_set_recovery_warns_v1",))
        monkeypatch.setattr(_library.platform, "system", lambda: "Windows")
        monkeypatch.setattr(_library.ctypes, "WinDLL", lambda value: selected, raising=False)
        memory = SimpleNamespace(
            CUDAPluggableAllocator=lambda *args: calls.append(("pluggable", args)) or "allocator",
            change_current_allocator=lambda allocator: calls.append(("install", (allocator,))))
        cuda = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False, memory=memory)
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
        monkeypatch.setattr(torch_allocator, "_installed_allocator", None)
        monkeypatch.setattr(torch_allocator, "_configured_library", None)
        monkeypatch.setattr(torch_allocator, "_pool_configuration",
                            {"segmented_auto": False, "growable_reserve_bytes": 0})
        return torch_allocator.install_pytorch_vmm_allocator(path, **kwargs)

    for name in (torch_allocator.SEGMENTED_AUTO_ENVIRONMENT, torch_allocator.GROWABLE_RESERVE_ENVIRONMENT):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("DDO_SHOW_RECOVERY_WARNS", raising=False)
    run.calls = calls
    return run


def names(calls):
    return [name for name, _ in calls]


def test_default_leaves_pool_unconfigured(install):
    assert install() == "allocator"
    assert names(install.calls) == ["pluggable", "install"]
    assert torch_allocator.installed_pool_configuration() == {"segmented_auto": False, "growable_reserve_bytes": 0}


@pytest.mark.parametrize("enabled", [False, True])
def test_recovery_notices_are_applied_before_allocator_install(install, enabled):
    install(Library(install.calls), show_recovery_warns=enabled)
    assert install.calls[0] == ("ddo_vmm_set_recovery_warns_v1", (int(enabled),))
    assert names(install.calls)[-2:] == ["pluggable", "install"]


def test_recovery_notification_configuration_failure_is_visible(install):
    library = Library(install.calls, statuses={"ddo_vmm_set_recovery_warns_v1": 1})
    with pytest.raises(torch_allocator.TorchAllocatorInstallError, match="notification configuration failed"):
        install(library)
    assert "install" not in names(install.calls)


def test_explicit_configuration_runs_before_install_and_retains_library(install):
    install(segmented_auto=True, growable_reserve_bytes=8 * 1024**3)
    assert install.calls == [
        ("ddo_exp_segmented_configure_auto", (1,)),
        ("ddo_exp_segmented_configure_growable", (8 * 1024**3,)),
        ("pluggable", (str(torch_allocator._installed_library_path), "ddo_vmm_torch_malloc", "ddo_vmm_torch_free")),
        ("install", ("allocator",)),
    ]
    assert torch_allocator._configured_library is not None
    assert torch_allocator.installed_pool_configuration() == {
        "segmented_auto": True, "growable_reserve_bytes": 8 * 1024**3}


def test_bootstrap_environment_is_read_when_arguments_are_absent(install, monkeypatch):
    monkeypatch.setenv(torch_allocator.SEGMENTED_AUTO_ENVIRONMENT, "1")
    monkeypatch.setenv(torch_allocator.GROWABLE_RESERVE_ENVIRONMENT, str(1024**3))
    install()
    assert names(install.calls)[:2] == ["ddo_exp_segmented_configure_auto", "ddo_exp_segmented_configure_growable"]


def test_explicit_arguments_override_environment(install, monkeypatch):
    monkeypatch.setenv(torch_allocator.SEGMENTED_AUTO_ENVIRONMENT, "1")
    monkeypatch.setenv(torch_allocator.GROWABLE_RESERVE_ENVIRONMENT, str(1024**3))
    install(segmented_auto=False, growable_reserve_bytes=0)
    assert names(install.calls) == ["pluggable", "install"]


def test_auto_without_growable_skips_reserve(install):
    install(segmented_auto=True, growable_reserve_bytes=0)
    assert names(install.calls) == ["ddo_exp_segmented_configure_auto", "pluggable", "install"]


@pytest.mark.parametrize("kwargs, message", [
    (dict(segmented_auto=False, growable_reserve_bytes=1024**3), "requires automatic segmented"),
    (dict(segmented_auto=True, growable_reserve_bytes=-1), "growable reserve must be within"),
    (dict(segmented_auto=True, growable_reserve_bytes=64 * 1024**3 + 1), "growable reserve must be within"),
])
def test_invalid_configuration_is_refused_before_loading(install, kwargs, message):
    with pytest.raises(torch_allocator.TorchAllocatorInstallError, match=message):
        install(**kwargs)
    assert install.calls == []
    assert torch_allocator._installed_allocator is None


def test_invalid_environment_values_are_refused(install, monkeypatch):
    monkeypatch.setenv(torch_allocator.SEGMENTED_AUTO_ENVIRONMENT, "yes")
    with pytest.raises(torch_allocator.TorchAllocatorInstallError, match="must be 0 or 1"):
        install()


def test_library_without_configuration_exports_is_refused(install):
    library = Library(install.calls, missing=["ddo_exp_segmented_configure_growable"])
    with pytest.raises(torch_allocator.TorchAllocatorInstallError, match="ddo_exp_segmented_configure_growable"):
        install(library, segmented_auto=True, growable_reserve_bytes=1024**3)
    assert install.calls == []
    assert torch_allocator._installed_allocator is None


@pytest.mark.parametrize("export", ["ddo_exp_segmented_configure_auto", "ddo_exp_segmented_configure_growable"])
def test_refused_native_configuration_does_not_install(install, export):
    library = Library(install.calls, statuses={export: 1})
    with pytest.raises(torch_allocator.TorchAllocatorInstallError, match="refused"):
        install(library, segmented_auto=True, growable_reserve_bytes=1024**3)
    assert "install" not in names(install.calls)
    assert torch_allocator._installed_allocator is None
    assert torch_allocator.installed_pool_configuration()["segmented_auto"] is False
