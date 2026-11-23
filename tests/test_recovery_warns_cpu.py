"""Notification visibility is independent of native allocation ownership."""

from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from ddo_vmm_allocator.native import NativeVmmLibrary, NativeVmmError


def test_optional_binding_reports_unavailable_on_legacy_binary():
    library = NativeVmmLibrary.__new__(NativeVmmLibrary)
    library._library = SimpleNamespace()
    assert library.set_recovery_warns(False) is False
    with pytest.raises(TypeError, match="boolean"):
        library.set_recovery_warns(1)


def test_binding_propagates_native_configuration_errors():
    class Export:
        def __call__(self, enabled):
            assert enabled == 1
            return 1

    library = NativeVmmLibrary.__new__(NativeVmmLibrary)
    library._library = SimpleNamespace(ddo_vmm_set_recovery_warns_v1=Export())
    with pytest.raises(NativeVmmError, match="notification configuration failed"):
        library.set_recovery_warns(True)


def test_native_visibility_contract_without_cuda(tmp_path):
    root = Path(__file__).parents[1]
    cmake = shutil.which("cmake")
    assert cmake, "CMake and a C++ compiler are required for the CPU notification contract"
    test = tmp_path / "notification_contract.cpp"
    test.write_text('''#include "ddo_vmm.h"
#include "ddo_vmm_recovery_warnings.h"
#include <cassert>
int main() {
    using namespace ddo_vmm_detail;
    assert(!show_recovery_warnings());
    assert(!show_capacity_report(true));
    assert(show_capacity_report(false));
    assert(ddo_vmm_set_recovery_warns_v1(1) == 0);
    assert(show_recovery_warnings() && show_capacity_report(true));
    assert(ddo_vmm_set_recovery_warns_v1(2) == 1);
    assert(show_recovery_warnings());
    assert(ddo_vmm_set_recovery_warns_v1(0) == 0);
    assert(!show_capacity_report(true));
    assert(show_capacity_report(false));
}
''')
    (tmp_path / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.25)
project(notification_contract LANGUAGES CXX)
add_executable(notification_contract "{test.as_posix()}"
    "{(root / 'native/src/ddo_vmm_recovery_warnings.cpp').as_posix()}")
target_compile_features(notification_contract PRIVATE cxx_std_20)
target_compile_definitions(notification_contract PRIVATE DDO_VMM_CPU_API)
target_include_directories(notification_contract PRIVATE "{(root / 'native/include').as_posix()}")
''')
    build = tmp_path / "build"
    subprocess.run([cmake, "-S", str(tmp_path), "-B", str(build)],
                   check=True, capture_output=True, text=True, timeout=60)
    subprocess.run([cmake, "--build", str(build), "--config", "Debug", "--parallel", "2"],
                   check=True, capture_output=True, text=True, timeout=60)
    binary = build / "notification_contract"
    if not binary.exists():
        binary = build / "Debug/notification_contract.exe"
    subprocess.run([str(binary)], check=True, timeout=10)


def test_quiet_mode_preserves_capacity_receipt_and_final_failure_text():
    source = (Path(__file__).parents[1] / "native/src/ddo_vmm_torch_pool.cpp").read_text()
    receipt = source.index("last_capacity_report = r; capacity_report_recorded = true")
    quiet_guard = source.index("if (!show_capacity_report(r.termination == term_satisfied)) return;")
    details = source.index('std::fprintf(stderr, "DDO segmented capacity transaction:')
    assert receipt < quiet_guard < details
    assert "DDO allocation recovery: status=failed" in source
    assert 'if (show_recovery_warnings()) std::fprintf(stderr, "DDO segmented materialization refusal:' in source
