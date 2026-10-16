"""CPU-only packaging selection checks; never load native code or CUDA."""

import hashlib
import importlib.util
from pathlib import Path
import struct
import json

import pytest

pytest.importorskip("hatchling")


spec = importlib.util.spec_from_file_location("ddo_test_build_hook", Path(__file__).resolve().parents[1] / "hatch_build.py")
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


def pe_bytes(machine=0x8664, characteristics=0x2000):
    data = bytearray(128)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<H", data, 68, machine)
    struct.pack_into("<H", data, 86, characteristics)
    return bytes(data)


def test_accepts_hash_selected_amd64_dll(tmp_path):
    path = tmp_path / "test.dll"
    data = pe_bytes()
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    assert hook.validate_windows_artifact(path, digest.upper()) == digest


def elf_bytes(machine=62, elf_type=3, elf_class=2, endian=1):
    data = bytearray(64)
    data[:7] = bytes([0x7F, 69, 76, 70, elf_class, endian, 1])
    struct.pack_into("<HHI", data, 16, elf_type, machine, 1)
    return bytes(data)


def test_accepts_hash_selected_linux_elf(tmp_path):
    path = tmp_path / "test.so"
    path.write_bytes(elf_bytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert hook.validate_linux_artifact(path, digest.upper()) == digest


@pytest.mark.parametrize("data", [b"invalid", elf_bytes(machine=183), elf_bytes(elf_type=2),
                                 elf_bytes(elf_class=1), elf_bytes(endian=2)])
def test_rejects_wrong_elf_architecture_format_or_type(tmp_path, data):
    path = tmp_path / "test.so"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        hook.validate_linux_artifact(path, hashlib.sha256(data).hexdigest())


def test_linux_wheel_has_host_platform_tag_and_manifest(tmp_path, monkeypatch):
    path = tmp_path / "existing.so"
    path.write_bytes(elf_bytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setenv("DDO_VMM_WHEEL_LIBRARY", str(path))
    monkeypatch.setenv("DDO_VMM_WHEEL_SHA256", digest)
    monkeypatch.delenv("DDO_VMM_WHEEL_ABI_REPORT", raising=False)
    builder, data = build_hook(tmp_path), {}
    try:
        builder.initialize("standard", data)
        assert data["tag"] == "py3-none-linux_x86_64"
        assert data["pure_python"] is False
        sources = {destination: Path(source) for source, destination in data["force_include"].items()}
        manifest = json.loads(sources["ddo_vmm_allocator/native/linux/artifact.json"].read_text())
        assert manifest["sha256"] == digest and manifest["platform"] == "linux_x86_64"
        assert sources["ddo_vmm_allocator/native/linux/libddo_vmm.so"].read_bytes() == elf_bytes()
    finally:
        builder.finalize("standard", data, "unused.whl")


@pytest.mark.parametrize("data", [b"not a dll", pe_bytes(0x14C), pe_bytes(characteristics=0), b"MZ" + bytes(62)])
def test_rejects_non_dll_wrong_architecture_and_malformed_headers(tmp_path, data):
    path = tmp_path / "test.dll"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        hook.validate_windows_artifact(path, hashlib.sha256(data).hexdigest())


@pytest.mark.parametrize("expected", ["", "abc", "z" * 64, "0" * 64])
def test_refuses_missing_invalid_or_mismatched_hash(tmp_path, expected):
    path = tmp_path / "test.dll"
    path.write_bytes(pe_bytes())
    with pytest.raises(ValueError):
        hook.validate_windows_artifact(path, expected)


def build_hook(tmp_path, target="wheel"):
    return hook.CustomBuildHook(str(tmp_path), {}, None, None, str(tmp_path), target)


def test_plain_wheel_and_sdist_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("DDO_VMM_WHEEL_LIBRARY", raising=False)
    monkeypatch.delenv("DDO_VMM_WHEEL_SHA256", raising=False)
    data = {"pure_python": True}
    build_hook(tmp_path).initialize("standard", data)
    assert data == {"pure_python": True}
    monkeypatch.setenv("DDO_VMM_WHEEL_LIBRARY", "does-not-exist.dll")
    build_hook(tmp_path, "sdist").initialize("standard", data)
    assert data == {"pure_python": True}


@pytest.mark.parametrize("variable", ["DDO_VMM_WHEEL_LIBRARY", "DDO_VMM_WHEEL_SHA256"])
def test_native_assembly_requires_paired_selection(tmp_path, monkeypatch, variable):
    monkeypatch.delenv("DDO_VMM_WHEEL_LIBRARY", raising=False)
    monkeypatch.delenv("DDO_VMM_WHEEL_SHA256", raising=False)
    monkeypatch.setenv(variable, "selected")
    with pytest.raises(ValueError, match="requires both"):
        build_hook(tmp_path).initialize("standard", {})


def test_native_assembly_snapshots_binary_and_records_platform(tmp_path, monkeypatch):
    binary = tmp_path / "existing.dll"
    original = pe_bytes()
    binary.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    monkeypatch.setenv("DDO_VMM_WHEEL_LIBRARY", str(binary))
    monkeypatch.setenv("DDO_VMM_WHEEL_SHA256", digest)
    builder = build_hook(tmp_path)
    data = {}
    try:
        builder.initialize("standard", data)
        assert data["tag"] == "py3-none-win_amd64"
        assert data["pure_python"] is False
        sources = {destination: Path(source) for source, destination in data["force_include"].items()}
        snapshot = sources["ddo_vmm_allocator/native/win32/ddo_vmm.dll"]
        binary.write_bytes(b"changed after selection")
        assert snapshot.read_bytes() == original
        manifest = json.loads(sources["ddo_vmm_allocator/native/win32/artifact.json"].read_text())
        assert manifest["sha256"] == digest
        assert manifest["platform"] == "win_amd64"
    finally:
        builder.finalize("standard", data, "unused.whl")
    assert not snapshot.exists()
