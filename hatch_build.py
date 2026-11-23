"""Opt-in assembly of an existing, hash-selected native library; never compile it."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def validate_windows_artifact(path: Path, expected_sha256: str) -> str:
    """Check immutable selection and PE architecture without loading the DLL."""
    expected = expected_sha256.lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("DDO_VMM_WHEEL_SHA256 must be an explicit 64-digit SHA256")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError("Selected native DLL SHA256 does not match; refusing to package")
    if len(data) < 64 or data[:2] != b"MZ":
        raise ValueError("Selected artifact is not a PE DLL")
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if offset + 24 > len(data) or data[offset:offset + 4] != b"PE\0\0":
        raise ValueError("Selected artifact has an invalid PE header")
    machine = struct.unpack_from("<H", data, offset + 4)[0]
    characteristics = struct.unpack_from("<H", data, offset + 22)[0]
    if machine != 0x8664 or not characteristics & 0x2000:
        raise ValueError("Selected artifact must be a Windows AMD64 DLL")
    return actual


def validate_linux_artifact(path: Path, expected_sha256: str) -> str:
    """Validate ELF64 little-endian x86-64 shared objects without loading them."""
    expected = expected_sha256.lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("DDO_VMM_WHEEL_SHA256 must be an explicit 64-digit SHA256")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError("Selected native ELF SHA256 does not match; refusing to package")
    if len(data) < 64 or data[:7] != b"\x7fELF\x02\x01\x01":
        raise ValueError("Selected artifact must be ELF64 little-endian")
    elf_type, machine, version = struct.unpack_from("<HHI", data, 16)
    if elf_type != 3 or machine != 62 or version != 1:
        raise ValueError("Selected artifact must be a Linux x86-64 shared object")
    return actual


TORCH_BRIDGE_FIELDS = ("abi", "torch_version", "torch_git", "python", "platform", "module", "sha256")


def validate_torch_bridge(module_path: Path, identity_path: Path) -> dict:
    """Check the optional Windows Torch bridge against its identity without loading it.

    The identity must name the module file and record its SHA256 and the exact
    PyTorch/CPython build it was compiled against; the installer uses the bridge
    only on that exact match at runtime.
    """
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    missing = [field for field in TORCH_BRIDGE_FIELDS if not identity.get(field)]
    if missing:
        raise ValueError(f"Torch bridge identity lacks {missing}")
    if identity["platform"] != "win32" or not str(identity["module"]).endswith("-win_amd64.pyd"):
        raise ValueError("The Torch bridge is assembled for Windows AMD64 only")
    if module_path.name != identity["module"]:
        raise ValueError("Torch bridge file name does not match its identity")
    data = module_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != str(identity["sha256"]).lower():
        raise ValueError("Torch bridge SHA256 does not match its identity; refusing to package")
    validate_windows_artifact(module_path, hashlib.sha256(data).hexdigest())
    return identity


class CustomBuildHook(BuildHookInterface):
    """Plain wheels stay pure; explicit native builds get a platform tag."""

    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        selected = os.environ.get("DDO_VMM_WHEEL_LIBRARY")
        expected = os.environ.get("DDO_VMM_WHEEL_SHA256")
        if not selected and not expected:
            return
        if not selected or not expected:
            raise ValueError("Native wheel assembly requires both DDO_VMM_WHEEL_LIBRARY and DDO_VMM_WHEEL_SHA256")
        path = Path(selected).expanduser().resolve(strict=True)
        if path.read_bytes()[:4] == b"\x7fELF":
            validate = validate_linux_artifact
            platform_tag, directory, filename = "linux_x86_64", "linux", "libddo_vmm.so"
        else:
            validate = validate_windows_artifact
            platform_tag, directory, filename = "win_amd64", "win32", "ddo_vmm.dll"
        actual = validate(path, expected)
        self._staging = tempfile.TemporaryDirectory(prefix="ddo-native-wheel-")
        stage = Path(self._staging.name)
        # Include a verified snapshot, not a mutable external build-directory file.
        binary = stage / filename
        binary.write_bytes(path.read_bytes())
        validate(binary, actual)
        abi = None
        abi_path = os.environ.get("DDO_VMM_WHEEL_ABI_REPORT")
        if abi_path:
            abi = json.loads(Path(abi_path).read_text(encoding="utf-8"))
            if abi.get("sha256") != actual or abi.get("platform") != platform_tag:
                raise ValueError("ABI report does not match the selected native artifact")
        manifest = stage / "artifact.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "platform": platform_tag,
            "library": filename,
            "sha256": actual,
            "abi": abi,
            "assembly": "existing binary; no compilation or CUDA execution",
        }, indent=2) + "\n", encoding="utf-8")
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{platform_tag}"
        includes = build_data.setdefault("force_include", {})
        includes[str(binary)] = f"ddo_vmm_allocator/native/{directory}/{filename}"
        includes[str(manifest)] = f"ddo_vmm_allocator/native/{directory}/artifact.json"
        # Optional checked Torch bridge: both paths or neither; never compiled here.
        bridge = os.environ.get("DDO_VMM_WHEEL_TORCH_BRIDGE")
        bridge_identity = os.environ.get("DDO_VMM_WHEEL_TORCH_BRIDGE_IDENTITY")
        if bridge or bridge_identity:
            if not bridge or not bridge_identity:
                raise ValueError("Torch bridge assembly requires both DDO_VMM_WHEEL_TORCH_BRIDGE and "
                                 "DDO_VMM_WHEEL_TORCH_BRIDGE_IDENTITY")
            if platform_tag != "win_amd64":
                raise ValueError("The Torch bridge is assembled for Windows AMD64 wheels only")
            bridge_path = Path(bridge).expanduser().resolve(strict=True)
            identity = validate_torch_bridge(bridge_path, Path(bridge_identity).expanduser().resolve(strict=True))
            bridge_copy = stage / identity["module"]
            bridge_copy.write_bytes(bridge_path.read_bytes())
            identity_copy = stage / "ddo_vmm_torch_bridge.json"
            identity_copy.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
            validate_torch_bridge(bridge_copy, identity_copy)
            includes[str(bridge_copy)] = f"ddo_vmm_allocator/native/{directory}/{identity['module']}"
            includes[str(identity_copy)] = f"ddo_vmm_allocator/native/{directory}/ddo_vmm_torch_bridge.json"

    def finalize(self, version, build_data, artifact_path):
        stage = getattr(self, "_staging", None)
        if stage is not None:
            stage.cleanup()
