"""Opt-in assembly of an existing, hash-selected Windows DLL; never compile it."""

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
        actual = validate_windows_artifact(path, expected)
        self._staging = tempfile.TemporaryDirectory(prefix="ddo-native-wheel-")
        stage = Path(self._staging.name)
        # Include a verified snapshot, not a mutable external build-directory file.
        binary = stage / "ddo_vmm.dll"
        binary.write_bytes(path.read_bytes())
        validate_windows_artifact(binary, actual)
        manifest = stage / "artifact.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "platform": "win_amd64",
            "library": "ddo_vmm.dll",
            "sha256": actual,
            "assembly": "existing binary; no compilation or CUDA execution",
        }, indent=2) + "\n", encoding="utf-8")
        build_data["pure_python"] = False
        build_data["tag"] = "py3-none-win_amd64"
        includes = build_data.setdefault("force_include", {})
        includes[str(binary)] = "ddo_vmm_allocator/native/win32/ddo_vmm.dll"
        includes[str(manifest)] = "ddo_vmm_allocator/native/win32/artifact.json"

    def finalize(self, version, build_data, artifact_path):
        stage = getattr(self, "_staging", None)
        if stage is not None:
            stage.cleanup()
