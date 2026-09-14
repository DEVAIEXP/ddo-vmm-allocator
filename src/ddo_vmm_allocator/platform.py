"""Platform and dependency checks kept separate from allocator policy."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import platform
import sys


@dataclass(frozen=True)
class CudaVmmCapabilities:
    """Facts required before a VMM backend may be constructed.

    Detection intentionally does not create a CUDA context or allocate memory.
    Runtime VMM probes will live in a separate module so importing this package
    remains safe in CPU-only processes and test discovery.
    """

    operating_system: str
    supported_operating_system: bool
    cuda_bindings_available: bool
    cuda_core_available: bool
    python_version: tuple[int, int]

    @property
    def ready_for_runtime_probe(self) -> bool:
        return self.supported_operating_system and self.cuda_bindings_available


def detect_cuda_vmm_capabilities() -> CudaVmmCapabilities:
    """Return import/platform capabilities without initializing CUDA."""

    operating_system = platform.system()
    return CudaVmmCapabilities(
        operating_system=operating_system,
        supported_operating_system=operating_system in {"Windows", "Linux"},
        cuda_bindings_available=importlib.util.find_spec("cuda.bindings") is not None,
        cuda_core_available=importlib.util.find_spec("cuda.core") is not None,
        python_version=(sys.version_info.major, sys.version_info.minor),
    )

