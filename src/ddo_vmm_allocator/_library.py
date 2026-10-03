"""DLL loading diagnostics without importing Torch or calling CUDA exports."""

import ctypes
import platform


def load_library(path, error_type):
    loader = ctypes.WinDLL if platform.system() == "Windows" else ctypes.CDLL
    try:
        return loader(str(path))
    except OSError as error:
        raise error_type(
            f"Cannot load native VMM library '{path}': {error}. "
            "Check library architecture and required DLL dependencies."
        ) from error


def require_exports(library, path, names, error_type):
    missing = [name for name in names if getattr(library, name, None) is None]
    if missing:
        raise error_type(
            f"Incompatible native VMM library '{path}'; missing exports: "
            f"{', '.join(missing)}. Use a DLL built for this allocator package. "
            "The selected library will not be replaced automatically."
        )
