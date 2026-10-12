"""Mocked ABI contracts; no native library loading or CUDA."""

from types import SimpleNamespace

import pytest
from ddo_vmm_allocator.native import NativeVmmError, NativeVmmLibrary


@pytest.mark.parametrize("status", [None, 0, 1, 2])
def test_trim_optional_export_and_result_contract(status):
    owner = object.__new__(NativeVmmLibrary)
    calls = []

    def export():
        calls.append("trim")
        return status

    owner._library = SimpleNamespace(**({"ddo_vmm_torch_trim_idle_v1": export} if status is not None else {}))
    owner.torch_pool_stats = lambda: {"pool_mapped_bytes": 100}
    if status == 1:
        with pytest.raises(NativeVmmError, match="unsafe backing retained"):
            owner.trim_idle_torch_pool()
    else:
        result = owner.trim_idle_torch_pool()
        assert result["trim_executed"] == (status == 0)
        assert result["status"] == ("completed" if status == 0 else "unsupported")
    assert calls == ([] if status is None else ["trim"])
