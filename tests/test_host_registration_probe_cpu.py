"""CPU-only tests of bounded diagnostic search; no CUDA imports."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "host_registration_probe", Path(__file__).parents[1] / "scripts" / "host_registration_probe.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_search_finds_boundary_and_repeats():
    calls = []

    def attempt(slots):
        calls.append(slots)
        return {"status": "PASS" if slots <= 17 else "OOM"}

    capacity, _ = probe.search_capacity(attempt, 1536, 64)
    assert capacity == 1088
    assert calls[-1] == 17
    assert max(calls) <= 24
    assert len(calls) <= 8


def test_ceiling_pass_is_not_exceeded():
    capacity, records = probe.search_capacity(lambda slots: {"status": "PASS"}, 1024, 64)
    assert capacity == 1024
    assert len(records) == 3


def test_first_oom_stops():
    capacity, records = probe.search_capacity(lambda slots: {"status": "OOM"}, 1024, 64)
    assert capacity == 0
    assert len(records) == 1


@pytest.mark.parametrize("status", ["TIMEOUT", "ERROR"])
def test_unexpected_failure_stops(status):
    with pytest.raises(RuntimeError, match="Unexpected probe failure"):
        probe.search_capacity(lambda slots: {"status": status}, 1024, 64)


def test_unstable_repeat_refuses_recommendation():
    responses = iter(["PASS", "PASS", "OOM"])
    with pytest.raises(RuntimeError, match="did not repeat"):
        probe.search_capacity(lambda slots: {"status": next(responses)}, 1024, 64)
