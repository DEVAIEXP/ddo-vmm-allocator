"""Explicit host-registration capacity gate, separate from routine micro tests."""

import os
from pathlib import Path

import pytest

from ddo_vmm_allocator.native import NativeVmmError, load_native_vmm


pytestmark = [
    pytest.mark.vmm,
    pytest.mark.skipif(
        os.environ.get("DDO_VMM_CAPACITY_GATE") != "1",
        reason="requires explicit approval for host-capacity gates",
    ),
]
MIB = 1024**2


def test_persistent_source_directory_capacity() -> None:
    """Explicit existing-shard gate; no model loading or synthetic large file."""
    _source_directory_capacity(full_registration=False)


@pytest.mark.skipif(
    os.environ.get("DDO_VMM_FULL_REGISTRATION_GATE") != "1",
    reason="requires separate approval for full-source CUDA registration",
)
def test_persistent_source_directory_full_registration() -> None:
    """Retain registrations across passes, with unchanged GPU destination size."""
    _source_directory_capacity(full_registration=True)


def _source_directory_capacity(*, full_registration: bool) -> None:
    import torch
    from ddo_vmm_allocator.tensor_view import cuda_uint8_tensor_from_address

    directory = os.environ.get("DDO_VMM_CAPACITY_SOURCE_DIR")
    if not directory:
        pytest.skip("requires explicitly selected immutable source directory")
    if os.name != "nt" or not torch.cuda.is_available():
        pytest.skip("Windows CUDA required")
    paths = sorted(Path(directory).glob("*.safetensors"))
    assert paths, "no source shards selected"
    sizes = [path.stat().st_size for path in paths]
    segment = 64 * MIB
    slots = sum((size + segment - 1) // segment for size in sizes)
    assert slots * segment <= 32 * 2**30
    registered_limit = slots * segment if full_registration else 512 * MIB
    if full_registration:
        approved_limit = os.environ.get("DDO_VMM_CAPACITY_REGISTERED_LIMIT_BYTES")
        if not approved_limit:
            pytest.skip("requires an explicit approved full-registration ceiling")
        assert registered_limit == int(approved_limit), "selected shards differ from approved capacity"
    torch.empty(1, device="cuda:0")
    library = load_native_vmm(Path(os.environ["DDO_VMM_LIBRARY_PATH"]))
    with library.create_source_cache(
        slot_bytes=segment, slots=slots, persistent_buffer=True,
        registered_limit_bytes=registered_limit,
        process_limit_bytes=slots * segment + 1536 * MIB,
        host_headroom_bytes=8 * 2**30,
    ) as cache, library.create_host_ring(segment, 24), library.create_model_arena(
        2 * MIB, page_size_bytes=2 * MIB,
    ) as arena:
        arena.prepare_view_range(0, 17)
        view, owner = cuda_uint8_tensor_from_address(arena.info.address, 17, owner=arena, backing="device")
        arena.release_view_range(0, 17)
        arena.fault_range(0, 17)
        try:
            upload = torch.cuda.Stream()
            for pass_index in range(2):
                for path, size in zip(paths, sizes):
                    with library.open_source_file(path) as source, path.open("rb") as expected:
                        for offset in range(0, size, segment):
                            count = min(17, size - offset)
                            with cache.request(source, offset, count) as ticket:
                                assert ticket is not None, "source capacity/headroom admission refused"
                                ticket.copy_to_arena_async(arena, 0, upload)
                                ticket.wait_on_stream(torch.cuda.current_stream())
                                expected.seek(offset)
                                assert bytes(view[:count].cpu().tolist()) == expected.read(count)
                stats = cache.snapshot()
                assert stats["source_read_bytes"] == sum(sizes)
                assert stats["committed_bytes"] == slots * segment
                assert stats["peak_registered_bytes"] <= registered_limit
                if full_registration:
                    assert stats["registered_bytes"] == slots * segment
                    assert stats["registrations"] == slots
                    assert stats["unregistrations"] == stats["registration_waits"] == 0
                    assert stats["dma_bytes"] == (pass_index + 1) * sum(
                        min(17, size - offset)
                        for size in sizes for offset in range(0, size, segment)
                    )
                assert stats["ring_fallback_bytes"] == stats["host_copy_bytes"] == 0
                assert stats["live_tickets"] == 0
                print(f"source capacity pass={pass_index + 1}: {stats}", flush=True)
            cache.trim(0)
            trimmed = cache.snapshot()
            assert trimmed["committed_bytes"] == slots * segment
            assert trimmed["registered_bytes"] == 0
            if full_registration:
                assert trimmed["unregistrations"] == slots
            cache.clear()
            assert cache.snapshot()["committed_bytes"] == 0
        finally:
            arena.unpin_range(0, 17)
            del view, owner
    with library.create_source_cache(slot_bytes=2 * MIB, slots=1) as probe:
        stats = probe.snapshot()
        assert stats["process_committed_bytes"] == stats["process_registered_bytes"] == 0
        assert stats["process_staging_registered_bytes"] == 0


@pytest.mark.parametrize("persistent", [False, True], ids=["legacy-v4", "persistent-v5"])
def test_model_windows_1536_mib_ring_plus_512_mib_cache(tmp_path: Path, persistent: bool) -> None:
    """Explicit 2 GiB host ceiling; no model or GPU residency-budget change."""
    import torch

    if os.name != "nt" or not torch.cuda.is_available():
        pytest.skip("Windows CUDA required")
    torch.empty(1, device="cuda:0")
    library = load_native_vmm(Path(os.environ.get(
        "DDO_VMM_LIBRARY_PATH", Path(__file__).parents[1] / "build/Release/ddo_vmm.dll",
    )))
    from ddo_vmm_allocator.tensor_view import cuda_uint8_tensor_from_address

    path = tmp_path / "model-windows.bin"
    with path.open("wb") as output:
        chunk = b"w" * MIB
        for _ in range(512):
            output.write(chunk)
    with library.open_source_file(path) as source, library.create_source_cache(
        slot_bytes=64 * MIB, slots=8, process_limit_bytes=2048 * MIB,
        host_headroom_bytes=4 * 2**30, model_windows=True,
        persistent_buffer=persistent,
        registered_limit_bytes=128 * MIB if persistent else 0,
    ) as cache:
        assert cache.snapshot()["registered_bytes"] == 0
        with library.create_host_ring(64 * MIB, 24):
            for _ in range(2):
                for offset in range(0, 512 * MIB, 64 * MIB):
                    with cache.request(source, offset + 13, 17) as ticket:
                        ticket.wait()
            stats = cache.snapshot()
            assert stats["process_staging_registered_bytes"] == 1536 * MIB
            assert stats["registered_bytes"] == (0 if persistent else 512 * MIB)
            assert stats["retained_bytes"] == stats["committed_bytes"] == 512 * MIB
            assert stats["source_read_bytes"] == 512 * MIB
            assert stats["hits"] == stats["misses"] == 8
            assert stats["live_tickets"] == 0
            if persistent:
                with library.create_model_arena(2 * MIB, page_size_bytes=2 * MIB) as arena:
                    arena.prepare_view_range(0, 17)
                    view, owner = cuda_uint8_tensor_from_address(
                        arena.info.address, 17, owner=arena, backing="device",
                    )
                    arena.release_view_range(0, 17)
                    arena.fault_range(0, 17)
                    try:
                        upload = torch.cuda.Stream()
                        for _ in range(2):
                            for offset in range(0, 512 * MIB, 64 * MIB):
                                with cache.request(source, offset + 13, 17) as ticket:
                                    ticket.copy_to_arena_async(arena, 0, upload)
                                    ticket.wait_on_stream(torch.cuda.current_stream())
                                    assert view.cpu().tolist() == [ord("w")] * 17
                        uploaded = cache.snapshot()
                        assert uploaded["source_read_bytes"] == 512 * MIB
                        assert uploaded["registered_bytes"] <= 128 * MIB
                        assert uploaded["peak_registered_bytes"] <= 128 * MIB
                        assert uploaded["registered_limit_bytes"] == 128 * MIB
                        # Eight distinct segments on each pass, capacity for
                        # two: the second pass may reuse up to two registrations.
                        # DMA count is not registration count.
                        assert 14 <= uploaded["registrations"] <= 16
                        assert uploaded["unregistrations"] == (
                            uploaded["registrations"] - uploaded["registered_bytes"] // (64 * MIB)
                        )
                        assert uploaded["dma_bytes"] == 16 * 17
                        assert uploaded["misses"] == 8 and uploaded["hits"] == 24
                        assert uploaded["ring_fallback_bytes"] == uploaded["host_copy_bytes"] == 0
                        cache.trim(0)
                        trimmed = cache.snapshot()
                        assert trimmed["registered_bytes"] == 0
                        assert trimmed["unregistrations"] == trimmed["registrations"]
                        assert trimmed["committed_bytes"] == trimmed["retained_bytes"] == 512 * MIB
                    finally:
                        arena.unpin_range(0, 17)
                        del view, owner
            with pytest.raises(NativeVmmError, match="active source-cache process limit"):
                library.create_host_ring(64 * MIB, 1)
            cache.clear()
            assert cache.snapshot()["registered_bytes"] == cache.snapshot()["committed_bytes"] == 0
        assert cache.snapshot()["process_staging_registered_bytes"] == 0
    with library.create_source_cache(slot_bytes=2 * MIB, slots=1) as probe:
        stats = probe.snapshot()
        assert stats["process_registered_bytes"] == stats["process_staging_registered_bytes"] == 0
        assert stats["process_committed_bytes"] == 0


def test_640_mib_ring_plus_128_mib_cache_releases_all_registrations(tmp_path: Path) -> None:
    import torch

    if os.name != "nt" or not torch.cuda.is_available():
        pytest.skip("Windows CUDA required")
    torch.empty(1, device="cuda:0")
    library_path = Path(os.environ.get(
        "DDO_VMM_LIBRARY_PATH", Path(__file__).parents[1] / "build/Release/ddo_vmm.dll"
    ))
    library = load_native_vmm(library_path)
    path = tmp_path / "source.bin"
    with path.open("wb") as output:
        chunk = b"c" * MIB
        for _ in range(128):
            output.write(chunk)

    with library.open_source_file(path) as source:
        with library.create_source_cache(
            slot_bytes=64 * MIB, slots=2, process_limit_bytes=768 * MIB
        ) as cache:
            assert cache.snapshot()["registered_bytes"] == 0
            with library.create_host_ring(64 * MIB, 10):
                assert cache.snapshot()["process_staging_registered_bytes"] == 640 * MIB
                print("host registration: ring=640 MiB, cache=0 MiB", flush=True)
                for offset in (0, 64 * MIB):
                    with cache.request(source, offset, 64 * MIB) as ticket:
                        ticket.wait()
                stats = cache.snapshot()
                assert stats["registered_bytes"] == stats["retained_bytes"] == 128 * MIB
                assert stats["process_registered_bytes"] == 128 * MIB
                assert stats["process_limit_bytes"] == 768 * MIB
                assert stats["source_read_bytes"] == 128 * MIB
                assert stats["live_tickets"] == 0
                print("host registration: ring=640 MiB, cache=128 MiB, total=768 MiB", flush=True)
                with pytest.raises(NativeVmmError, match="active source-cache process limit"):
                    library.create_host_ring(64 * MIB, 1)
                assert cache.snapshot()["process_staging_registered_bytes"] == 640 * MIB
                cache.clear()
                assert cache.snapshot()["process_registered_bytes"] == 0
            assert cache.snapshot()["process_staging_registered_bytes"] == 0
    # A fresh micro owner observes the process ledger after all prior owners
    # closed and also verifies that the strictest live cap was refunded.
    with library.create_source_cache(slot_bytes=2 * MIB, slots=1) as probe:
        stats = probe.snapshot()
        assert stats["process_registered_bytes"] == stats["process_staging_registered_bytes"] == 0
        assert stats["process_limit_bytes"] == 128 * MIB
    print("host registration after close: ring=0 MiB, cache=0 MiB", flush=True)
