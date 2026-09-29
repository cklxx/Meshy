"""DCP writer must not pin the full saved state into persistent host memory.

Regression (T5g): torchtitan's default FileSystemWriter runs thread_count=1,
so DCP's _write_files_from_queue picks _OverlappingCpuLoader and stages every
GPU tensor with .to("cpu", non_blocking=True). The non-blocking copy pins the
destination; the CUDA caching host allocator then retains ~the full saved
state (model + Adam) as /dev/zero MAP_SHARED pages for the whole process —
~10 GB on Qwen3-0.6B, never freed, not swappable. thread_count>1 switches DCP
to the blocking _SerialCpuLoader (plain .cpu(), no pinning).
"""

from __future__ import annotations

import os

import pytest
import torch
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint import save as dcp_save

from meshy.backend.titan.dcp_io import (
    _DCP_WRITE_THREADS,
    make_unpinned_dcp_save,
    prune_incomplete_step_dirs,
)


def test_write_threads_above_overlapping_loader_threshold() -> None:
    # DCP selects _OverlappingCpuLoader (pinned non-blocking staging) ONLY at
    # thread_count==1; anything above uses the blocking _SerialCpuLoader.
    assert _DCP_WRITE_THREADS > 1


class _AsyncMode:
    def __init__(self, value: str) -> None:
        self.value = value


def test_sync_path_uses_multithread_writer(tmp_path) -> None:
    captured: dict = {}

    def fake_save(state_dict, *, storage_writer=None, checkpoint_id=None, **kw):
        captured["thread_count"] = storage_writer.thread_count
        captured["single_file"] = storage_writer.single_file_per_rank
        captured["sync_files"] = storage_writer.sync_files
        return dcp_save(
            state_dict, storage_writer=storage_writer, checkpoint_id=checkpoint_id
        )

    sd = {
        "model": {"w": torch.randn(128, 64), "b": torch.zeros(64)},
        "optimizer": {"m": torch.randn(128, 64)},
    }
    wrapped = make_unpinned_dcp_save(dcp_save)
    # Monkeypatch the module-level dcp_save the wrapper resolves to.
    import meshy.backend.titan.dcp_io as mod

    orig = mod.dcp_save
    mod.dcp_save = fake_save
    try:
        wrapped(sd, checkpoint_id=str(tmp_path / "s"),
                async_mode=_AsyncMode("disabled"), to_hf=False)
    finally:
        mod.dcp_save = orig

    assert captured["thread_count"] == _DCP_WRITE_THREADS > 1
    assert captured["sync_files"] is True


def test_hf_and_async_paths_delegate(tmp_path) -> None:
    calls = []

    def fake_orig(state_dict, checkpoint_id, async_mode,
                  enable_garbage_collection=False, to_hf=False):
        calls.append((str(getattr(async_mode, "value", async_mode)), to_hf))

    wrapped = make_unpinned_dcp_save(fake_orig)
    wrapped({}, checkpoint_id="x", async_mode=_AsyncMode("async"), to_hf=False)
    wrapped({}, checkpoint_id="x", async_mode=_AsyncMode("disabled"), to_hf=True)
    assert calls == [("async", False), ("disabled", True)]


def test_writer_round_trips_and_shards(tmp_path) -> None:
    sd = {f"t{i}": torch.randn(64, 64) for i in range(_DCP_WRITE_THREADS)}
    target = str(tmp_path / "step-2")
    wrapped = make_unpinned_dcp_save(dcp_save)
    wrapped(sd, checkpoint_id=target, async_mode=_AsyncMode("disabled"), to_hf=False)

    files = os.listdir(target)
    assert ".metadata" in files
    # thread_count=N -> N balanced data shards (not one whole-rank file).
    distcp = [f for f in files if f.endswith(".distcp")]
    assert len(distcp) == _DCP_WRITE_THREADS

    dst = {f"t{i}": torch.empty(64, 64) for i in range(_DCP_WRITE_THREADS)}
    dcp_load(dst, checkpoint_id=target)
    assert all(torch.equal(dst[f"t{i}"], sd[f"t{i}"]) for i in range(_DCP_WRITE_THREADS))


def test_prune_removes_only_metadata_less_dirs(tmp_path) -> None:
    bad = tmp_path / "step-10"
    bad.mkdir()
    (bad / "__0_0.distcp").write_bytes(b"x")
    good = tmp_path / "step-9"
    good.mkdir()
    (good / ".metadata").write_bytes(b"m")
    (good / "t.distcp").write_bytes(b"d")

    removed = prune_incomplete_step_dirs(str(tmp_path))
    assert removed == [str(bad)]
    assert not bad.exists() and good.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_no_pinned_devzero_residency_after_save(tmp_path) -> None:
    """GPU: a synchronous save must not leave pinned /dev/zero pages behind.

    Measures the process-wide MAP_SHARED /dev/zero (cudaHostAlloc-style)
    footprint before and after a save through the wrapper; the pinned
    overlapping loader added ~335 MB for the small repro model, the serial
    loader must add ~0.
    """
    import gc

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29601")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    import torch.distributed as dist

    dist.init_process_group("nccl")
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    try:
        m = torch.nn.Sequential(
            torch.nn.Linear(2048, 4096), torch.nn.ReLU(), torch.nn.Linear(4096, 2048)
        ).cuda()
        m = FSDP(m)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(3):
            opt.zero_grad()
            m(torch.randn(8, 2048, device="cuda")).sum().backward()
            opt.step()
        torch.cuda.synchronize()
        gc.collect()

        def zero_mb() -> float:
            tot = 0
            for line in open("/proc/self/maps"):
                if "/dev/zero" in line:
                    a, b = line.split()[0].split("-")
                    tot += int(b, 16) - int(a, 16)
            return tot / 1e6

        before = zero_mb()
        wrapped = make_unpinned_dcp_save(dcp_save)
        wrapped(
            {"model": m, "optimizer": opt},
            checkpoint_id=str(tmp_path / "step-1"),
            async_mode=_AsyncMode("disabled"),
            to_hf=False,
        )
        torch.cuda.synchronize()
        gc.collect()
        after = zero_mb()
        assert after - before < 50, f"pinned /dev/zero grew {after - before:.0f} MB"
    finally:
        dist.destroy_process_group()
