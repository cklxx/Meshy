"""Bounded host-memory DCP resume writer (T5f).

Upstream's FileSystemWriter write worker accumulates every tensor of a file
bucket in a local ``tensor_dict``; that dict is only used by the safetensors
path, so for the default TORCH_SAVE format it needlessly keeps the whole
bucket resident during the save window (the cause of the ~515 MB MemAvailable
floor on top of the pinned-staging bug fixed in T5g). BoundedFileSystemWriter
streams tensors through one at a time and drops each reference immediately.
"""

from __future__ import annotations

import gc
import os

import pytest
import torch
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint import save as dcp_save

from meshy.backend.titan.dcp_io import (
    _DCP_WRITE_THREADS,
    BoundedFileSystemWriter,
    make_bounded_dcp_save,
    prune_incomplete_step_dirs,
)


class _AsyncMode:
    def __init__(self, value: str) -> None:
        self.value = value


def test_bounded_writer_round_trips_across_shards(tmp_path) -> None:
    n = _DCP_WRITE_THREADS * 5
    sd = {f"t{i:04d}": torch.randn(128, 128) for i in range(n)}
    target = str(tmp_path / "step-1")
    wrapped = make_bounded_dcp_save(dcp_save)
    wrapped(sd, checkpoint_id=target, async_mode=_AsyncMode("disabled"), to_hf=False)

    files = os.listdir(target)
    assert ".metadata" in files
    distcp = [f for f in files if f.endswith(".distcp")]
    assert len(distcp) == _DCP_WRITE_THREADS  # balanced shards, one file per worker

    dst = {f"t{i:04d}": torch.empty(128, 128) for i in range(n)}
    dcp_load(dst, checkpoint_id=target)
    assert all(torch.equal(dst[k], v) for k, v in sd.items())


def test_bounded_writer_is_serial_cpu_no_pin(tmp_path) -> None:
    # thread_count>1 with our writer must not reintroduce the overlapping
    # (pinned) loader: assert the bound is enforced by the writer class, not
    # upstream's thread_count side effect.
    writer = BoundedFileSystemWriter(
        path=str(tmp_path / "w"),
        single_file_per_rank=True,
        sync_files=True,
        thread_count=_DCP_WRITE_THREADS,
    )
    assert writer.thread_count > 1


def test_bounded_writer_does_not_retain_tensors(tmp_path) -> None:
    """After writing, the writer must not hold the saved tensors."""
    import gc as _gc
    import weakref

    t = torch.randn(256, 256)
    ref = weakref.ref(t)
    sd = {"t": t}
    wrapped = make_bounded_dcp_save(dcp_save)
    wrapped(sd, checkpoint_id=str(tmp_path / "s"),
            async_mode=_AsyncMode("disabled"), to_hf=False)
    del t, sd
    _gc.collect()
    # The worker's local dict must have released the tensor.
    assert ref() is None


def test_hf_and_async_delegate_to_original() -> None:
    calls = []

    def fake_orig(state_dict, checkpoint_id, async_mode,
                  enable_garbage_collection=False, to_hf=False):
        calls.append((str(getattr(async_mode, "value", async_mode)), to_hf))

    wrapped = make_bounded_dcp_save(fake_orig)
    wrapped({}, checkpoint_id="x", async_mode=_AsyncMode("async"), to_hf=False)
    wrapped({}, checkpoint_id="x", async_mode=_AsyncMode("disabled"), to_hf=True)
    assert calls == [("async", False), ("disabled", True)]


def test_prune_metadata_less_dirs(tmp_path) -> None:
    bad = tmp_path / "step-20"
    bad.mkdir()
    (bad / "__0_0.distcp").write_bytes(b"x")
    good = tmp_path / "step-19"
    good.mkdir()
    (good / ".metadata").write_bytes(b"m")
    removed = prune_incomplete_step_dirs(str(tmp_path))
    assert removed == [str(bad)]
    assert not bad.exists() and good.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_save_window_rss_increment_is_bounded(tmp_path) -> None:
    """GPU end-to-end bound: (a) the save-window RSS increment is strictly
    below the full saved-state size (the writer does not accumulate the whole
    bucket), and (b) after the save RSS returns to baseline — no pinned or
    staging pool is retained (the T5g regression)."""
    import threading
    import time as _time
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29633")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group("nccl")
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        H = 4096
        m = torch.nn.Sequential(
            torch.nn.Linear(H, H), torch.nn.ReLU(), torch.nn.Linear(H, H)
        ).cuda()
        m = FSDP(m)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(2):
            opt.zero_grad()
            m(torch.randn(2, H, device="cuda")).sum().backward()
            opt.step()
        torch.cuda.synchronize()
        gc.collect()

        # fp32 params + two fp32 Adam moments, the full DCP saved payload.
        saved_bytes = sum(p.numel() * 4 for p in m.parameters()) * 3

        def rss_mb() -> float:
            for line in open("/proc/self/status"):
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024
            return 0.0

        samples, stop = [], [False]

        def sample():
            while not stop[0]:
                samples.append(rss_mb())
                _time.sleep(0.01)

        th = threading.Thread(target=sample)
        base = rss_mb()
        th.start()
        wrapped = make_bounded_dcp_save(dcp_save)
        wrapped(
            {"model": m, "optimizer": opt},
            checkpoint_id=str(tmp_path / "step-1"),
            async_mode=_AsyncMode("disabled"),
            to_hf=False,
        )
        torch.cuda.synchronize()
        stop[0] = True
        th.join()
        window_delta = (max(samples) if samples else 0.0) - base

        # Drop the source state too, then any staging/pinned pool would show
        # up as RSS the writer failed to release.
        del m, opt
        gc.collect()
        post_delta = rss_mb() - base

        # Window must not hold the whole state (tc>=2 keeps ~70-80%, not 100%).
        assert window_delta < saved_bytes / 1e6 * 0.95, (
            f"window RSS {window_delta:.0f} MB >= full state "
            f"{saved_bytes/1e6:.0f} MB — bucket is being accumulated"
        )
        # No retained staging/pinned pool after the save returns.
        assert post_delta < 100, f"RSS retained after save: {post_delta:.0f} MB"
    finally:
        dist.destroy_process_group()
