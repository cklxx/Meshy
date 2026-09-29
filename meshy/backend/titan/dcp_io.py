"""DCP resume-save writer selection for low host memory.

Root cause of the ~10 GB persistent shared memory after the first DCP save
(2026-09-29, ``/dev/zero (deleted)`` MAP_SHARED regions, Pss_Shmem ~10 GB,
never reclaimed; it OOM-froze the host even with an 8 GB zram because pinned
pages are not swappable):

torchtitan's default :class:`FileSystemWriter` runs with ``thread_count=1``.
In DCP's ``_write_files_from_queue`` that single-thread branch selects
``_OverlappingCpuLoader``, which stages every GPU tensor to host with
``tensor.to("cpu", non_blocking=True)``. A non-blocking GPU->Host copy needs a
pinned destination, so the CUDA caching host allocator pins ~the full saved
state (model + both Adam moments) on the first save. Those pages are mmap'd
``/dev/zero`` MAP_SHARED and are retained by the allocator for reuse — they
are never returned, which is exactly why a second save does not grow them.

With ``thread_count > 1`` DCP instead uses ``_SerialCpuLoader``, which does a
plain blocking ``tensor.cpu()``: no pinning, the host pages are ordinary and
released after write. Measured on V100 (FSDP + Adam, fresh process):

==============================  ==================  =================
FileSystemWriter                /dev/zero resident  save wall time
==============================  ==================  =================
thread_count=1 (torchtitan def) +335 MB (retained)   0.92 s
thread_count=8 (this fix)       +0 MB                0.66 s
==============================  ==================  =================

The bump also shards the rank into N size-balanced files, which on the
write-through 3FS-FUSE target raises throughput (concurrently written, each
fsynced): ~97 MB/s vs ~68 MB/s for one file.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Callable

from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint import save as dcp_save

#: >1 makes DCP use the blocking _SerialCpuLoader (no pinned non-blocking
#: staging -> no persistent /dev/zero pool) and shards files for FUSE speed.
_DCP_WRITE_THREADS = 8


def make_unpinned_dcp_save(orig: Callable[..., Any]) -> Callable[..., Any]:
    """Bind onto ``CheckpointManager.dcp_save`` for the synchronous resume save.

    HF export (``to_hf=True``) and the async modes keep torchtitan's writer:
    only the plain synchronous DCP resume path is redirected.
    """

    def dcp_save_unpinned(
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: Any,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ) -> Any:
        if to_hf or str(getattr(async_mode, "value", async_mode)) != "disabled":
            return orig(
                state_dict,
                checkpoint_id=checkpoint_id,
                async_mode=async_mode,
                enable_garbage_collection=enable_garbage_collection,
                to_hf=to_hf,
            )

        writer = FileSystemWriter(
            path=checkpoint_id,
            single_file_per_rank=True,
            sync_files=True,
            # >1 -> blocking _SerialCpuLoader (no pinned non-blocking host
            # staging) and N balanced file shards.
            thread_count=_DCP_WRITE_THREADS,
        )
        return dcp_save(state_dict, storage_writer=writer, checkpoint_id=checkpoint_id)

    return dcp_save_unpinned


def prune_incomplete_step_dirs(folder: str) -> list[str]:
    """Remove ``step-N`` checkpoint dirs that never reached ``.metadata``.

    DCP writes ``.metadata`` last and renames it into place atomically, so a
    save killed mid-flight leaves data files without it. Pruning at startup
    guarantees resume never mistakes a torn save for a usable checkpoint.
    """
    removed: list[str] = []
    if not folder or not os.path.isdir(folder):
        return removed
    for name in os.listdir(folder):
        if not name.startswith("step-"):
            continue
        path = os.path.join(folder, name)
        if os.path.isdir(path) and not os.path.exists(os.path.join(path, ".metadata")):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    return removed
