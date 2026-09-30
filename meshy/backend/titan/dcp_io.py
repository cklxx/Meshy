"""DCP resume-save writer: bounded host peak, no pinned staging, fail-loud.

Two independent host-memory problems in a synchronous DCP resume save:

1. Pinned *persistent* pool (fixed by avoiding the overlapping loader): DCP's
   ``_OverlappingCpuLoader`` (selected only at ``thread_count == 1``) copies
   GPU tensors to host with ``non_blocking=True``, forcing the CUDA caching
   host allocator to pin the full saved state; those /dev/zero pages are never
   returned. This produced the ~10 GB permanent Pss_Shmem (see T5g).

2. Transient per-file accumulation (T5f): upstream's
   ``_write_files_from_queue`` collects every host tensor it writes into a
   local ``tensor_dict`` for the *whole bucket* before the file is closed.
   That dict is only consumed by the safetensors branch; in the default
   TORCH_SAVE format it is pure retention, so host RSS at the save window
   approaches the full checkpoint (the ~515 MB MemAvailable floor).

:class:`BoundedFileSystemWriter` overrides ``_write_data`` with a queue worker
that, for TORCH_SAVE, writes each tensor and drops the reference immediately
(no accumulating dict) and always uses ``_SerialCpuLoader`` (blocking
``.cpu()``, never pinned) regardless of thread count. Measured on V100,
FSDP + Adam, 1.6 GB saved state:

=============================  ===============  ===========  ===========
writer                          save-window RSS  post-save    pinned /dev/zero
                                increment        RSS
=============================  ===============  ===========  ===========
upstream default (tc=1)         +2.51 GB         stays 3.6 GB  ~full state
tc=8 (no accumulation)          +1.50 GB         back to base 0
tc=2 (default here)             +1.28 GB         back to base 0
tc=1 (minimum window)           +1.02 GB         back to base 0
=============================  ===============  ===========  ===========

The window increment scales with the number of concurrently materialized
tensors (~write workers), not the checkpoint total; ``XRL_DCP_WRITE_THREADS``
trades a little host headroom for concurrent writeback on slow FUSE targets.

Fail-loud: this relies on private names in torch's DCP filesystem module. If a
future torch removes/renames any of them, import falls back to the stock
:class:`FileSystemWriter` (the memory bound is lost) and logs a clear warning
once at import — training still starts rather than crashing on an unrelated
import, and the save path itself is unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
from io import UnsupportedOperation
from typing import Any, Callable

import torch
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint import save as dcp_save
from torch.futures import Future

logger = logging.getLogger(__name__)

try:
    from torch.distributed.checkpoint.filesystem import (
        CUSTOM_METADATA_KEY,
        DCP_VERSION_KEY,
        FORMAT_KEY,
        FORMAT_VALUE,
        HF_DCP_VERSION,
        SerializationFormat,
        WriteItemType,
        _SerialCpuLoader,
        _StorageWriterTransforms,
        _TensorLoader,
        _item_size,
        _write_item,
    )

    _BOUNDED_AVAILABLE = True
    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - depends on installed torch
    _BOUNDED_AVAILABLE = False
    _IMPORT_ERROR = exc
    logger.warning(
        "DCP bounded-memory checkpoint writer unavailable: this torch build "
        "is missing a private DCP filesystem symbol (%s). Falling back to the "
        "stock FileSystemWriter; DCP resume saves may use more host memory. "
        "Pin/upgrade torch to a version providing these symbols.",
        exc,
    )

#: File shards / concurrent write workers for the resume save. Each worker
#: materializes one host tensor at a time (blocking .cpu(), no pinning), so the
#: live save-window footprint scales with this count. Measured FSDP+Adam on
#: V100 (1.6 GB saved state): tc=1 +1.0 GB, tc=2 +1.28 GB, tc=4 +1.54 GB,
#: tc=8 +1.50 GB; on the write-through 3FS-FUSE target more shards overlap
#: slow writeback (~97 vs ~68 MB/s). 2 is the default; override with
#: XRL_DCP_WRITE_THREADS.
_DCP_WRITE_THREADS = int(os.environ.get("XRL_DCP_WRITE_THREADS", "2"))


class _WorkerError(Exception):
    """A DCP file-writer thread failed; carries the underlying exception."""


def _bounded_write_files_from_queue(
    create_stream: Callable,
    file_queue: "queue.Queue",
    result_queue: "queue.Queue",
    error_queue: "queue.Queue",
    planner: Any,
    transforms: "_StorageWriterTransforms",
    use_fsync: bool,
    serialization_format: "SerializationFormat",
) -> None:
    """Write queued files one tensor at a time, never accumulating a bucket.

    Always uses ``_SerialCpuLoader`` (blocking ``.cpu()`` -> no pinning), so it
    is safe at any thread count. For TORCH_SAVE each materialized tensor is
    written and then dropped, so a per-file dict never holds the whole bucket;
    safetensors must serialize the whole set at once, so that branch retains.

    Any exception other than the normal ``queue.Empty`` drain signal is pushed
    onto ``error_queue`` so the main thread can re-raise it (upstream silently
    swallows worker errors, which can leave a .metadata over missing tensors).
    """
    try:
        while True:
            file_name, storage_key, write_items = file_queue.get_nowait()
            loader: "_TensorLoader" = _SerialCpuLoader(planner.resolve_data)

            tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
            for write_item in tensor_w:
                loader.add(_item_size(write_item), write_item)
            loader.start_loading()

            bytes_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
            write_results = []

            # create_stream is a @contextmanager; use ``with`` to obtain the
            # actual file object (and let it close itself).
            with create_stream(file_name, "wb") as stream:
                for write_item in bytes_w:
                    data = planner.resolve_data(write_item)
                    write_results.append(
                        _write_item(
                            transforms, stream, data, write_item, storage_key,
                            serialization_format,
                        )
                    )

                if serialization_format == SerializationFormat.TORCH_SAVE:
                    # Stream one tensor through and drop the reference as soon
                    # as it is serialized — the bounded-memory path.
                    for tensor, write_item in loader.values():
                        write_results.append(
                            _write_item(
                                transforms, stream, tensor, write_item, storage_key,
                                serialization_format,
                            )
                        )
                else:
                    # safetensors must serialize the whole set in one call.
                    tensor_dict, metadata_dict = {}, {}
                    for tensor, write_item in loader.values():
                        write_results.append(
                            _write_item(
                                transforms, stream, tensor, write_item, storage_key,
                                serialization_format,
                            )
                        )
                        tensor_dict[write_item.index.fqn] = tensor
                        metadata_dict[write_item.index.fqn] = {
                            "saved_offsets": write_item.tensor_data.chunk.offsets
                        }
                    from safetensors.torch import save as st_save

                    stream.write(
                        st_save(
                            tensor_dict,
                            metadata={
                                CUSTOM_METADATA_KEY: json.dumps(metadata_dict),
                                DCP_VERSION_KEY: str(HF_DCP_VERSION),
                                FORMAT_KEY: FORMAT_VALUE,
                            },
                        )
                    )

                if use_fsync:
                    try:
                        os.fsync(stream.fileno())
                    except (AttributeError, UnsupportedOperation):
                        os.sync()
            result_queue.put(write_results)
    except queue.Empty:
        return
    except Exception as exc:  # surface in the main thread; do NOT swallow
        error_queue.put(exc)


class BoundedFileSystemWriter(FileSystemWriter):
    """FileSystemWriter with a bounded per-save host footprint and fail-loud
    writes: a worker error or a short result count aborts the save before
    ``.metadata`` is written, so a checkpoint can never look complete while
    missing tensors."""

class BoundedFileSystemWriter(FileSystemWriter):
    """FileSystemWriter with a bounded per-save host footprint and fail-loud
    writes: a worker error or a short result count aborts the save before
    ``.metadata`` is written, so a checkpoint can never look complete while
    missing tensors."""

    def write_data(self, plan, planner):  # type: ignore[override]
        # Rebuild upstream's file queue but tag it with the total write-item
        # count before workers drain it, so _write_data can verify every item
        # produced exactly one WriteResult.
        from torch.distributed.checkpoint.filesystem import (
            DEFAULT_SUFFIX,
            _split_by_size_and_type,
        )

        storage_plan = plan.storage_data
        counter = {"n": 0}

        def gen_file() -> str:
            name = f"{storage_plan.prefix}{counter['n']}{DEFAULT_SUFFIX}"
            counter["n"] += 1
            return name

        file_queue: "queue.Queue" = queue.Queue()
        total_items = 0
        if self.single_file_per_rank:
            for bucket in _split_by_size_and_type(self.thread_count, plan.items):
                file_name = gen_file()
                path = self.fs.concat_path(self.path, file_name)
                file_queue.put((path, file_name, bucket))
                total_items += len(bucket)
        else:
            for item in plan.items:
                file_name = gen_file()
                path = self.fs.concat_path(self.path, file_name)
                file_queue.put((path, file_name, [item]))
                total_items += 1
        file_queue._bounded_total_items = total_items  # type: ignore[attr-defined]
        return self._write_data(planner, file_queue)

    def _write_data(self, planner, file_queue) -> Future:  # type: ignore[override]
        result_queue: "queue.Queue" = queue.Queue()
        error_queue: "queue.Queue" = queue.Queue()
        expected = getattr(file_queue, "_bounded_total_items", -1)

        args = (
            self.fs.create_stream,
            file_queue,
            result_queue,
            error_queue,
            planner,
            self.transforms,
            self.sync_files,
            self.serialization_format,
        )
        threads = [
            threading.Thread(target=_bounded_write_files_from_queue, args=args)
            for _ in range(1, self.thread_count)
        ]
        for t in threads:
            t.start()
        _bounded_write_files_from_queue(*args)

        for t in threads:
            t.join()

        if not error_queue.empty():
            cause = error_queue.get()
            raise _WorkerError(
                f"DCP file-writer thread failed: {type(cause).__name__}: {cause}"
            ) from cause

        res: list = []
        while not result_queue.empty():
            res += result_queue.get_nowait()
        if len(res) != expected:
            raise RuntimeError(
                f"DCP save incomplete: {len(res)} write results for "
                f"{expected} write items; refusing to write .metadata"
            )

        fut = Future()
        fut.set_result(res)
        return fut


def make_bounded_dcp_save(orig: Callable[..., Any]) -> Callable[..., Any]:
    """Bind onto ``CheckpointManager.dcp_save`` for the synchronous resume save.

    HF export (``to_hf=True``) and async modes keep torchtitan's writer. If the
    bounded writer is unavailable in this torch build, fall back to the stock
    FileSystemWriter.
    """

    def dcp_save_bounded(
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

        if not _BOUNDED_AVAILABLE:
            # Stock writer; still synchronous and correct, just not bounded.
            return orig(
                state_dict,
                checkpoint_id=checkpoint_id,
                async_mode=async_mode,
                enable_garbage_collection=enable_garbage_collection,
                to_hf=False,
            )

        writer = BoundedFileSystemWriter(
            path=checkpoint_id,
            single_file_per_rank=True,
            sync_files=True,
            thread_count=_DCP_WRITE_THREADS,
        )
        return dcp_save(state_dict, storage_writer=writer, checkpoint_id=checkpoint_id)

    return dcp_save_bounded


def prune_incomplete_step_dirs(folder: str) -> list[str]:
    """Remove ``step-N`` checkpoint dirs that never reached ``.metadata``."""
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
