"""DCP resume-save writer: bounded host peak, no pinned staging.

Two independent host-memory problems in a synchronous DCP resume save:

1. Pinned *persistent* pool (fixed by ``thread_count > 1``): DCP's
   ``_OverlappingCpuLoader`` (selected only at ``thread_count == 1``) copies
   GPU tensors to host with ``non_blocking=True``, forcing the CUDA caching
   host allocator to pin the full saved state; those /dev/zero pages are never
   returned. This produced the ~10 GB permanent Pss_Shmem (see T5g).

2. Transient per-file accumulation (this module, T5f): upstream's
   ``_write_files_from_queue`` collects every host tensor it writes into a
   local ``tensor_dict`` for the *whole bucket* before the file is closed.
   That dict is only consumed by the safetensors branch; in the default
   TORCH_SAVE format it is pure retention. So even though ``_SerialCpuLoader``
   materializes tensors one at a time via blocking ``.cpu()``, the dict pins
   (references) the whole bucket in RAM at once — host RSS at the save window
   approaches the full checkpoint and drove MemAvailable to ~515 MB.

:class:`BoundedFileSystemWriter` overrides ``_write_data`` with a queue worker
that, for TORCH_SAVE, writes each tensor and drops the reference immediately
(no accumulating dict) and always uses ``_SerialCpuLoader`` regardless of
thread count. Measured on V100, FSDP + Adam, 1.6 GB saved state:

=============================  ===============  ===========  ===========
writer                          save-window RSS  post-save    pinned /dev/zero
                                increment        RSS
=============================  ===============  ===========  ===========
upstream default (tc=1)         +2.51 GB         stays 3.6 GB  ~full state
tc=8 (T5g, no accumulation)    +1.50 GB         back to base 0
tc=2 (default here)             +1.28 GB         back to base 0
tc=1 (minimum window)           +1.02 GB         back to base 0
=============================  ===============  ===========  ===========

The window increment scales with the number of concurrently materialized
tensors (~write workers), not the checkpoint total; ``XRL_DCP_WRITE_THREADS``
trades a little host headroom for concurrent writeback on slow FUSE targets.
"""

from __future__ import annotations

import os
import queue
import shutil
import threading
from io import UnsupportedOperation
from typing import Any, Callable

import torch
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint import save as dcp_save
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
from torch.futures import Future

import json

#: File shards / concurrent write workers for the resume save. Each worker
#: materializes one host tensor at a time (blocking .cpu(), no pinning), so the
#: live save-window footprint scales with this count. Measured FSDP+Adam on
#: V100 (1.6 GB saved state): tc=1 +1.0 GB / 13.9 s, tc=2 +1.28 GB / 12.8 s,
#: tc=4 +1.54 GB / 12.4 s, tc=8 +1.50 GB / 12.0 s on local disk; on the
#: write-through 3FS-FUSE target more shards overlap slow writeback (~97 vs
#: ~68 MB/s). 2 is the default balance; override with XRL_DCP_WRITE_THREADS.
_DCP_WRITE_THREADS = int(os.environ.get("XRL_DCP_WRITE_THREADS", "2"))


def _bounded_write_files_from_queue(
    create_stream: Callable,
    file_queue: "queue.Queue",
    result_queue: "queue.Queue",
    planner: Any,
    transforms: _StorageWriterTransforms,
    inflight_threshhold: int,
    use_fsync: bool,
    thread_count: int,
    serialization_format: SerializationFormat,
) -> None:
    """Write queued files one tensor at a time, never accumulating a bucket.

    Mirrors upstream ``_write_files_from_queue`` except:
    * always uses ``_SerialCpuLoader`` (blocking ``.cpu()`` -> no pinning), so
      it is safe at any thread count;
    * for the default TORCH_SAVE format each materialized tensor is written and
      then dropped, so a per-file ``tensor_dict`` never holds the whole bucket.
    Safetensors needs the full dict to serialize in one call, so that branch
    falls back to retaining tensors (Meshy resume saves use TORCH_SAVE).
    """
    try:
        while True:
            file_name, storage_key, write_items = file_queue.get_nowait()
            # Blocking loader: host pages are ordinary, not pinned.
            loader: _TensorLoader = _SerialCpuLoader(planner.resolve_data)

            tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
            for write_item in tensor_w:
                loader.add(_item_size(write_item), write_item)
            loader.start_loading()

            bytes_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
            write_results = []

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
                    # Stream one tensor through; release the reference as soon
                    # as it is serialized. This is the bounded-memory path.
                    for tensor, write_item in loader.values():
                        write_results.append(
                            _write_item(
                                transforms, stream, tensor, write_item, storage_key,
                                serialization_format,
                            )
                        )
                else:
                    # safetensors must serialize the whole set at once.
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
                stream.close()
            result_queue.put(write_results)
    except queue.Empty:
        pass


class BoundedFileSystemWriter(FileSystemWriter):
    """FileSystemWriter with a bounded per-save host footprint."""

    def _write_data(self, planner, file_queue) -> Future:  # type: ignore[override]
        result_queue: "queue.Queue" = queue.Queue()
        threads = []
        for _ in range(1, self.thread_count):
            t = threading.Thread(
                target=_bounded_write_files_from_queue,
                args=(
                    self.fs.create_stream,
                    file_queue,
                    result_queue,
                    planner,
                    self.transforms,
                    self.per_thread_copy_ahead,
                    self.sync_files,
                    self.thread_count,
                    self.serialization_format,
                ),
            )
            t.start()
            threads.append(t)

        _bounded_write_files_from_queue(
            create_stream=self.fs.create_stream,
            file_queue=file_queue,
            result_queue=result_queue,
            planner=planner,
            transforms=self.transforms,
            inflight_threshhold=self.per_thread_copy_ahead,
            use_fsync=self.sync_files,
            thread_count=self.thread_count,
            serialization_format=self.serialization_format,
        )

        for t in threads:
            t.join()
        res = []
        try:
            while True:
                res += result_queue.get_nowait()
        except queue.Empty:
            fut = Future()
            fut.set_result(res)
            return fut


def make_bounded_dcp_save(orig: Callable[..., Any]) -> Callable[..., Any]:
    """Bind onto ``CheckpointManager.dcp_save`` for the synchronous resume save.

    HF export (``to_hf=True``) and async modes keep torchtitan's writer.
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
