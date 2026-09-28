#!/usr/bin/env python3
"""Standalone hf3fs usrbio put/get validation for SGLang HiCache.

Mirrors sglang.srt.mem_cache.storage.hf3fs.hf3fs_usrbio_client.Hf3fsUsrBioClient
(minus the sglang-specific metadata/shm plumbing), using only the usrbio wheel
(hf3fs_fuse.io -> hf3fs_py_usrbio) and torch shared memory. This lets us test
the L3 storage backend on the V100 while sglang itself still cannot import
(libnvrtc.so.13 for sgl_kernel).

Reads the same JSON SGLang reads via SGLANG_HICACHE_HF3FS_CONFIG_PATH:
  {file_path_prefix, file_size, numjobs, entries}
Creates/preallocates <prefix>.<rank>.bin, writes deterministic bytes through
the write io_uring, reads them back through the read io_ring, checksums both,
and reports sequential put/get bandwidth.

Run inside the venv with the usrbio wheel installed and /3fs mounted.
"""
import datetime
import hashlib
import json
import multiprocessing
import os
import sys
import time

import torch
from hf3fs_fuse.io import (
    deregister_fd,
    extract_mount_point,
    make_iovec,
    make_ioring,
    register_fd,
)

# Single rank for the standalone check; size must be < file_size.
RANK = 0
PAGE = 64 * 1024                      # one usrbio "page" (HiCache page_size=64)
N_PAGES_DEFAULT = 4096               # 4096 * 64 KiB = 256 MiB


class UsrBioProbe:
    def __init__(self, path: str, file_size: int, entries: int):
        self.path = path
        self.size = file_size
        self.entries = entries
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        os.ftruncate(self.fd, file_size)
        register_fd(self.fd)
        self.mnt = extract_mount_point(path)
        self.bs = PAGE
        self.shm_w = multiprocessing.shared_memory.SharedMemory(
            size=self.bs * entries, create=True)
        self.shm_r = multiprocessing.shared_memory.SharedMemory(
            size=self.bs * entries, create=True)
        self.buf_w = torch.frombuffer(self.shm_w.buf, dtype=torch.uint8)
        self.buf_r = torch.frombuffer(self.shm_r.buf, dtype=torch.uint8)
        self.ior_w = make_ioring(self.mnt, entries, for_read=False,
                                 timeout=1, numa=-1)
        self.ior_r = make_ioring(self.mnt, entries, for_read=True,
                                 timeout=1, numa=-1)
        self.iov_w = make_iovec(self.shm_w, self.mnt)
        self.iov_r = make_iovec(self.shm_r, self.mnt)
        self.shm_w.unlink()
        self.shm_r.unlink()

    def _round(self, ior, iov, shm_tensor, fd, payload: bytes, base: int,
              write: bool):
        """Submit ``payload`` as page-sized ops (one per PAGE), SGLang-style.

        The ring holds ``entries`` ops; submit in batches of that depth. Each
        op maps one iov byte-range [off:off+PAGE] at the matching file offset.
        """
        nops = len(payload) // self.bs
        for lo in range(0, nops, self.entries):
            hi = min(lo + self.entries, nops)
            if write:
                seg = payload[lo*self.bs:hi*self.bs]
                shm_tensor[:len(seg)].copy_(
                    torch.frombuffer(bytearray(seg), dtype=torch.uint8))
            for j in range(lo, hi):
                # prepare(is_read, fd, offset): write ring -> is_read=False.
                ior.prepare(iov[(j-lo)*self.bs:(j-lo+1)*self.bs], not write,
                            fd, j*self.bs + base)
            resv = list(ior.submit().wait(
                min_results=hi-lo, timeout=datetime.timedelta(seconds=60)))
            assert len(resv) == hi-lo, f"{len(resv)} != {hi-lo}"
            for r in resv:
                if r.result < 0:
                    raise OSError(-r.result)
            if not write:
                seg = bytes(shm_tensor[:(hi-lo)*self.bs].numpy().tobytes())
                yield seg

    def write_all(self, data: bytes) -> None:
        assert len(data) % self.bs == 0
        for _ in self._round(self.ior_w, self.iov_w, self.buf_w, self.fd,
                             data, 0, True):
            pass
        os.fsync(self.fd)

    def read_all(self, n: int) -> bytes:
        return b"".join(self._round(self.ior_r, self.iov_r, self.buf_r,
                                    self.fd, b"\x00"*n, 0, False))

    def close(self):
        deregister_fd(self.fd)
        os.close(self.fd)
        del self.ior_w, self.ior_r, self.iov_w, self.iov_r
        self.shm_w.close()
        self.shm_r.close()


def main() -> int:
    cfg_path = os.environ["SGLANG_HICACHE_HF3FS_CONFIG_PATH"]
    with open(cfg_path) as f:
        cfg = json.load(f)
    n_pages = int(os.environ.get("USRBIO_NPAGES", N_PAGES_DEFAULT))
    n_bytes = n_pages * PAGE
    assert n_bytes <= cfg["file_size"], f"{n_bytes} > file_size {cfg['file_size']}"

    path = f"{cfg['file_path_prefix']}.{RANK}.bin"
    print(f"file={path} bytes={n_bytes} ({n_bytes/2**20:.0f} MiB)")
    p = UsrBioProbe(path, cfg["file_size"], cfg["entries"])
    print(f"mount={p.mnt}")
    try:
        data = bytes((i * 7 + 3) & 0xFF for i in range(4096))
        data = (data * (n_bytes // 4096 + 1))[:n_bytes]
        h_w = hashlib.sha256(data).hexdigest()

        t0 = time.perf_counter()
        p.write_all(data)
        t_w = time.perf_counter() - t0

        t0 = time.perf_counter()
        got = p.read_all(n_bytes)
        t_r = time.perf_counter() - t0
        h_r = hashlib.sha256(got).hexdigest()

        print(f"sha write={h_w}")
        print(f"sha read ={h_r}")
        print(f"INTEGRITY_{'OK' if h_w == h_r else 'FAIL'}")
        print(f"usrbio put BW = {n_bytes/2**20/t_w:8.1f} MiB/s")
        print(f"usrbio get BW = {n_bytes/2**20/t_r:8.1f} MiB/s")
        return 0 if h_w == h_r else 1
    finally:
        p.close()


if __name__ == "__main__":
    sys.exit(main())
