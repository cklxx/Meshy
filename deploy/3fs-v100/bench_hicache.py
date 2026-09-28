#!/usr/bin/env python3
"""Parameter-sweep benchmark for the SGLang hf3fs HiCache usrbio backend.

Mirrors storage_hf3fs.HiCacheHF3FS: ``numjobs`` independent clients (each its
own fd + shared-memory segment + read/write io_uring), dispatched round-robin
by a ThreadPoolExecutor; each client batches ``entries`` page-sized ops into
one submit().wait(). Payload is pre-generated OUTSIDE the timed region so the
numbers reflect io only. Sweeps numjobs x entries x page size to locate the
write bottleneck on the soft-RoCE single node.

Only imports hf3fs_fuse.io + torch/numpy (no sglang / sgl_kernel).
"""
import argparse
import concurrent.futures
import datetime
import multiprocessing
import os
import time

import numpy as np
import torch
from hf3fs_fuse.io import (
    deregister_fd,
    extract_mount_point,
    make_iovec,
    make_ioring,
    register_fd,
)


class Client:
    def __init__(self, path, mnt, size, page, entries):
        self.page, self.entries = page, entries
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        os.ftruncate(self.fd, size)
        register_fd(self.fd)
        cap = page * entries
        self.shm_w = multiprocessing.shared_memory.SharedMemory(size=cap, create=True)
        self.shm_r = multiprocessing.shared_memory.SharedMemory(size=cap, create=True)
        self.buf_w = torch.frombuffer(self.shm_w.buf, dtype=torch.uint8)
        self.buf_r = torch.frombuffer(self.shm_r.buf, dtype=torch.uint8)
        self.ior_w = make_ioring(mnt, entries, for_read=False, timeout=1)
        self.ior_r = make_ioring(mnt, entries, for_read=True, timeout=1)
        self.iov_w = make_iovec(self.shm_w, mnt)
        self.iov_r = make_iovec(self.shm_r, mnt)
        self.shm_w.unlink(); self.shm_r.unlink()

    def _run(self, ior, iov, buf, file_offsets, page_bytes, write):
        # SGLang: each op maps one contiguous page slot in shm, at its own
        # file offset. prepare(is_read, fd, offset).
        for j, off in enumerate(file_offsets):
            ior.prepare(iov[j*self.page:(j+1)*self.page], not write,
                        self.fd, off)
        res = list(ior.submit().wait(
            min_results=len(file_offsets), timeout=datetime.timedelta(seconds=60)))
        if len(res) != len(file_offsets):
            raise RuntimeError(f"only {len(res)}/{len(file_offsets)} completed")
        for r in res:
            if r.result < 0:
                raise OSError(-r.result)

    def batch_write(self, file_offsets, page_bytes):
        self.buf_w[:len(page_bytes)].copy_(
            torch.from_numpy(page_bytes.reshape(-1)))
        self._run(self.ior_w, self.iov_w, self.buf_w, file_offsets,
                  page_bytes, True)

    def batch_read(self, file_offsets, nbytes):
        self._run(self.ior_r, self.iov_r, self.buf_r, file_offsets,
                  None, False)
        return self.buf_r[:nbytes].numpy().copy()

    def close(self):
        deregister_fd(self.fd); os.close(self.fd)
        self.shm_w.close(); self.shm_r.close()


def run(path, size, total, page, entries, numjobs, payload):
    fd0 = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd0, size); os.close(fd0)
    mnt = extract_mount_point(path)
    clients = [Client(path, mnt, size, page, entries) for _ in range(numjobs)]

    n_pages = total // page
    pages = np.arange(n_pages, dtype=np.int64)
    # round-robin entries-sized chunks over clients (SGLang AtomicCounter)
    batches = []
    for k, i in enumerate(range(0, n_pages, entries)):
        sl = pages[i:min(i+entries, n_pages)]
        batches.append((k % numjobs, sl))

    def do_write(b):
        c, sl = b
        clients[c].batch_write((sl*page).tolist(), payload[sl[0]*page:(sl[-1]+1)*page])

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=numjobs) as ex:
        list(ex.map(do_write, batches))
    os.sync()
    wbw = total / 2**20 / (time.perf_counter() - t0)

    def do_read(b):
        c, sl = b
        return sl, clients[c].batch_read((sl*page).tolist(), len(sl)*page)

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=numjobs) as ex:
        rres = list(ex.map(do_read, batches))
    rbw = total / 2**20 / (time.perf_counter() - t0)

    got = np.empty(total, dtype=np.uint8)
    for sl, chunk in rres:
        got[sl[0]*page:(sl[-1]+1)*page] = chunk.reshape(-1)
    ok = np.array_equal(got, payload[:total])

    for c in clients:
        c.close()
    return wbw, rbw, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--size-mib", type=int, default=2048)
    ap.add_argument("--total-mib", type=int, default=64)
    ap.add_argument("--numjobs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--entries", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--page-kib", type=int, nargs="+", default=[64])
    args = ap.parse_args()

    # deterministic payload generated once, outside timing
    rng = np.random.RandomState(0)
    payload = rng.randint(0, 256, dtype=np.uint8, size=args.size_mib*2**20)
    # ensure file holds at least the random payload region used
    print(f"path={args.path} total={args.total_mib}MiB (payload pre-generated)")
    print(f"{'pageKiB':>7} {'entries':>7} {'numjobs':>7} "
          f"{'writeMiB/s':>11} {'readMiB/s':>10}  ok")
    for pk in args.page_kib:
        for e in args.entries:
            for j in args.numjobs:
                try:
                    w, r, ok = run(args.path, args.size_mib*2**20,
                                   args.total_mib*2**20, pk*1024, e, j, payload)
                    print(f"{pk:>7} {e:>7} {j:>7} {w:>11.1f} {r:>10.1f}  "
                          f"{'OK' if ok else 'FAIL'}", flush=True)
                except Exception as ex:
                    print(f"{pk:>7} {e:>7} {j:>7} {'ERR':>11} {'':>10}  "
                          f"{type(ex).__name__}: {ex}", flush=True)


if __name__ == "__main__":
    main()
