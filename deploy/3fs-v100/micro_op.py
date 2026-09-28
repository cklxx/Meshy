#!/usr/bin/env python3
"""Micro-benchmark: isolate per-op fixed cost vs transfer size.

Issues a SINGLE io_uring batch of N page-sized ops per submit, serial
submit->wait, on one client. Varies op size (one big iov op vs many 64KiB ops)
to tell whether the ~10 MiB/s write ceiling is per-op RTT (soft-RoCE) or
per-byte throughput. One ring, no threads.
"""
import datetime
import multiprocessing
import os
import sys
import time

import numpy as np
import torch
from hf3fs_fuse.io import (deregister_fd, extract_mount_point,
                           make_iovec, make_ioring, register_fd)

path = sys.argv[1]
total_mib = int(sys.argv[2]) if len(sys.argv) > 2 else 32
op_kibs = [int(x) for x in sys.argv[3:]] or [64, 256, 1024, 4096]

fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
size = total_mib * 2**20
os.ftruncate(fd, size)
register_fd(fd)
mnt = extract_mount_point(path)

# big enough shm/ring for the largest op
maxop = max(op_kibs) * 1024
entries = max(8, maxop // 65536)
shm = multiprocessing.shared_memory.SharedMemory(size=maxop*2, create=True)
buf = torch.frombuffer(shm.buf, dtype=torch.uint8)
ior_w = make_ioring(mnt, entries*2, for_read=False, timeout=1)
ior_r = make_ioring(mnt, entries*2, for_read=True, timeout=1)
iov = make_iovec(shm, mnt)
shm.unlink()

rng = np.random.RandomState(1)
print(f"{'opKiB':>7} {'nOps':>6} {'wMiB/s':>9} {'rMiB/s':>9}")
for opk in op_kibs:
    op = opk * 1024
    n = size // op
    # lay payload into shm once
    buf[:op].copy_(torch.from_numpy(rng.randint(0,256,dtype=np.uint8,size=op)))
    # WRITE: each iteration one op of `op` bytes at a fresh offset
    t0 = time.perf_counter()
    for i in range(n):
        iov2 = iov  # same slot; reuse
        ior_w.prepare(iov[0:op], False, fd, i*op)
        res = list(ior_w.submit().wait(min_results=1,
                    timeout=datetime.timedelta(seconds=30)))
        assert res and res[0].result == op, res
    os.fsync(fd)
    tw = time.perf_counter()-t0
    t0 = time.perf_counter()
    for i in range(n):
        ior_r.prepare(iov[0:op], True, fd, i*op)
        res = list(ior_r.submit().wait(min_results=1,
                    timeout=datetime.timedelta(seconds=30)))
        assert res and res[0].result == op, res
    tr = time.perf_counter()-t0
    print(f"{opk:>7} {n:>6} {size/2**20/tw:>9.1f} {size/2**20/tr:>9.1f}",
          flush=True)

deregister_fd(fd); os.close(fd); shm.close()
