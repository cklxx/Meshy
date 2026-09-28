#!/usr/bin/env python3
"""GPU/host memory peaks from smoke_mem.csv (2s sampler).

Sampler wrote 3 comma fields: ts, "MiB util", avail_gib (nvidia-smi
query gap): parse defensively.
"""
import csv
import sys

ts0 = None
g, a, gmin_t = [], [], None
n = 0
for line in open(sys.argv[1]):
    parts = line.strip().split(",")
    if len(parts) < 3 or not parts[0].isdigit():
        continue
    ts = int(parts[0]); f2 = parts[1].split()
    mib = int(f2[0])
    avail = int(parts[2])
    if ts0 is None:
        ts0 = ts
    g.append((ts, mib)); a.append((ts, avail)); n += 1

peak = max(g, key=lambda x: x[1])
amin = min(a, key=lambda x: x[1])
print(f"samples={n} span_s={(a[-1][0]-ts0)}")
print(f"gpu_peak_MiB={peak[1]} at_min{round((peak[0]-ts0)/60,1)} gpu_end={g[-1][1]}")
print(f"host_avail_min_GiB={amin[1]} at_min{round((amin[0]-ts0)/60,1)} host_avail_end={a[-1][1]}")
# gpu residency distribution by minute
bymin = {}
for ts, mib in g:
    bymin.setdefault((ts - ts0) // 60, []).append(mib)
print("gpu max per minute:", {m: max(v) for m, v in sorted(bymin.items())})
