#!/usr/bin/env python3
"""3FS storage-budget watchdog for the V100 GRPO/MATH runs (T5i).

One-screen report for cron / pre-flight / patrol:

  * logical 3FS usage         — bytes the filesystem actually hands out
                                 (``du`` under the FUSE mount); what the run
                                 can still allocate
  * physical /data00 usage    — the preallocated chunk-engine slab footprint
                                 (df); this DOES NOT shrink on delete, see
                                 docs/3fs_v100.md §12.5
  * effective writable headroom and, from the per-window growth rate, how many
    more rollout windows fit before the logical budget is exhausted.

Everything is read-only. Override the budget / window size via env:
  XRL_BUDGET_GB       logical budget in GiB the runs may consume (default 200)
  XRL_WINDOW_GB       expected logical GB written per window (default 1.0)
  XRL_MESHY_ROOT      meshy storage root (default /data00/meshy/store)
  XRL_LOCAL_MOUNT     physical data mount (default /data00)
"""

from __future__ import annotations

import os
import subprocess
import time

ROOT = os.environ.get("XRL_MESHY_ROOT", "/data00/meshy/store")
LOCAL = os.environ.get("XRL_LOCAL_MOUNT", "/data00")
BUDGET_GB = float(os.environ.get("XRL_BUDGET_GB", "200"))
WINDOW_GB = float(os.environ.get("XRL_WINDOW_GB", "1.0"))


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return out.stdout.strip()
    except Exception:
        return ""


def du_gb(path: str) -> float:
    if not os.path.isdir(path):
        return 0.0
    raw = _run(["du", "-sm", path])
    # du -sm: "12345\t/path"
    return float(raw.split()[0]) / 1024 if raw and raw.split()[0].isdigit() else 0.0


def df(path: str) -> tuple[float, float]:
    """Return (used_gb, avail_gb) for a mount."""
    st = os.statvfs(path)
    used = (st.f_blocks - st.f_bfree) * st.f_frsize / 1e9
    avail = st.f_bavail * st.f_frsize / 1e9
    return used, avail


def latest_window_marker(root: str) -> tuple[float, int, str]:
    """Estimate per-window growth from the active run's trajectories mtime.

    Returns (current_size_gb, windows_seen, run_name); windows are inferred
    from sample lines / 512 when a trajectories file is present.
    """
    rollout = os.path.join(root, "rollout")
    best = None
    if os.path.isdir(rollout):
        for name in os.listdir(rollout):
            tj = os.path.join(rollout, name, "trajectories.jsonl")
            if os.path.isfile(tj):
                mt = os.path.getmtime(tj)
                if best is None or mt > best[0]:
                    best = (mt, tj, name)
    if not best:
        return 0.0, 0, ""
    _, tj, name = best
    size = os.path.getsize(tj) / 1e9
    lines = 0
    try:
        with open(tj, "rb") as fh:
            for _ in fh:
                lines += 1
    except Exception:
        pass
    windows = lines // 512  # 64 prompts x 8 samples per window
    return size, windows, name


def main() -> None:
    total_logical = du_gb(ROOT)
    by_part = {
        sub: du_gb(os.path.join(ROOT, sub))
        for sub in ("rollout", "ckpt", "kvcache")
        if os.path.isdir(os.path.join(ROOT, sub))
    }
    phys_used, phys_avail = df(LOCAL)
    logical_free = max(BUDGET_GB - total_logical, 0.0)
    traj_gb, windows, run = latest_window_marker(ROOT)

    print(f"storage watchdog  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"root          {ROOT}")
    print(f"used          {total_logical:7.1f} GB   (budget {BUDGET_GB:.0f} GB, "
          f"headroom {logical_free:7.1f} GB)")
    for k in ("rollout", "ckpt", "kvcache"):
        if k in by_part:
            print(f"  - {k:8s} {by_part[k]:7.1f} GB")
    print(f"disk          {LOCAL} used {phys_used:7.1f} GB, df-avail {phys_avail:7.1f} GB"
          f"  (local /data00; delete frees space)")
    if run:
        print(f"active run    {run}: {windows} windows, trajectories {traj_gb:.2f} GB")
    budget_windows = int(logical_free / WINDOW_GB) if WINDOW_GB > 0 else 0
    print(f"windows left  ~{budget_windows} at {WINDOW_GB:.2f} GB/window logical write")
    # Exit non-zero only when the logical budget is the binding constraint and
    # less than one window remains.
    if logical_free < WINDOW_GB:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
