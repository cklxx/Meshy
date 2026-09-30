"""T3h/T11 pre-flight: flash-decode vs triton colocate memory handoff.

Runs two release/resume cycles of the torch memory saver on a live
graph-enabled sm70 server, with flash decode on vs off, and records GPU
memory at each phase. Confirms the flash-decode plan workspaces (~3.6 MB
across all 12 graph buckets) do not thin the resume headroom that
already OOMed once (torch_memory_saver cuMemCreate / csrc core.cpp).

Run on the V100 with the GPU held, CUDA 12.4 nvcc on PATH:

    python scripts/sm70_flash_decode_colocate_mem.py
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import requests

MODEL_DEFAULT = "/data00/meshy/models/Qwen3-0.6B"


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def gpu_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    return float(out.stdout.strip())


def start(model, port, flash):
    env = os.environ.copy()
    env.update({
        "MESHY_SGLANG_SM70": "1",
        "MESHY_SM70_FLASH_DECODE": "1" if flash else "0",
        "MESHY_SM70_CUDA_GRAPH": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "PATH": "/usr/local/cuda-12.4/bin:" + env.get("PATH", ""),
    })
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from meshy.backend.sglang_sm70 import bootstrap_pythonpath, sm70_server_defaults
    boot = bootstrap_pythonpath()
    env["PYTHONPATH"] = boot + os.pathsep + env.get("PYTHONPATH", "")
    args = [sys.executable, "-m", "sglang.launch_server",
            "--model-path", model, "--host", "127.0.0.1", "--port", str(port),
            "--tp-size", "1", "--mem-fraction-static", "0.6"]
    for k, v in sm70_server_defaults().items():
        f = "--" + k.replace("_", "-")
        args += [f] if isinstance(v, bool) and v else [f, str(v)]
    log = open(f"/data00/meshy/kern/logs/colocate_flash{flash}.log", "ab")
    return subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True), log


def wait_ready(port, proc, t=1800):
    end = time.time() + t
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError("server exited")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health_generate",
                            timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("not ready")


def call(base, path, payload=None):
    r = requests.post(base + path, json=payload or {}, timeout=600)
    return r.status_code, r.text[:200]


def wait_idle(base):
    for _ in range(200):
        try:
            j = requests.get(base + "/v1/loads", timeout=10).json()
            loads = j.get("loads") or []
            n = sum(int(x.get("num_running_reqs", 0))
                    + int(x.get("num_waiting_reqs", 0)) for x in loads)
            if n == 0:
                return
        except requests.RequestException:
            pass
        time.sleep(2)


def arm(model, flash):
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc, log = start(model, port, flash)
    rec = {"flash": flash}
    try:
        wait_ready(port, proc)
        # Drive a decode across graph buckets so plans are materialized.
        requests.post(base + "/generate", timeout=900, json={
            "text": "Question: 2+3? Answer:",
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 16}})
        time.sleep(4)
        rec["used_after_capture_mib"] = gpu_mib()
        for cyc in (1, 2):
            call(base, "/pause_generation", {"mode": "abort"})
            wait_idle(base)
            call(base, "/release_memory_occupation",
                 {"tags": ["kv_cache", "weights"]})
            time.sleep(8)
            rec[f"cycle{cyc}_released_mib"] = gpu_mib()
            sc, body = call(base, "/resume_memory_occupation",
                            {"tags": ["kv_cache", "weights"]})
            time.sleep(10)
            rec[f"cycle{cyc}_resume_http"] = sc
            rec[f"cycle{cyc}_resume_body"] = body
            rec[f"cycle{cyc}_resumed_mib"] = gpu_mib()
            call(base, "/continue_generation", {})
            rec[f"cycle{cyc}_ok"] = (
                sc == 200 and proc.poll() is None
                and "out of memory" not in open(log.name, errors="ignore").read()
                    [-6000:].lower())
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=120)
        except Exception:
            pass
        log.close()
        # Poll until GPU drains.
        for _ in range(40):
            if gpu_mib() < 50:
                break
            time.sleep(3)
        rec["drained_mib"] = gpu_mib()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--out", default="/data00/meshy/kern/colocate_mem.json")
    args = ap.parse_args()
    results = []
    for flash in (False, True):
        r = arm(args.model, flash)
        results.append(r)
        print(json.dumps(r), flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    # Delta: resumed footprint flash vs triton.
    t = next(r for r in results if not r["flash"])
    fl = next(r for r in results if r["flash"])
    for cyc in (1, 2):
        d = fl[f"cycle{cyc}_resumed_mib"] - t[f"cycle{cyc}_resumed_mib"]
        print(json.dumps({"cycle": cyc, "flash_resumed_delta_mib": round(d, 1)}),
              flush=True)
    print("COLOCATE_MEM_DONE ->", args.out)


if __name__ == "__main__":
    main()
