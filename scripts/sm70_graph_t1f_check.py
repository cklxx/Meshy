#!/usr/bin/env python
"""T1f fast on-card checks (<5 min, one card):

1. mem_fraction_static=0.6 (colocate setting) + decode CUDA graph + TileLang:
   capture must succeed in ONE pass with no OOM/failure branch; greedy correct;
   release/resume then greedy still correct (graph survives a colocate cycle).
2. graph OFF (real --disable-cuda-graph) single + batch-64 tok/s, to confirm
   the ~8x graph delta with the switch actually applied.

Run (GPU must be held):
  MESHY_SGLANG_SM70=1 MESHY_SM70_TILELANG=1 python scripts/sm70_graph_t1f_check.py
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time

import requests

MODEL = "/data00/meshy/models/Qwen3-0.6B"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(port, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited (capture failure?)")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health_generate", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("not ready")


def start(port, graph_on, mem_fraction, log):
    from meshy.backend.sglang_sm70 import bootstrap_pythonpath, sm70_server_defaults

    env = os.environ.copy()
    env["MESHY_SGLANG_SM70"] = "1"
    env["MESHY_SM70_TILELANG"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["PATH"] = "/usr/local/cuda-12.4/bin:" + env.get("PATH", "")
    boot = bootstrap_pythonpath()
    env["PYTHONPATH"] = os.pathsep.join([boot, env.get("PYTHONPATH", "")])
    args = [sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL,
            "--host", "127.0.0.1", "--port", str(port), "--tp-size", "1",
            "--mem-fraction-static", str(mem_fraction)]
    for k, v in sm70_server_defaults(cuda_graph=graph_on, max_bs=64).items():
        f = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                args.append(f)
        else:
            args += [f, str(v)]
    lf = open(log, "ab", buffering=0)
    p = subprocess.Popen(args, env=env, stdout=lf, stderr=subprocess.STDOUT,
                         start_new_session=True)
    return p, lf


def gen(port, n=1, max_new=128):
    payload = {"text": "The capital of France is",
               "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new}}

    def one(_):
        r = requests.post(f"http://127.0.0.1:{port}/generate", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()
    if n == 1:
        return [one(0)]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(one, range(n)))


def timed(port, n):
    t0 = time.perf_counter()
    out = gen(port, n=n)
    dt = time.perf_counter() - t0
    def ct(o):
        x = o["meta_info"]["completion_tokens"]
        return len(x) if isinstance(x, list) else int(x)
    toks = sum(ct(o) for o in out)
    return round(toks / dt, 2), out[0]["text"]


def mem():
    return int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().strip())


report = {}

# ---- 1. graph ON at mem_fraction 0.6 ----
port = free_port(); log = "/data00/meshy/env/t1f_graph_on.log"
print("=== graph ON mem_fraction=0.6 ===", flush=True)
p, lf = start(port, True, 0.6, log)
try:
    wait_ready(port, p)
    cap = subprocess.run(["grep", "-cE", "Capture target decode CUDA graph end", log],
                         capture_output=True, text=True).stdout.strip()
    fails = subprocess.run(["grep", "-cE", "Capture cuda graph failed|out of memory", log],
                           capture_output=True, text=True).stdout.strip()
    s1, t1 = timed(port, 1)
    b1, _ = timed(port, 64)
    before = mem()
    requests.post(f"http://127.0.0.1:{port}/release_memory_occupation",
                  json={"tags": ["kv_cache", "weights"]}, timeout=120).raise_for_status()
    time.sleep(3); rel = mem()
    requests.post(f"http://127.0.0.1:{port}/resume_memory_occupation",
                  json={"tags": ["weights"]}, timeout=300).raise_for_status()
    requests.post(f"http://127.0.0.1:{port}/resume_memory_occupation",
                  json={"tags": ["kv_cache"]}, timeout=300).raise_for_status()
    after = ""
    for _ in range(30):
        after = gen(port, 1, 16)[0]["text"]
        if "Paris" in after:
            break
        time.sleep(2)
    report["graph_on_0.6"] = {
        "capture_success_lines": cap, "capture_failure_lines": fails,
        "single_tok_s": s1, "batch64_tok_s": b1, "greedy": t1,
        "mem_before": before, "mem_after_release": rel,
        "after_resume": after, "paris_ok": "Paris" in after,
        "capture_once_ok": cap == "1" and fails == "0",
    }
    print(json.dumps(report["graph_on_0.6"], indent=1)[:700], flush=True)
finally:
    try: os.killpg(os.getpgid(p.pid), signal.SIGTERM); p.wait(60)
    except Exception: pass
    lf.close(); time.sleep(6)

# ---- 2. graph OFF real (optional: T1F_INCLUDE_OFF=1; adds ~1.5 min) ----
if os.environ.get("T1F_INCLUDE_OFF") == "1":
    port = free_port(); log = "/data00/meshy/env/t1f_graph_off.log"
    print("=== graph OFF mem_fraction=0.6 ===", flush=True)
    p, lf = start(port, False, 0.6, log)
    try:
        wait_ready(port, p)
        disabled = subprocess.run(["grep", "-c", "Cuda graph is disabled because"],
                                  capture_output=True, text=True).stdout.strip()
        s0, _ = timed(port, 1)
        b0, _ = timed(port, 64)
        report["graph_off_0.6"] = {"disable_notice_lines": disabled,
                                   "single_tok_s": s0, "batch64_tok_s": b0}
        print(json.dumps(report["graph_off_0.6"], indent=1), flush=True)
    finally:
        try: os.killpg(os.getpgid(p.pid), signal.SIGTERM); p.wait(60)
        except Exception: pass
        lf.close()

    report["single_speedup"] = round(report["graph_on_0.6"]["single_tok_s"] / report["graph_off_0.6"]["single_tok_s"], 2)
    report["batch64_speedup"] = round(report["graph_on_0.6"]["batch64_tok_s"] / report["graph_off_0.6"]["batch64_tok_s"], 2)

with open("/data00/meshy/env/t1f_check.json", "w") as f:
    json.dump(report, f, indent=2)
print("T1F_CHECK_DONE", json.dumps({k: report[k] for k in ("single_speedup", "batch64_speedup") if k in report}), flush=True)
