#!/usr/bin/env python
"""T1g on-card check (<10 min, one card): memory-saver release/resume in 3 modes.

All servers use production-real params (mirrors recipe/grpo_gsm8k_v100.py via
sm70_server_defaults): --enable-memory-saver --enable-weights-cpu-backup
--mem-fraction-static 0.6, triton attention, pytorch sampling, fp16, TileLang.

Modes:
  off           graph disabled
  on            decode CUDA graph, graph pool NOT saver-managed (5e3cc90 reverted)
  on_saver      decode CUDA graph + MESHY_SM70_SAVER_MANAGES_GRAPH=1
                (SGLANG_MEMORY_SAVER_CUDA_GRAPH, graph pool saver-managed)

Per mode: release-stable MiB (polled), release/resume latency, post-resume
greedy correctness, and whether decode still replays cuda graph after resume.

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
OUT = "/data00/meshy/env/t1g_check.json"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(port, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health_generate", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("not ready")


def start(port, mode, mem_fraction, log):
    from meshy.backend.sglang_sm70 import bootstrap_pythonpath, sm70_server_defaults

    env = os.environ.copy()
    env["MESHY_SGLANG_SM70"] = "1"
    env["MESHY_SM70_TILELANG"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["PATH"] = "/usr/local/cuda-12.4/bin:" + env.get("PATH", "")
    boot = bootstrap_pythonpath()
    env["PYTHONPATH"] = os.pathsep.join([boot, env.get("PYTHONPATH", "")])

    graph_on = mode != "off"
    saver_graph = mode == "on_saver"
    # Child env only: opt into the saver-managed graph pool for that mode.
    if saver_graph:
        env["MESHY_SM70_SAVER_MANAGES_GRAPH"] = "1"
    else:
        env.pop("MESHY_SM70_SAVER_MANAGES_GRAPH", None)
        env.pop("SGLANG_MEMORY_SAVER_CUDA_GRAPH", None)

    args = [sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL,
            "--host", "127.0.0.1", "--port", str(port), "--tp-size", "1",
            "--mem-fraction-static", str(mem_fraction)]
    for k, v in sm70_server_defaults(
        cuda_graph=graph_on, max_bs=64, saver_manages_graph=saver_graph
    ).items():
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


def gen(port, max_new=16):
    payload = {"text": "The capital of France is",
               "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new}}
    r = requests.post(f"http://127.0.0.1:{port}/generate", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["text"]


def timed_single(port, max_new=128):
    t0 = time.perf_counter()
    out = gen(port, max_new)
    dt = time.perf_counter() - t0
    return round(max_new / dt, 2), out


def mem():
    return int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().strip())


def wait_mem_settle(stop=60.0, stable_for=3.0):
    t0 = time.time()
    last, stable_since = mem(), time.time()
    trace = [last]
    while time.time() - t0 < stop:
        time.sleep(1.0)
        cur = mem()
        trace.append(cur)
        if cur != last:
            last, stable_since = cur, time.time()
        elif time.time() - stable_since >= stable_for:
            return cur, trace
    return last, trace


def grep_count(log, pattern):
    return subprocess.run(["grep", "-cE", pattern, log],
                          capture_output=True, text=True).stdout.strip() or "0"


def run_mode(mode):
    port = free_port()
    log = f"/data00/meshy/env/t1g_{mode}.log"
    open(log, "w").close()
    print(f"=== {mode} ===", flush=True)
    p, lf = start(port, mode, 0.6, log)
    try:
        wait_ready(port, p)
        tok_s, greedy0 = timed_single(port)
        before = mem()
        graph_lines_before = int(grep_count(log, "cuda graph: True"))
        cap_begin = int(grep_count(log, "Capture target decode CUDA graph begin"))
        cap_fail = int(grep_count(log, "Capture cuda graph failed|out of memory"))

        tags = ["kv_cache", "weights"] + (["cuda_graph"] if mode == "on_saver" else [])

        def post(path, ts):
            t = time.perf_counter()
            requests.post(f"http://127.0.0.1:{port}/{path}",
                          json={"tags": ts}, timeout=600).raise_for_status()
            return round(time.perf_counter() - t, 3)

        release_http = post("release_memory_occupation", tags)
        t0 = time.perf_counter()
        rel, trace = wait_mem_settle()
        settle = round(time.perf_counter() - t0, 3)

        resume_s = {}
        if mode == "on_saver":
            resume_s["cuda_graph"] = post("resume_memory_occupation", ["cuda_graph"])
        resume_s["weights"] = post("resume_memory_occupation", ["weights"])
        resume_s["kv_cache"] = post("resume_memory_occupation", ["kv_cache"])

        after = ""
        for _ in range(30):
            after = gen(port)
            if "Paris" in after:
                break
            time.sleep(2)
        # one more decode to make the scheduler log its graph usage
        time.sleep(1)
        gen(port, 8)
        time.sleep(1)
        graph_lines_after = int(grep_count(log, "cuda graph: True"))
        cap_recapture = int(grep_count(log, "Capture target decode CUDA graph begin"))

        r = {
            "single_tok_s": tok_s, "greedy_before": greedy0[:60],
            "mem_before_release_mib": before,
            "mem_release_settled_mib": rel,
            "released_mib": before - rel,
            "release_tags": tags,
            "release_http_s": release_http, "release_settle_s": settle,
            "resume_http_s": resume_s,
            "after_resume": after[:60], "paris_ok": "Paris" in after,
            "capture_begin_count": cap_begin, "capture_fail_lines": cap_fail,
            "recaptured_during_resume": cap_recapture > cap_begin,
            "decode_graph_after_resume": (
                None if mode == "off" else graph_lines_after > graph_lines_before
            ),
            "mem_trace_mib": trace,
        }
        print(json.dumps(r, indent=1)[:1200], flush=True)
        return r
    finally:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM); p.wait(60)
        except Exception:
            pass
        lf.close()
        time.sleep(6)


report = {}
for mode in ("off", "on", "on_saver"):
    report[mode] = run_mode(mode)

with open(OUT, "w") as f:
    json.dump(report, f, indent=2)
print("T1G_CHECK_DONE", OUT, flush=True)
