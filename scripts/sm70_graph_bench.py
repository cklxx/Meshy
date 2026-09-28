"""sm70 CUDA-graph A/B benchmark for SGLang Qwen3-0.6B.

Starts one server per graph mode (on then off, order swap via --order) with the
sm70 patch + TileLang fused ops + triton attention, and for each:

* single-stream greedy decode tok/s and batch-N aggregate tok/s, each run twice
  hot (first run is warmup and discarded);
* the exact greedy text, compared token-for-token against the graph-off output;
* memory-saver release/resume, then a greedy generation that must still match.

Graph capture on V100 is the thing under test: a capture failure shows up as a
non-ready server with the traceback in the per-run log. Requires an awb-held
GPU; CPU is not used for compute.

    MESHY_SGLANG_SM70=1 MESHY_SM70_TILELANG=1 \
    python scripts/sm70_graph_bench.py --model /data00/meshy/models/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time

import requests

MODEL_DEFAULT = "/data00/meshy/models/Qwen3-0.6B"
PROMPT = (
    "Question: Natalia sold 48 clips to her friends in April, and then she "
    "sold half as many clips in May. How many clips did Natalia sell "
    "altogether in April and May? Give the final number.\nAnswer:"
)
MAX_NEW = 256
PARIS = ("The capital of France is", 16)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(port: int, proc, timeout: float = 1800.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited during startup (capture failure?)")
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health_generate", timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("server not ready in time")


def start_server(model: str, port: int, graph: str, max_bs: int, log: str):
    env = os.environ.copy()
    env["MESHY_SGLANG_SM70"] = "1"
    env["MESHY_SM70_TILELANG"] = env.get("MESHY_SM70_TILELANG", "1")
    env["MESHY_SM70_CUDA_GRAPH"] = graph
    env["MESHY_SM70_CUDA_GRAPH_MAX_BS"] = str(max_bs)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["PATH"] = "/usr/local/cuda-12.4/bin:" + env.get("PATH", "")

    from meshy.backend.sglang_sm70 import bootstrap_pythonpath, sm70_server_defaults

    boot = bootstrap_pythonpath()
    env["PYTHONPATH"] = (
        os.pathsep.join([boot, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else boot
    )
    args = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model, "--host", "127.0.0.1", "--port", str(port),
        "--tp-size", "1",
    ]
    for key, value in sm70_server_defaults(
        cuda_graph=(graph == "on"), max_bs=max_bs
    ).items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args += [flag, str(value)]
    lf = open(log, "ab", buffering=0)
    proc = subprocess.Popen(args, env=env, stdout=lf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    return proc, lf


def gen(port, prompt=PROMPT, n=1, max_new=MAX_NEW):
    payload = {"text": prompt, "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new}}

    def one(_):
        r = requests.post(f"http://127.0.0.1:{port}/generate", json=payload, timeout=900)
        r.raise_for_status()
        return r.json()

    if n == 1:
        return [one(0)]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(one, range(n)))


def count_tokens(o):
    ct = o.get("meta_info", {}).get("completion_tokens", 0)
    return len(ct) if isinstance(ct, (list, tuple)) else int(ct or 0)


def timed(port, n):
    t0 = time.perf_counter()
    outs = gen(port, n=n)
    dt = time.perf_counter() - t0
    toks = sum(count_tokens(o) for o in outs)
    return {
        "requests": n, "wall_s": round(dt, 3), "total_new_tokens": toks,
        "tok_s": round(toks / dt, 2) if toks else None,
        "text": outs[0].get("text", ""),
    }


def mem_mib():
    return int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode().strip())


def release_resume(port):
    before = mem_mib()
    requests.post(f"http://127.0.0.1:{port}/release_memory_occupation",
                  json={"tags": ["kv_cache", "weights"]}, timeout=120).raise_for_status()
    time.sleep(3)
    released = mem_mib()
    requests.post(f"http://127.0.0.1:{port}/resume_memory_occupation",
                  json={"tags": ["weights"]}, timeout=300).raise_for_status()
    requests.post(f"http://127.0.0.1:{port}/resume_memory_occupation",
                  json={"tags": ["kv_cache"]}, timeout=300).raise_for_status()
    # resume is async: poll until output is restored
    text = ""
    for _ in range(30):
        text = gen(port, prompt=PARIS[0], n=1, max_new=PARIS[1])[0].get("text", "")
        if "Paris" in text:
            break
        time.sleep(2)
    return before, released, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--out", default="/data00/meshy/env/graph_bench.json")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-bs", type=int, default=64)
    ap.add_argument("--order", default="on,off,off,on", help="graph flag sequence")
    args = ap.parse_args()

    sequence = [s.strip() for s in args.order.split(",")]
    runs = []
    graph_off_text = None
    for i, flag in enumerate(sequence):
        port = free_port()
        log = f"/data00/meshy/env/graph_{flag}_{i}.log"
        print(f"=== run {i+1}/{len(sequence)} graph={flag} port={port} ===", flush=True)
        proc, lf = start_server(args.model, port, flag, args.max_bs, log)
        try:
            wait_ready(port, proc)
            gen(port, n=1, max_new=8)  # warmup
            s1 = timed(port, 1)
            sb = timed(port, args.batch)
            if flag == "off":
                graph_off_text = s1["text"]
            # release/resume only on the first "on" of each mode (keeps bench bounded)
            resume = None
            if flag == "on" and not any(r.get("resume") for r in runs if r["graph"] == "on"):
                before, released, after = release_resume(port)
                resume = {"mem_before_mib": before, "mem_released_mib": released,
                          "after_resume_text": after, "paris_ok": "Paris" in after}
            runs.append({
                "run": i, "graph": flag,
                "single": {k: s1[k] for k in ("wall_s", "total_new_tokens", "tok_s")},
                f"batch{args.batch}": {
                    k: sb[k] for k in ("wall_s", "total_new_tokens", "tok_s")},
                "greedy_text": s1["text"],
                "resume": resume,
            })
            print(json.dumps(runs[-1], indent=2)[:500], flush=True)
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=60)
            except Exception:
                pass
            lf.close()
            time.sleep(5)

    # every run's greedy output must match graph-off token-for-token
    mismatches = [r["run"] for r in runs if graph_off_text
                  and r["greedy_text"].strip() != graph_off_text.strip()]
    resume_runs = [r["resume"] for r in runs if r.get("resume")]
    report = {
        "model": args.model, "order": sequence, "batch": args.batch,
        "max_graph_bs": args.max_bs,
        "greedy_matches_graph_off": mismatches == [],
        "mismatched_runs": mismatches,
        "resume_after_release": resume_runs,
        "runs": runs,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("GRAPH_BENCH_DONE matches=", mismatches == [], "->", args.out, flush=True)


if __name__ == "__main__":
    main()
