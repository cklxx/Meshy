"""End-to-end sm70 fused-op comparison: native vs TileLang inside SGLang.

Starts one SGLang server per backend (MESHY_SM70_TILELANG 0 then 1, in the
order given by --order so warmup bias cancels over two repeats), then
measures:

* single-stream greedy decode tok/s  (batch 1, shared prompt);
* batch-64 greedy decode throughput    (64 identical prompts);
* greedy token agreement between native and TileLang outputs.

Qwen3-0.6B, triton attention, fp16, no CUDA graph. The first run per
backend is the JIT/prewarm warmup and is discarded; reported numbers are
the timed run. Requires the sm70 patch (MESHY_SGLANG_SM70=1).

    python scripts/sm70_e2e_bench.py --model /data00/meshy/models/Qwen3-0.6B

Outputs JSON to --out. Does NOT launch torchrun: point it at an already
configured venv. GPU is assumed held by the caller (awb hold).
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
    "Question: Natalia sold 48 clips to her friends in April, and then "
    "she sold half as many clips in May. How many clips did Natalia sell "
    "altogether in April and May? Give the final number.\nAnswer:"
)
MAX_NEW = 256


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(port: int, proc, timeout: float = 1800.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited during startup")
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health_generate", timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("server not ready in time")


def start_server(model: str, port: int, tilelang: str, log: str):
    env = os.environ.copy()
    env["MESHY_SGLANG_SM70"] = "1"
    env["MESHY_SM70_TILELANG"] = tilelang
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["PATH"] = "/usr/local/cuda-12.4/bin:" + env.get("PATH", "")
    from meshy.backend.sglang_sm70 import bootstrap_pythonpath, sm70_server_defaults

    boot = bootstrap_pythonpath()
    env["PYTHONPATH"] = (
        boot + os.pathsep + env.get("PYTHONPATH", "")
        if env.get("PYTHONPATH") else boot
    )
    args = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model, "--host", "127.0.0.1", "--port", str(port),
        "--tp-size", "1",
    ]
    defaults = sm70_server_defaults()
    for key, value in defaults.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            # store_true flags take no value; skip when False.
            if value:
                args.append(flag)
        else:
            args += [flag, str(value)]
    lf = open(log, "ab", buffering=0)
    proc = subprocess.Popen(args, env=env, stdout=lf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    return proc, lf


def generate(port: int, n: int, max_new: int = MAX_NEW):
    payload = {
        "input_ids": None,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new},
    }
    # Use the same prompt n times via /generate repeated single streams for
    # batch1 and the /generate parallel client for batch64.
    text_prompt = {"text": PROMPT, **payload}
    if n == 1:
        r = requests.post(f"http://127.0.0.1:{port}/generate", json=text_prompt,
                          timeout=600)
        r.raise_for_status()
        return [r.json()]
    # batch64: fire n concurrent requests; measure wall time.
    from concurrent.futures import ThreadPoolExecutor

    def one(_):
        rr = requests.post(f"http://127.0.0.1:{port}/generate", json=text_prompt,
                           timeout=900)
        rr.raise_for_status()
        return rr.json()

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(one, range(n)))


def timed_decode(port: int, n: int):
    torch_sync_hint()
    t0 = time.perf_counter()
    outs = generate(port, n)
    dt = time.perf_counter() - t0
    def _count(o):
        ct = o.get("meta_info", {}).get("completion_tokens", 0)
        return len(ct) if isinstance(ct, (list, tuple)) else int(ct or 0)

    toks = sum(_count(o) for o in outs)
    return {
        "requests": n,
        "wall_s": round(dt, 3),
        "total_new_tokens": toks,
        "tok_s": round(toks / dt, 2) if toks else None,
        "texts": [o.get("text", "") for o in outs],
    }


def torch_sync_hint():
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--out", default="/data00/meshy/kern/e2e_bench.json")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--order", default="0,1,1,0",
                    help="tilelang flag sequence: 0=native,1=tilelang")
    args = ap.parse_args()

    sequence = [s.strip() for s in args.order.split(",")]
    runs = []
    last_text = {"0": None, "1": None}
    for i, flag in enumerate(sequence):
        port = free_port()
        log = f"/data00/meshy/kern/logs/e2e_{flag}_{i}.log"
        print(f"=== run {i+1}/{len(sequence)} tilelang={flag} port={port} ===",
              flush=True)
        proc, lf = start_server(args.model, port, flag, log)
        try:
            wait_ready(port, proc)
            # Warmup (also triggers TileLang prewarm already at startup).
            generate(port, 1, max_new=8)
            s1 = timed_decode(port, 1)
            sb = timed_decode(port, args.batch)
            last_text[flag] = s1["texts"][0]
            runs.append({
                "run": i, "tilelang": flag,
                "single": {k: s1[k] for k in ("wall_s", "total_new_tokens", "tok_s")},
                f"batch{args.batch}": {
                    k: sb[k] for k in ("wall_s", "total_new_tokens", "tok_s")},
                "greedy_text": s1["texts"][0],
            })
            print(json.dumps(runs[-1], indent=2)[:600], flush=True)
        finally:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=60)
            lf.close()
            time.sleep(5)  # let the GPU memory free fully

    # Greedy agreement between the two backends (exact token sequence).
    agree = None
    if last_text["0"] and last_text["1"]:
        agree = last_text["0"].strip() == last_text["1"].strip()
    report = {
        "model": args.model,
        "order": sequence,
        "greedy_text_exact_match": agree,
        "runs": runs,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("E2E_BENCH_DONE", "greedy_match=", agree, "->", args.out, flush=True)


if __name__ == "__main__":
    main()
