"""T3g e2e: flash-decode vs triton inside SGLang on sm70.

Two parts (run on the V100 with the GPU held):

* consistency: same GSM8K prompt, greedy, compare the full token stream
  (or logprob diff) between triton and flash decode under BOTH eager
  (MESHY_SM70_CUDA_GRAPH=0) and graph (=1, production default);
* throughput: 512 GSM8K prompts x 8 samples each (4096 requests), the
  clean40b production config (mem 0.60, decode graph on, bs<=64),
  compare whole-batch wall time and aggregate decode tok/s, triton vs
  flash, in crossed order.

Each arm starts its own server with the rl/sm70 defaults; the only
varying flags are MESHY_SM70_FLASH_DECODE and MESHY_SM70_CUDA_GRAPH.

    python scripts/sm70_flash_decode_e2e.py --model /data00/meshy/models/Qwen3-0.6B
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
from concurrent.futures import ThreadPoolExecutor

MODEL_DEFAULT = "/data00/meshy/models/Qwen3-0.6B"
DATA_DEFAULT = "/data00/meshy/models/gsm8k"
SUFFIX = ' Let\'s think step by step and output the final answer after "####".'


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(port, proc, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited during startup")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health_generate",
                            timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("server not ready")


def load_gsm8k_prompts(model_path, data_dir, n):
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(model_path)
    files = {"test": f"{data_dir}/test.parquet"}
    try:
        ds = load_dataset("parquet", data_files=files)["test"]
    except Exception:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts = []
    for row in list(ds)[:n]:
        q = row["question"] if "question" in row else row["moves"]["question"]
        msgs = [{"role": "user", "content": q + SUFFIX}]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    return prompts


def start_server(model, port, flash, graph, mem_fraction, log):
    env = os.environ.copy()
    env["MESHY_SGLANG_SM70"] = "1"
    env["MESHY_SM70_FLASH_DECODE"] = "1" if flash else "0"
    env["MESHY_SM70_CUDA_GRAPH"] = "1" if graph else "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PATH"] = "/usr/local/cuda-12.4/bin:" + env.get("PATH", "")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from meshy.backend.sglang_sm70 import bootstrap_pythonpath, sm70_server_defaults

    boot = bootstrap_pythonpath()
    env["PYTHONPATH"] = (boot + os.pathsep + env["PYTHONPATH"]
                         if env.get("PYTHONPATH") else boot)
    args = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model, "--host", "127.0.0.1", "--port", str(port),
        "--tp-size", "1",
        "--mem-fraction-static", str(mem_fraction),
    ]
    for key, value in sm70_server_defaults().items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args += [flag, str(value)]
    lf = open(log, "ab", buffering=0)
    proc = subprocess.Popen(args, env=env, stdout=lf,
                            stderr=subprocess.STDOUT, start_new_session=True)
    return proc, lf


def generate(base, prompt, sampling, timeout=1800):
    r = requests.post(base + "/generate",
                      json={"text": prompt, "sampling_params": sampling},
                      timeout=timeout)
    r.raise_for_status()
    return r.json()


def consistency(model, data_dir, mem):
    """Greedy token agreement triton vs flash in eager and graph."""
    prompt = load_gsm8k_prompts(model, data_dir, 1)[0]
    sampling = {"temperature": 0.0, "max_new_tokens": 256}
    report = {}
    for graph in (False, True):
        texts = {}
        for flash in (False, True):
            tag = f"{'flash' if flash else 'triton'}-{'graph' if graph else 'eager'}"
            port = free_port()
            log = f"/data00/meshy/kern/logs/fdec_e2e_{tag}.log"
            proc, lf = start_server(model, port, flash, graph, mem, log)
            try:
                wait_ready(port, proc)
                time.sleep(3)  # settle plan/capture after first decode
                out = generate(f"http://127.0.0.1:{port}", prompt, sampling)
                texts[tag] = out.get("text", "")
            finally:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=120)
                lf.close()
                time.sleep(6)
        mode = "graph" if graph else "eager"
        tr, fl = texts[f"triton-{mode}"], texts[f"flash-{mode}"]
        report[mode] = {
            "exact_match": tr.strip() == fl.strip(),
            "triton_chars": len(tr), "flash_chars": len(fl),
        }
        print(json.dumps({"consistency": mode, **report[mode]}), flush=True)
    return report


def throughput(model, data_dir, mem, n_prompts, n_samples):
    """512 prompts x 8 samples, crossed order; wall + aggregate tok/s."""
    prompts = load_gsm8k_prompts(model, data_dir, n_prompts)
    sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                "max_new_tokens": 512}
    jobs = [(p, i) for p in prompts for i in range(n_samples)]
    order = [("triton", False), ("flash", True),
             ("flash", True), ("triton", False)]
    runs = []
    for label, flash in order:
        graph = True
        port = free_port()
        log = f"/data00/meshy/kern/logs/fdec_e2e_tp_{label}.log"
        proc, lf = start_server(model, port, flash, graph, mem, log)
        try:
            wait_ready(port, proc)
            # Warmup then the measured finite batch.
            generate(f"http://127.0.0.1:{port}", prompts[0],
                     {**sampling, "max_new_tokens": 8}, timeout=900)
            base = f"http://127.0.0.1:{port}"
            t0 = time.perf_counter()

            def one(job):
                p, _ = job
                r = generate(base, p, sampling, timeout=1800)
                ct = r.get("meta_info", {}).get("completion_tokens", 0)
                return ct if isinstance(ct, int) else len(ct)

            with ThreadPoolExecutor(max_workers=64) as pool:
                toks = list(pool.map(one, jobs))
            wall = time.perf_counter() - t0
            total = sum(toks)
            row = {"backend": label, "requests": len(jobs),
                   "wall_s": round(wall, 2),
                   "total_new_tokens": total,
                   "decode_tok_s": round(total / wall, 1)}
            runs.append(row)
            print(json.dumps(row), flush=True)
        finally:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=120)
            lf.close()
            time.sleep(6)
    # Cross-order average.
    for label in ("triton", "flash"):
        rs = [r for r in runs if r["backend"] == label]
        avg = sum(r["decode_tok_s"] for r in rs) / len(rs)
        print(json.dumps({"avg": label, "decode_tok_s": round(avg, 1)}),
              flush=True)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--mem", type=float, default=0.60)
    ap.add_argument("--prompts", type=int, default=512)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--out", default="/data00/meshy/kern/flash_decode_e2e.json")
    ap.add_argument("--only", choices=["consistency", "throughput", "all"],
                    default="all")
    args = ap.parse_args()
    os.makedirs("/data00/meshy/kern/logs", exist_ok=True)

    report = {}
    if args.only in ("consistency", "all"):
        report["consistency"] = consistency(args.model, args.data, args.mem)
    if args.only in ("throughput", "all"):
        report["throughput"] = throughput(
            args.model, args.data, args.mem, args.prompts, args.samples)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("FLASH_DECODE_E2E_DONE ->", args.out)


if __name__ == "__main__":
    main()
