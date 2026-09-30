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


def start_server(model, port, flash, graph, mem_fraction, log,
                 graph_max_bs=64, sentinel=None):
    env = os.environ.copy()
    env["MESHY_SGLANG_SM70"] = "1"
    env["MESHY_SM70_FLASH_DECODE"] = "1" if flash else "0"
    env["MESHY_SM70_CUDA_GRAPH"] = "1" if graph else "0"
    env["MESHY_SM70_CUDA_GRAPH_MAX_BS"] = str(graph_max_bs)
    if sentinel:
        env["MESHY_FLASH_SENTINEL"] = sentinel
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
    for key, value in sm70_server_defaults(max_bs=graph_max_bs).items():
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


def stop_server(proc, lf):
    """Terminate by exact PID, then poll until GPU memory drains."""
    import subprocess as sp
    pid = proc.pid
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=120)
    except Exception:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    lf.close()
    # Wait until this server leaves no GPU compute process.
    for _ in range(40):
        out = sp.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader"], capture_output=True, text=True)
        pids = {int(x) for x in out.stdout.split() if x.isdigit()}
        if pid not in pids:
            break
        time.sleep(3)
    time.sleep(4)


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
        evidence = {}
        for flash in (False, True):
            tag = f"{'flash' if flash else 'triton'}-{'graph' if graph else 'eager'}"
            port = free_port()
            log = f"/data00/meshy/kern/logs/fdec_e2e_{tag}.log"
            sentinel = f"/tmp/fdec_sent_{tag}.txt"
            try:
                os.remove(sentinel)
            except FileNotFoundError:
                pass
            proc, lf = start_server(model, port, flash, graph, mem, log,
                                    sentinel=sentinel)
            try:
                wait_ready(port, proc)
                time.sleep(3)  # settle plan/capture after first decode
                out = generate(f"http://127.0.0.1:{port}", prompt, sampling)
                texts[tag] = out.get("text", "")
            finally:
                stop_server(proc, lf)
            if flash:
                with open(sentinel) as f:
                    sent = f.read()
                installed = "flash-decode attention installed" in open(
                    log, errors="ignore").read()
                evidence[tag] = {
                    "sentinel_apply_flash": "apply-flash=True" in sent,
                    "log_installed": installed,
                    "sentinel_flash_called": "flash-called" in sent,
                }
        mode = "graph" if graph else "eager"
        tr, fl = texts[f"triton-{mode}"], texts[f"flash-{mode}"]
        report[mode] = {
            "exact_match": tr.strip() == fl.strip(),
            "triton_chars": len(tr), "flash_chars": len(fl),
            "flash_evidence": evidence[f"flash-{mode}"],
        }
        print(json.dumps({"consistency": mode, **report[mode]}), flush=True)
    return report


MATH_SUFFIX = " Let's think step by step and put the final answer in \\boxed{}."


def load_math_prompts(model_path, n):
    """First n rows of the concatenated Hendrycks MATH train split,
    formatted exactly like meshy.dataset.hendrycks_math."""
    from transformers import AutoTokenizer
    from meshy.dataset.hendrycks_math import _load_all_subjects

    tok = AutoTokenizer.from_pretrained(model_path)
    ds = _load_all_subjects("train")
    prompts = []
    for row in list(ds)[:n]:
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": row["problem"] + MATH_SUFFIX},
        ]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    return prompts


def _decode_stats(log_path):
    """Parse SGLang 'Decode batch' lines: gen throughput token/s plus
    mean #running-req and #token over the saturated window."""
    import re
    import statistics

    gen, run, tok = [], [], []
    pat = re.compile(
        r"Decode batch, #running-req: (\d+), #token: (\d+).*?gen throughput "
        r"\(token/s\): ([\d.]+)")
    try:
        with open(log_path, "rb") as f:
            for line in f:
                m = pat.search(line.decode("utf-8", "ignore"))
                if m:
                    run.append(int(m.group(1)))
                    tok.append(int(m.group(2)))
                    gen.append(float(m.group(3)))
    except FileNotFoundError:
        pass
    if not gen:
        return None
    # Drop the first ~10 warmup/ramp lines so the window reflects the
    # saturated long-context regime.
    g = gen[10:] or gen
    return {
        "gen_tok_s_mean": round(statistics.mean(g), 1),
        "gen_tok_s_median": round(statistics.median(g), 1),
        "gen_tok_s_p90": round(sorted(g)[int(0.9 * (len(g) - 1))], 1),
        "mean_running_req": round(statistics.mean(run), 1),
        "mean_token": round(statistics.mean(tok), 1),
        "decode_samples": len(g),
    }


def throughput(model, mem, n_prompts, n_samples, inflight, window_s,
               graph_max_bs, order=("triton", "flash"), dataset="math"):
    """Rollout-shaped load: n_prompts*n_samples queued jobs, at most
    ``inflight`` concurrent (rl ROLLOUT_BATCH*GROUP_SIZE=64). Runs for
    ``window_s`` in the saturated regime, then truncates. Throughput
    comes from the server's own per-step 'Decode batch' gen throughput.

    dataset="math" uses Hendrycks MATH train; "gsm8k" uses GSM8K train.
    Both production recipes (grpo_*_v100) roll out with
    temp 1.0/top_p 1.0/top_k -1/max_new 4096, so sampling is identical;
    only the prompt source differs."""
    if dataset == "math":
        prompts = load_math_prompts(model_path=model, n=n_prompts)
        log_tag, sent_tag = "math", "math"
    else:
        prompts = load_gsm8k_prompts(model, DATA_DEFAULT, n_prompts)
        log_tag, sent_tag = f"gsm{n_prompts}", f"gsm{n_prompts}"
    sampling = {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                "max_new_tokens": 4096}
    jobs = [p for p in prompts for _ in range(n_samples)]
    runs = []
    for label in order:
        flash = label == "flash"
        port = free_port()
        log = f"/data00/meshy/kern/logs/fdec_{log_tag}_{label}.log"
        sentinel = f"/tmp/fdec_sent_{sent_tag}_{label}.txt"
        try:
            os.remove(sentinel)
        except FileNotFoundError:
            pass
        proc, lf = start_server(model, port, flash, True, mem, log,
                                graph_max_bs=graph_max_bs, sentinel=sentinel)
        try:
            wait_ready(port, proc)
            base = f"http://127.0.0.1:{port}"
            generate(base, prompts[0],
                     {**sampling, "max_new_tokens": 8}, timeout=900)

            def one(p):
                try:
                    generate(base, p, sampling, timeout=window_s + 1200)
                except Exception:
                    return

            pool = ThreadPoolExecutor(max_workers=inflight)
            pool.map(one, jobs)
            # Saturated window; requests run to EOS/4096 in background.
            time.sleep(window_s)
            stats = _decode_stats(log) or {}
            try:
                sent = open(sentinel).read()
            except FileNotFoundError:
                sent = ""
            row = {"backend": label, "dataset": dataset,
                   "inflight": inflight,
                   "kv_layout": "nhd3-token-major",
                   "window_s": window_s, "queued": len(jobs),
                   "sentinel_apply_flash": "apply-flash=True" in sent,
                   "sentinel_flash_called": "flash-called" in sent,
                   "log_installed": (
                       "flash-decode attention installed"
                       in open(log, errors="ignore").read()),
                   **stats}
            runs.append(row)
            print(json.dumps(row), flush=True)
            pool.shutdown(wait=False, cancel_futures=True)
        finally:
            stop_server(proc, lf)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--mem", type=float, default=0.60)
    ap.add_argument("--prompts", type=int, default=512)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--inflight", type=int, default=64,
                    help="concurrent requests; rl in-flight = 8*8 = 64")
    ap.add_argument("--window", type=int, default=600,
                    help="saturated decode window seconds per backend")
    ap.add_argument("--graph-max-bs", type=int, default=64,
                    help="matches rl MESHY_SM70_CUDA_GRAPH_MAX_BS")
    ap.add_argument("--order", default="triton,flash")
    ap.add_argument("--dataset", choices=["math", "gsm8k"], default="math")
    ap.add_argument("--out", default="/data00/meshy/kern/flash_decode_e2e.json")
    ap.add_argument("--only", choices=["consistency", "throughput", "all"],
                    default="all")
    args = ap.parse_args()
    os.makedirs("/data00/meshy/kern/logs", exist_ok=True)

    report = {}
    if args.only in ("consistency", "all"):
        report["consistency"] = consistency(args.model, args.data, args.mem)
    if args.only in ("throughput", "all"):
        report[f"throughput_{args.dataset}"] = throughput(
            args.model, args.mem, args.prompts, args.samples,
            args.inflight, args.window, args.graph_max_bs,
            tuple(args.order.split(",")), dataset=args.dataset)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("FLASH_DECODE_E2E_DONE ->", args.out)


if __name__ == "__main__":
    main()
