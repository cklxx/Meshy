#!/usr/bin/env python3
"""GSM8K test-set holdout eval: greedy accuracy against a running SGLang server.

Matches the GRPO recipe's prompt exactly (same messages as
:class:`meshy.dataset.gsm8k.GSM8K`, tokenizer chat template, fp16 engine). By
default launches a single-card SGLang server itself; pass ``--base-url`` to
evaluate an already running server.

Examples::

    # V100, one command (starts and tears down its own server):
    python scripts/eval_gsm8k.py \\
        --model /data00/meshy/models/Qwen3-0.6B \\
        --data /data00/meshy/models/gsm8k --n 200

    # Against a server the recipe started:
    python scripts/eval_gsm8k.py --base-url http://127.0.0.1:30000 --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from datasets import load_dataset
from transformers import AutoTokenizer

from meshy.dataset.gsm8k import _extract_gsm8k_answer

SYSTEM_PROMPT = "You are a helpful assistant."
SUFFIX = ' Let\'s think step by step and output the final answer after "####".'


def build_prompts(model_path: str, rows, thinking: bool):
    tok = AutoTokenizer.from_pretrained(model_path)
    prompts, answers = [], []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"] + SUFFIX},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if not thinking:
            kwargs["enable_thinking"] = False
        prompts.append(tok.apply_chat_template(messages, **kwargs))
        answers.append(_extract_gsm8k_answer(row["answer"]))
    return prompts, answers


def generate(base_url: str, prompt: str, max_new_tokens: int) -> str:
    r = requests.post(
        f"{base_url}/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["text"]


def wait_healthy(base_url: str, timeout_s: int = 600) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"server at {base_url} not healthy after {timeout_s}s")


def start_server(model: str, port: int, mem_fraction: float, server_log: str) -> subprocess.Popen:
    # On sm70 (V100) SGLang 0.5.18 floors at sm75; Meshy's bootstrap patch
    # bypasses the gate, stubs sgl_kernel, forces native fused ops and pins
    # the CUDA 12.6 runtime. Harmless elsewhere (it no-ops off sm70).
    import meshy.backend.sglang_sm70 as sm70

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sm70.__file__))))
    bootstrap = os.path.join(repo_root, "meshy", "backend", "_sm70bootstrap")
    env = dict(os.environ, MESHY_SGLANG_SM70="1",
               PYTHONPATH=bootstrap + os.pathsep + repo_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--tp-size", "1",
        "--dtype", "float16",
        "--attention-backend", "triton",
        "--sampling-backend", "pytorch",
        "--cuda-graph-backend-decode", "disabled",
        "--cuda-graph-backend-prefill", "disabled",
        "--mem-fraction-static", str(mem_fraction),
        "--port", str(port),
        "--host", "127.0.0.1",
    ]
    logf = open(server_log, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    try:
        wait_healthy(f"http://127.0.0.1:{port}")
    except Exception:
        proc.terminate()
        raise
    return proc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data00/meshy/models/Qwen3-0.6B")
    ap.add_argument("--data", default="/data00/meshy/models/gsm8k",
                    help="local openai/gsm8k snapshot dir, or an HF dataset id")
    ap.add_argument("--base-url", default=None, help="use a running server instead of starting one")
    ap.add_argument("--port", type=int, default=30013)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mem-fraction", type=float, default=0.6)
    ap.add_argument("--no-thinking", action="store_true",
                    help="render the chat template with enable_thinking=False")
    ap.add_argument("--out", default=None, help="optional jsonl dump of prompt/response/prediction")
    ap.add_argument("--server-log", default="/data00/meshy/rl/logs/eval_server.log")
    args = ap.parse_args()

    ds = load_dataset(args.data, "main", split="test")
    rows = [ds[i] for i in range(min(args.n, len(ds)))]
    prompts, ground_truth = build_prompts(args.model, rows, thinking=not args.no_thinking)

    base_url = args.base_url
    proc = None
    if base_url is None:
        proc = start_server(args.model, args.port, args.mem_fraction, args.server_log)
        base_url = f"http://127.0.0.1:{args.port}"

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            responses = list(pool.map(lambda p: generate(base_url, p, args.max_new_tokens), prompts))
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()

    correct = 0
    fh = open(args.out, "w") if args.out else None
    for prompt, gt, text in zip(prompts, ground_truth, responses):
        pred = _extract_gsm8k_answer(text)
        ok = pred is not None and pred == gt
        correct += ok
        if fh:
            fh.write(json.dumps({"pred": pred, "gt": gt, "correct": ok, "response": text}, ensure_ascii=False) + "\n")
    if fh:
        fh.close()

    n = len(rows)
    print(f"accuracy: {correct}/{n} = {correct / n:.4f}")
    malformed = sum(_extract_gsm8k_answer(t) is None for t in responses)
    print(f"malformed (no #### answer): {malformed}/{n}")


if __name__ == "__main__":
    main()
