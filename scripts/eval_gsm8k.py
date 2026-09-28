#!/usr/bin/env python3
"""GSM8K test-set holdout eval for a running / self-launched SGLang server.

Two judging modes:

* **lenient** (default, the headline metric): answer is read from the
  post-thinking span, accepting, in order, ``\\boxed{N}``, ``#### N`` or the
  last number in the span. Sampling follows the Qwen3 thinking-mode recipe
  (temperature 0.6 / top_p 0.95 / top_k 20), 4 completions per question;
  per-question accuracy is the pass rate averaged over the 4 samples.
* **strict**: the GRPO reward rule (``#### N`` via
  :func:`meshy.dataset.gsm8k._extract_gsm8k_answer`); also reported as the
  "strict format rate".

Also reports truncation rate and completion-length p50/p90/p99 (output
tokens). By default launches a single-card SGLang server itself; pass
``--base-url`` to evaluate an already running server.

Examples::

    # V100 full test set, thinking-mode sampling, 4 samples/question:
    python scripts/eval_gsm8k.py \\
        --model /data00/meshy/models/Qwen3-0.6B \\
        --data /data00/meshy/models/gsm8k

    # Old greedy 200-question smoke against a running server:
    python scripts/eval_gsm8k.py --base-url http://127.0.0.1:30000 \\
        --greedy --samples 1 --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
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
_NUM = r"-?[\d,]+\.?\d*"


def build_prompts(model_path: str, rows, no_thinking: bool):
    tok = AutoTokenizer.from_pretrained(model_path)
    prompts = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"] + SUFFIX},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if no_thinking:
            kwargs["enable_thinking"] = False
        prompts.append(tok.apply_chat_template(messages, **kwargs))
    return prompts


def ground_truths(rows):
    return [_extract_gsm8k_answer(row["answer"]) for row in rows]


def answer_span(text: str) -> str:
    """Everything after the closing think tag; the whole text when absent."""
    idx = text.rfind("</think>")
    return text[idx + len("</think>"):] if idx >= 0 else text


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace(" ", "").rstrip("."))
    except ValueError:
        return None


def lenient_answer(text: str) -> float | None:
    """Extract the answer from the post-think span.

    Order: last \\boxed{N}, last `#### N`, last bare number. Currency symbols
    and trailing dots are tolerated; only the first numeric run inside a boxed
    expression is read (e.g. \\boxed{\\$18}, \\boxed{260\\text{ sheep}}).
    """
    span = answer_span(text)
    boxes = re.findall(r"\\boxed\s*\{([^{}]*)\}", span)
    if boxes:
        m = re.search(_NUM, boxes[-1])
        if m:
            return _to_float(m.group(0))
    strict = _extract_gsm8k_answer(span)
    if strict is not None:
        return strict
    nums = re.findall(_NUM, span)
    return _to_float(nums[-1]) if nums else None


def generate(base_url: str, prompt: str, sp: dict) -> dict:
    r = requests.post(
        f"{base_url}/generate",
        json={"text": prompt, "sampling_params": sp},
        timeout=1800,
    )
    r.raise_for_status()
    return r.json()


def wait_healthy(base_url: str, timeout_s: int = 900) -> None:
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
    # the CUDA 12.6 runtime. The bootstrap itself no-ops on non-sm70 hosts.
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


def _finish_type(meta: dict) -> str | None:
    reason = meta.get("finish_reason") or {}
    return reason.get("type") if isinstance(reason, dict) else None


def pct(values, q):
    if not values:
        return 0
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1] if len(values) > 1 else values[0]


def _selftest() -> None:
    cases = [
        ("<think>x</think>\n\\(\\boxed{\\$18}\\)", 18.0),
        ("<think>x</think>\n#### 260", 260.0),
        ("<think>x</think>\nSo the answer is 3.", 3.0),
        ("<think>x</think>\n#### 1,000 sheep", 1000.0),
        ("<think>x</think>\n\\boxed{260\\text{ sheep}}", 260.0),
        ("<think>x</think>\nno numbers here", None),
        ("<think>long 9 in reasoning #### 7</think>\nanswer #### 7", 7.0),
    ]
    for text, want in cases:
        got = lenient_answer(text)
        assert got == want, f"{text!r}: {got} != {want}"
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data00/meshy/models/Qwen3-0.6B")
    ap.add_argument("--data", default="/data00/meshy/models/gsm8k",
                    help="local openai/gsm8k snapshot dir, or an HF dataset id")
    ap.add_argument("--base-url", default=None, help="use a running server instead of starting one")
    ap.add_argument("--port", type=int, default=30013)
    ap.add_argument("--n", type=int, default=0, help="0 = full test set (1319)")
    ap.add_argument("--samples", type=int, default=4, help="completions per question")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--greedy", action="store_true", help="greedy decoding (temp 0), implies --samples 1")
    ap.add_argument("--no-thinking", action="store_true",
                    help="render the chat template with enable_thinking=False")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--mem-fraction", type=float, default=0.6)
    ap.add_argument("--out", default="/data00/meshy/rl/logs/gsm8k_eval.jsonl")
    ap.add_argument("--server-log", default="/data00/meshy/rl/logs/eval_server.log")
    ap.add_argument("--score-only", default=None,
                    help="recompute the summary from an existing eval jsonl, no server")
    args = ap.parse_args()

    if args.score_only:
        rows = [json.loads(l) for l in open(args.score_only)]
        n = max(r["q"] for r in rows) + 1
        samples = len(rows) // n
        per_q = {qi: [] for qi in range(n)}
        for r in rows:
            per_q[r["q"]].append(r["lenient"])
        lengths = [r["tokens"] for r in rows]
        acc = statistics.mean(sum(v) / len(v) for v in per_q.values())
        strict_ok = sum(r["strict"] for r in rows)
        strict_fmt = sum(r["pred_strict"] is not None for r in rows) / len(rows)
        truncated = sum(r["finish"] == "length" for r in rows)
        total = len(rows)
        print(f"[score-only {args.score_only}]")
        print(f"questions={n} samples/q={samples}")
        print(f"lenient accuracy (per-q mean): {acc:.4f}")
        print(f"strict accuracy: {strict_ok}/{total} = {strict_ok / total:.4f}")
        print(f"strict format rate: {strict_fmt:.4f}")
        print(f"truncation rate: {truncated}/{total} = {truncated / total:.4f}")
        print(f"completion tokens p50/p90/p95/p99: "
              f"{int(pct(lengths,50))}/{int(pct(lengths,90))}/{int(pct(lengths,95))}/{int(pct(lengths,99))}")
        print(f"tokens min/max/mean: {min(lengths)}/{max(lengths)}/{int(statistics.mean(lengths))}")
        return

    if args.greedy:
        args.samples = 1

    ds = load_dataset(args.data, "main", split="test")
    n = len(ds) if args.n == 0 else min(args.n, len(ds))
    rows = [ds[i] for i in range(n)]
    prompts = build_prompts(args.model, rows, no_thinking=args.no_thinking)
    gts = ground_truths(rows)

    sp = {"max_new_tokens": args.max_new_tokens}
    if args.greedy:
        sp["temperature"] = 0
    else:
        sp.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)

    base_url = args.base_url
    proc = None
    if base_url is None:
        proc = start_server(args.model, args.port, args.mem_fraction, args.server_log)
        base_url = f"http://127.0.0.1:{args.port}"

    jobs = [(qi, prompts[qi]) for qi in range(n) for _ in range(args.samples)]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            responses = list(pool.map(lambda j: (j[0], generate(base_url, j[1], sp)), jobs))
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()

    tok = AutoTokenizer.from_pretrained(args.model)
    per_q = {qi: [] for qi in range(n)}
    strict_ok = 0
    truncated = 0
    lengths = []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for qi, resp in responses:
            text = resp.get("text", "")
            meta = resp.get("meta_info") or {}
            finish = _finish_type(meta)
            toks = meta.get("output_token_length")
            if not isinstance(toks, int):
                toks = len(tok(text, add_special_tokens=False)["input_ids"])
            lengths.append(int(toks))
            if finish == "length":
                truncated += 1
            pred_l = lenient_answer(text)
            pred_s = _extract_gsm8k_answer(answer_span(text))
            ok_l = pred_l is not None and pred_l == gts[qi]
            ok_s = pred_s is not None and pred_s == gts[qi]
            strict_ok += ok_s
            per_q[qi].append(ok_l)
            fh.write(json.dumps({
                "q": qi, "gt": gts[qi], "lenient": ok_l, "strict": ok_s,
                "pred_lenient": pred_l, "pred_strict": pred_s,
                "tokens": int(toks), "finish": finish, "response": text,
            }, ensure_ascii=False) + "\n")

    total = n * args.samples
    acc = statistics.mean(sum(v) / len(v) for v in per_q.values())
    strict_fmt = sum(1 for _, resp in responses
                     if _extract_gsm8k_answer(answer_span(resp.get("text", ""))) is not None) / total
    print(f"questions={n} samples/q={args.samples} mode={'greedy' if args.greedy else 'sample'}")
    print(f"lenient accuracy (per-q mean): {acc:.4f}")
    print(f"strict accuracy: {strict_ok}/{total} = {strict_ok / total:.4f}")
    print(f"strict format rate: {strict_fmt:.4f}")
    print(f"truncation rate: {truncated}/{total} = {truncated / total:.4f}")
    print(f"completion tokens p50/p90/p99: {int(pct(lengths,50))}/{int(pct(lengths,90))}/{int(pct(lengths,99))}")
    print(f"tokens min/max/mean: {min(lengths)}/{max(lengths)}/{int(statistics.mean(lengths))}")


if __name__ == "__main__":
    main()
