#!/usr/bin/env python3
"""MATH-500 holdout eval with unbiased pass@k for a running/self-launched server.

Same launch/sampling plumbing as ``eval_gsm8k.py`` (server, sm70 bootstrap,
thinking-mode sampling); this script differs only in data and scoring:

* data: ``HuggingFaceH4/MATH-500`` (``--data math500``), a local snapshot dir,
  or an HF dataset id; columns ``problem`` / ``answer``;
* prompt asks for the final answer in ``\\boxed{}`` (``meshy.dataset.hendrycks_math``);
* scoring judges the **last** post-``</think>`` boxed expression against the
  gold answer with ``math_verify`` (SymPy), with a normalised-string fallback;
* output is the unbiased pass@k estimator of Chen et al. 2021
  (``meshy.utils.passatk``) for every k <= samples: 1,2,4,8,16,32.

Examples::

    # 500 problems x 32 samples against a live server:
    python scripts/eval_math.py --base-url http://127.0.0.1:30000 \\
        --data math500 --samples 32

    # quick greedy 200-question smoke, self-launched:
    python scripts/eval_math.py --model /data00/meshy/models/Qwen3-0.6B \\
        --data math500 --n 200 --samples 1

    # recompute pass@k from a saved jsonl without a server:
    python scripts/eval_math.py --score-only run.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_gsm8k as g  # reuse start_server / wait_healthy / generate helpers

from meshy.dataset.hendrycks_math import (
    SUFFIX,
    SYSTEM_PROMPT,
    extract_boxed,
    score_math_response,
    score_math_response_strict,
)
from meshy.utils.passatk import aggregate_pass_at_k

MATH500_ID = "HuggingFaceH4/MATH-500"
DEFAULT_KS = (1, 2, 4, 8, 16, 32)


def _load_split(spec):
    """Local snapshot: DatasetDict[test] or a single-split Dataset."""
    from datasets import load_from_disk
    try:
        d = load_from_disk(spec)
        return d["test"] if hasattr(d, "keys") else d
    except Exception:
        pass
    try:
        return load_dataset(spec, split="test")
    except Exception:
        return load_dataset(spec)["test"]


def load_rows(data: str):
    """Resolve --data: 'math500' alias, a local snapshot dir, or an HF id."""
    spec = MATH500_ID if data == "math500" else data
    return list(_load_split(spec))


def build_prompts(model_path: str, rows):
    tok = AutoTokenizer.from_pretrained(model_path)
    prompts, gts = [], []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["problem"] + SUFFIX},
        ]
        prompts.append(tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))
        gts.append(row["answer"])
    return prompts, gts


def _print_passk(label, agg):
    for k, st in sorted(agg.items()):
        print(f"{label} pass@{k}: {st['pass_at_k']:.4f}  "
              f"(over {st['problems']} problems, n>={st['samples_per_problem']})")


def _tally(rows, key):
    n = max(r["q"] for r in rows) + 1
    per_q = {i: [] for i in range(n)}
    for r in rows:
        per_q[r["q"]].append(r[key])
    return per_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data00/meshy/models/Qwen3-0.6B")
    ap.add_argument("--data", default="math500",
                    help="'math500', a local snapshot dir, or an HF dataset id")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--port", type=int, default=30014)
    ap.add_argument("--n", type=int, default=0, help="0 = all 500")
    ap.add_argument("--samples", type=int, default=32,
                    help="independent samples per question (pass@k needs >=k)")
    ap.add_argument("--max-new-tokens", type=int, default=8192,
                    help="match the GSM8K endpoint eval (8192); in-loop uses 4096")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--greedy", action="store_true",
                    help="temp 0; implies --samples 1")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--mem-fraction", type=float, default=0.6)
    ap.add_argument("--out", default="/data00/meshy/rl/logs/math500_eval.jsonl")
    ap.add_argument("--server-log", default="/data00/meshy/rl/logs/eval_math_server.log")
    ap.add_argument("--score-only", default=None)
    args = ap.parse_args()

    if args.score_only:
        rows = [json.loads(line) for line in open(args.score_only)]
        _print_passk("lenient", aggregate_pass_at_k(
            [(len(v), sum(v)) for v in _tally(rows, "lenient").values()]))
        _print_passk("strict", aggregate_pass_at_k(
            [(len(v), sum(v)) for v in _tally(rows, "strict").values()]))
        total = len(rows)
        fmt = sum(r.get("pred") is not None for r in rows) / total
        print(f"boxed format rate: {fmt:.4f}  questions={max(r['q'] for r in rows)+1}")
        return

    if args.greedy:
        args.samples = 1

    rows = load_rows(args.data)
    n = len(rows) if args.n == 0 else min(args.n, len(rows))
    rows = rows[:n]
    prompts, gts = build_prompts(args.model, rows)

    sp = {"max_new_tokens": args.max_new_tokens}
    if args.greedy:
        sp["temperature"] = 0
    else:
        sp.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)

    base_url = args.base_url
    proc = None
    if base_url is None:
        proc = g.start_server(args.model, args.port, args.mem_fraction, args.server_log)
        base_url = f"http://127.0.0.1:{args.port}"
    else:
        g.wait_healthy(base_url)

    jobs = [(qi, prompts[qi]) for qi in range(n)
            for _ in range(args.samples)]
    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            responses = list(pool.map(
                lambda j: (j[0], g.generate(base_url, j[1], sp)), jobs))
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()

    per_q_lenient = {qi: [] for qi in range(n)}
    per_q_strict = {qi: [] for qi in range(n)}
    trunc = lengths = fmt = 0
    total = n * args.samples
    # SGLang 0.5.18 on sm70 does not populate meta_info.output_token_length, so
    # fall back through the usage block and finally to a tokenizer count.
    counter_tok = AutoTokenizer.from_pretrained(args.model)

    def _response_tokens(meta: dict, text: str) -> int:
        usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
        for key in ("output_token_length", "completion_tokens", "output_tokens"):
            v = meta.get(key) or usage.get(key)
            if isinstance(v, (int, float)) and int(v) > 0:
                return int(v)
        return len(counter_tok.encode(text, add_special_tokens=False))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for qi, resp in responses:
            text = resp.get("text", "")
            pred = extract_boxed(text)
            ok_l = score_math_response(text, gts[qi])       # math_verify/SymPy
            ok_s = score_math_response_strict(text, gts[qi])  # exact normalised
            meta = resp.get("meta_info") or {}
            if g._finish_type(meta) == "length":
                trunc += 1
            toks = _response_tokens(meta, text)
            lengths += int(toks)
            fmt += pred is not None
            per_q_lenient[qi].append(int(ok_l))
            per_q_strict[qi].append(int(ok_s))
            fh.write(json.dumps({
                "q": qi, "gt": gts[qi],
                "lenient": bool(ok_l), "strict": bool(ok_s),
                "correct": bool(ok_l),  # back-compat alias for lenient
                "pred": pred, "tokens": int(toks),
                "finish": g._finish_type(meta), "response": text,
            }, ensure_ascii=False) + "\n")

    agg_l = aggregate_pass_at_k(
        [(len(v), sum(v)) for v in per_q_lenient.values()])
    agg_s = aggregate_pass_at_k(
        [(len(v), sum(v)) for v in per_q_strict.values()])
    print(f"data={args.data} questions={n} samples/q={args.samples} "
          f"mode={'greedy' if args.greedy else 'sample'} "
          f"max_new_tokens={args.max_new_tokens} split=test "
          f"elapsed={time.time()-started:.1f}s")
    _print_passk("lenient", agg_l)
    _print_passk("strict", agg_s)
    print(f"boxed format rate: {fmt}/{total} = {fmt/total:.4f}")
    print(f"truncation rate: {trunc}/{total} = {trunc/total:.4f}")
    print(f"mean output tokens: {lengths/total:.0f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
