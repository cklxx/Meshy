"""In-loop holdout eval + checkpoint pruning for the V100 GRPO run.

Wired as the rollout worker's ``version_hook``, so every call happens at the
same point: the colocated engine has just been granted the GPU and loaded
weight ``v{version}``, and the trainer is blocked waiting for this window's
data. No release/abort can interrupt the eval, and every request is served
under the exact version being scored.

Holdout dataset is selected with ``XRL_EVAL_DATASET``:

* ``gsm8k`` (default) — local openai/gsm8k snapshot, ``#### N`` numeric scoring;
* ``math500`` — local/HF ``HuggingFaceH4/MATH-500`` snapshot, ``\\boxed{}`` final
  answers scored by ``math_verify`` (SymPy) with a normalised-string fallback.

At ``eval every`` versions it runs a 200-question x1 holdout (Qwen3
thinking-mode sampling; XRL_EVAL_N / XRL_EVAL_MAX_NEW override n and the
4096-token cap), copies that version to the 3FS milestone dir, and prunes
local weight versions older than keep=3. Set XRL_EVAL_N smaller than 200 /
XRL_EVAL_MAX_NEW for quick checks. The final v{steps} version never gets a
rollout window (the bounded dataset is exhausted first), so the multi-sample
endpoint eval (``scripts/eval_gsm8k.py`` / ``scripts/eval_math.py``) runs
standalone after the run.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import statistics
import time

from datasets import load_dataset
from transformers import AutoTokenizer

from meshy.dataset.gsm8k import _extract_gsm8k_answer, answer_span, lenient_gsm8k_answer
from meshy.dataset.hendrycks_math import (
    SUFFIX as MATH_SUFFIX,
    SYSTEM_PROMPT as MATH_SYSTEM_PROMPT,
    score_math_response,
)

SYSTEM_PROMPT = "You are a helpful assistant."
SUFFIX = ' Let\'s think step by step and output the final answer after "####".'

# Holdout dataset switch (XRL_EVAL_DATASET):
#   gsm8k  -> local openai/gsm8k snapshot, #### numeric scoring (default)
#   math500-> local/HF MATH-500 snapshot, \boxed{} math_verify scoring
EVAL_DATASET = os.environ.get("XRL_EVAL_DATASET", "gsm8k").lower()
GSM8K_DEFAULT_DIR = "/data00/meshy/models/gsm8k"
MATH500_DEFAULT_DIR = "/data00/meshy/models/MATH-500"
MATH500_HF_ID = "HuggingFaceH4/MATH-500"

EVAL_EVERY = int(os.environ.get("XRL_EVAL_EVERY", "50"))
# Cumulative-window alignment. version N of this run == cumulative window
# N+EVAL_OFFSET. Evals fire on cumulative windows EVAL_FIRST_CUM,
# FIRST+EVERY, ... . Expressing the gate on the *cumulative* window (rather
# than on version) is what lets the first eval version be below the offset
# (e.g. cont24: offset 16, first cumulative 18 -> eval at version 2,8,...).
# The offset is normally derived from the single XRL_START_WINDOW knob (0 on a
# DCP resume, since restored versions are already absolute); an explicit
# XRL_EVAL_OFFSET still wins for backward compatibility.
from recipe.v100_windows import resolve_eval_offset

EVAL_OFFSET = int(
    os.environ.get("XRL_EVAL_OFFSET", str(resolve_eval_offset()))
)
EVAL_FIRST_CUM = int(os.environ.get("XRL_EVAL_FIRST_CUM", str(EVAL_OFFSET)))
EVAL_N = int(os.environ.get("XRL_EVAL_N", "200"))
KEEP_VERSIONS = int(os.environ.get("XRL_KEEP_VERSIONS", "3"))
MAX_NEW_TOKENS = int(os.environ.get("XRL_EVAL_MAX_NEW", "4096"))
EVAL_CONCURRENCY = int(os.environ.get("XRL_EVAL_CONCURRENCY", "64"))


def _is_eval_version(version: int) -> bool:
    cum = version + EVAL_OFFSET
    return cum >= EVAL_FIRST_CUM and (cum - EVAL_FIRST_CUM) % EVAL_EVERY == 0

_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": MAX_NEW_TOKENS}


def _runtime_root() -> str:
    root = os.environ.get("XRL_RUNTIME_DIR")
    if not root:
        raise RuntimeError("version hook needs XRL_RUNTIME_DIR")
    return root


def _weights_dir(root: str) -> str:
    # TitanEngine._weights_path -> runtime.checkpoint_path("actor_train-0", v)
    return os.path.join(root, "weights", "actor_train-0")


def _milestone_dir(root: str) -> str:
    return os.path.join(os.environ.get("XRL_CKPT_DIR", "/data00/meshy/store/ckpt"),
                        os.path.basename(root.rstrip("/")))


def _render_ids(tok, messages: list[dict]) -> list[int]:
    # Some transformers return a BatchEncoding from apply_chat_template; list()
    # of that yields the dict keys, not token ids (the 400-on-every-request
    # bug). Normalize exactly like meshy.utils.sample.SampleBuilder._render.
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return list(out["input_ids"] if hasattr(out, "keys") else out)


def _build_prompts(model_path: str, rows):
    tok = AutoTokenizer.from_pretrained(model_path)
    prompts, gts = [], []
    if EVAL_DATASET == "math500":
        qcol, acol, sys_msg, suffix = "problem", "answer", MATH_SYSTEM_PROMPT, MATH_SUFFIX
    else:
        qcol, acol, sys_msg, suffix = "question", "answer", SYSTEM_PROMPT, SUFFIX
    for row in rows:
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": row[qcol] + suffix},
        ]
        prompts.append(_render_ids(tok, messages))
        gts.append(row[acol] if EVAL_DATASET == "math500"
                   else _extract_gsm8k_answer(row[acol]))
    return prompts, gts, tok


def _load_split(data_dir, name=None, split="test"):
    """Local snapshot: DatasetDict[split] or a single-split Dataset."""
    from datasets import load_from_disk
    try:
        d = load_from_disk(data_dir)
        return d[split] if hasattr(d, "keys") else d
    except Exception:
        return load_dataset(data_dir, name, split=split)


def _load_eval_rows():
    data_dir = os.environ.get(
        "XRL_MATH500_DIR" if EVAL_DATASET == "math500" else "XRL_GSM8K_DIR",
        MATH500_DEFAULT_DIR if EVAL_DATASET == "math500" else GSM8K_DEFAULT_DIR,
    )
    name = None if EVAL_DATASET == "math500" else "main"
    try:
        ds = _load_split(data_dir, name)
    except Exception:
        # XRL_*_DIR unset and no local snapshot: fall back to the HF id.
        ds = load_dataset(MATH500_HF_ID, split="test")
    return list(ds)[:EVAL_N]


async def _holdout_eval(engine, model_path: str, root: str, version: int) -> dict:
    rows = await asyncio.to_thread(_load_eval_rows)
    prompts, gts, tok = await asyncio.to_thread(_build_prompts, model_path, rows)

    sem = asyncio.Semaphore(EVAL_CONCURRENCY)

    async def one(qi: int, input_ids: list[int]):
        async with sem:
            gen = await engine.generate(input_ids, sampling_params=_SAMPLING)
        text = await asyncio.to_thread(tok.decode, gen.tokens)
        return qi, text, len(gen.tokens), gen.finish_reason

    started = time.time()
    results = await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
    elapsed = time.time() - started

    per_q = {}
    ok_total = trunc = 0
    lengths = []
    fmt_total = 0
    out_path = os.path.join(root, "eval", f"step{version}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        for qi, text, toks, finish in results:
            if EVAL_DATASET == "math500":
                from meshy.dataset.hendrycks_math import extract_boxed
                pred = extract_boxed(text)
                ok = score_math_response(text, gts[qi])
                pred_l = pred_s = pred
                ok_l = ok_s = ok
                fmt_total += pred is not None
            else:
                pred_l = lenient_gsm8k_answer(text)
                pred_s = _extract_gsm8k_answer(answer_span(text))
                ok_l = pred_l is not None and pred_l == gts[qi]
                ok_s = pred_s is not None and pred_s == gts[qi]
                fmt_total += pred_s is not None
            ok_total += ok_l
            trunc += finish == "length"
            lengths.append(toks)
            per_q.setdefault(qi, []).append(ok_l)
            fh.write(json.dumps({
                "q": qi, "gt": gts[qi], "correct": ok_l,
                "lenient": ok_l, "strict": ok_s,
                "pred": (pred_l if EVAL_DATASET == "math500" else pred_s),
                "tokens": toks, "finish": finish,
            }, ensure_ascii=False) + "\n")

    acc = statistics.mean(sum(v) / len(v) for v in per_q.values())
    p95 = statistics.quantiles(lengths, n=100)[94] if len(lengths) > 1 else lengths[0]
    summary = {
        "version": version, "dataset": EVAL_DATASET, "n": EVAL_N,
        "accuracy": acc,
        "answer_format_rate": fmt_total / EVAL_N,
        "trunc": trunc / EVAL_N,
        "p50": int(statistics.quantiles(lengths, n=100)[49]),
        "p95": int(p95), "mean_tok": int(statistics.mean(lengths)),
        "elapsed_s": round(elapsed, 1),
    }
    with open(os.path.join(root, "eval", "summary.jsonl"), "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    return summary


def _prune(root: str, version: int) -> list[str]:
    """Delete v{k} for k <= version-KEEP.

    Safe because the hook runs only after SGLang loaded v{version} (the grant
    callback awaits load_weights before waking the rollout), so anything older
    than keep=3 can never be needed by rollback/resume. Eval milestones are
    separate copies under ``$XRL_CKPT_DIR/<run>/step{n}`` and are untouched.
    """
    wdir = _weights_dir(root)
    if not os.path.isdir(wdir):
        return []
    floor = version - KEEP_VERSIONS + 1
    removed = []
    for name in os.listdir(wdir):
        if not (name.startswith("v") and name[1:].isdigit()):
            continue
        k = int(name[1:])
        if k < floor:
            shutil.rmtree(os.path.join(wdir, name))
            removed.append(name)
    return removed


async def version_hook(*, version: int, engine, model_path: str, **_: object) -> None:
    """200x1 holdout + milestone copy + old-version prune, once per version."""
    root = _runtime_root()

    if _is_eval_version(version):
        summary = await _holdout_eval(engine, model_path, root, version)
        print(f"[inloop-eval v{version}] {json.dumps(summary)}", flush=True)
        src = os.path.join(_weights_dir(root), f"v{version}")
        if os.path.isdir(src):
            dst = os.path.join(_milestone_dir(root), f"step{version}")
            if not os.path.isdir(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copytree(src, dst)

    removed = await asyncio.to_thread(_prune, root, version)
    if removed:
        print(f"[inloop-prune v{version}] removed {sorted(removed)}", flush=True)
