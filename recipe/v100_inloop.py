"""In-loop holdout eval + checkpoint pruning for the V100 GSM8K GRPO run.

Wired as the rollout worker's ``version_hook``, so every call happens at the
same point: the colocated engine has just been granted the GPU and loaded
weight ``v{version}``, and the trainer is blocked waiting for this window's
data. No release/abort can interrupt the eval, and every request is served
under the exact version being scored.

At ``eval every`` versions it runs a 200-question x1 holdout (Qwen3
thinking-mode sampling, lenient scoring), copies that version to the 3FS
milestone dir, and prunes local weight versions older than keep=3. The final
v{steps} version never gets a rollout window (the bounded dataset is exhausted
first), so the 200x4 endpoint eval runs standalone after the run.
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

SYSTEM_PROMPT = "You are a helpful assistant."
SUFFIX = ' Let\'s think step by step and output the final answer after "####".'

EVAL_EVERY = int(os.environ.get("XRL_EVAL_EVERY", "50"))
# Cumulative-window alignment. version N of this run == cumulative window
# N+EVAL_OFFSET. Evals fire on cumulative windows EVAL_FIRST_CUM,
# FIRST+EVERY, ... . Expressing the gate on the *cumulative* window (rather
# than on version) is what lets the first eval version be below the offset
# (e.g. cont24: offset 16, first cumulative 18 -> eval at version 2,8,...).
EVAL_OFFSET = int(os.environ.get("XRL_EVAL_OFFSET", "0"))
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
    return os.path.join(os.environ.get("XRL_CKPT_DIR", "/3fs/stage/meshy/ckpt"),
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
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"] + SUFFIX},
        ]
        prompts.append(_render_ids(tok, messages))
        gts.append(_extract_gsm8k_answer(row["answer"]))
    return prompts, gts, tok


async def _holdout_eval(engine, model_path: str, root: str, version: int) -> dict:
    data_dir = os.environ.get("XRL_GSM8K_DIR", "/data00/meshy/models/gsm8k")
    rows = list(load_dataset(data_dir, "main", split="test"))[:EVAL_N]
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
    strict_ok = trunc = 0
    lengths, strict_fmt = [], 0
    out_path = os.path.join(root, "eval", f"step{version}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        for qi, text, toks, finish in results:
            pred_l = lenient_gsm8k_answer(text)
            pred_s = _extract_gsm8k_answer(answer_span(text))
            ok_l = pred_l is not None and pred_l == gts[qi]
            ok_s = pred_s is not None and pred_s == gts[qi]
            strict_ok += ok_s
            strict_fmt += pred_s is not None
            trunc += finish == "length"
            lengths.append(toks)
            per_q.setdefault(qi, []).append(ok_l)
            fh.write(json.dumps({
                "q": qi, "gt": gts[qi], "lenient": ok_l, "strict": ok_s,
                "pred_lenient": pred_l, "pred_strict": pred_s,
                "tokens": toks, "finish": finish,
            }, ensure_ascii=False) + "\n")

    acc = statistics.mean(sum(v) / len(v) for v in per_q.values())
    p95 = statistics.quantiles(lengths, n=100)[94] if len(lengths) > 1 else lengths[0]
    summary = {
        "version": version, "n": EVAL_N, "lenient": acc,
        "strict": strict_ok / EVAL_N, "strict_fmt": strict_fmt / EVAL_N,
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
