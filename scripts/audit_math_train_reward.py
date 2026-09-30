#!/usr/bin/env python3
"""T11b: manual audit of boxed extraction + math_verify grading on the
Hendrycks MATH train split, 20 rows spread over all seven subjects and the
boundary answer shapes (fractions, negatives, radicals, units, multiple /
missing boxed, intervals).

CPU only. Run on the box (math_verify installed):

    PYTHONPATH=. python scripts/audit_math_train_reward.py \
        --train /data00/meshy/models/hendrycks_math_train

For every chosen row it prints:
* prompt id (global train index + subject/level);
* the gold boxed answer as it appears at the solution tail;
* the extractor's value;
* whether a verbatim-boxed model response scores 1;
* "probe" answers: an equivalent-form, a wrong form, a no-boxed response;
each must match the expected verdict (equiv -> 1, wrong/missing -> 0).

Any mismatch is printed with FAIL and exits non-zero; full per-row detail is
written to --out JSON.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from datasets import load_from_disk

from meshy.dataset.hendrycks_math import (
    extract_boxed,
    math_equiv,
    score_math_response,
)

# boundary shape detectors over the gold string
SHAPE_RULES = [
    ("negative", lambda g: g.lstrip().startswith("-")),
    ("fraction", lambda g: "\\frac" in g or "\\dfrac" in g),
    ("radical", lambda g: "\\sqrt" in g),
    ("unit_text", lambda g: "\\text{" in g),
    ("interval", lambda g: bool(re.search(r"[(\[\{][^)\]}]+,\s*[^)\]}]+[)\]\}]", g))),
    ("set_list", lambda g: g.startswith("\\{") or g.startswith("(")),
    ("pi_or_e", lambda g: ("\\pi" in g or g.strip() in ("e",)) ),
    ("decimal", lambda g: bool(re.match(r"^-?\d+\.\d+$", g.strip()))),
]


def count_boxed(sol: str) -> int:
    return len(re.findall(r"\\boxed\b", sol))


def equivalent_probe(gold: str) -> str | None:
    """A deliberately different-looking but mathematically equivalent answer.

    Best-effort, per shape; None when we cannot cheaply construct one (that
    row then only checks verbatim + wrong, not a variant).
    """
    g = gold.strip()
    m = re.fullmatch(r"\\(?:d?frac)\{(-?\d+)\}\{(\d+)\}", g)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a % b == 0:
            return str(a // b)  # \frac{4}{2} -> 2
        return f"{a}/{b}"          # \frac{1}{2} -> 1/2
    if re.fullmatch(r"-?\d+", g):
        return g  # integer: identical is fine, no variant needed
    if g == r"\frac{1}{2}":
        return "0.5"
    return None


def wrong_probe(gold: str) -> str:
    """A guaranteed-different token answer (used to assert rejection)."""
    g = gold.strip()
    if re.fullmatch(r"-?\d+", g):
        return str(int(g) + 1)
    # prepend a harmless symbol so the token string differs; for symbolic
    # answers wrap a different integer.
    m = re.search(r"-?\d+", g)
    if m:
        n = int(m.group(0)) + 7
        return g[:m.start()] + str(n) + g[m.end():]
    return "0" if g != "0" else "1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/data00/meshy/models/hendrycks_math_train")
    ap.add_argument("--total", type=int, default=20)
    ap.add_argument("--out", default="/data00/meshy/spec/t11b_audit.json")
    args = ap.parse_args()

    d = load_from_disk(args.train)
    # Weight each subject by its size so the quota sums to --total:
    # largest subjects get 3 rows, smallest get 2.
    sizes = {s: len([i for i, r in enumerate(d) if r["type"] == s])
             for s in {r["type"] for r in d}}
    raw = {s: args.total * n / len(d) for s, n in sizes.items()}
    quota = {s: max(1, round(v)) for s, v in raw.items()}
    # adjust the largest subjects so the rounded quota sums exactly to total
    diff = args.total - sum(quota.values())
    for s in sorted(sizes, key=lambda x: -sizes[x]):
        if diff == 0:
            break
        quota[s] = max(1, quota[s] + (1 if diff > 0 else -1))
        diff += -1 if diff > 0 else 1

    rows: list[dict[str, Any]] = []
    by_subj: dict[str, list[int]] = {}
    for i, r in enumerate(d):
        by_subj.setdefault(r["type"], []).append(i)

    for subj in sorted(by_subj):
        idxs = by_subj[subj]
        want = quota.get(subj, 1)
        boundary = []
        for i in idxs:
            g = extract_boxed(d[i]["solution"])
            if not g:
                continue
            shapes = [name for name, fn in SHAPE_RULES if fn(g)]
            if count_boxed(d[i]["solution"]) >= 2 or shapes:
                boundary.append((i, shapes))
        step = max(1, len(boundary) // max(1, want))
        picks = [boundary[j][0] for j in range(0, len(boundary),
                                               max(1, step))][:want]
        if len(picks) < want:  # fill with ordinary rows
            fill_step = max(1, len(idxs) // want)
            for j in range(0, len(idxs), max(1, fill_step)):
                if idxs[j] not in picks and extract_boxed(d[j]["solution"]):
                    picks.append(idxs[j])
                if len(picks) >= want:
                    break
        rows.extend(picks[:want])

    rows = rows[:args.total]
    failures = []
    detail = []
    for rank, i in enumerate(rows):
        r = d[i]
        sol = r["solution"]
        gold_boxed = extract_boxed(sol)          # the training ground truth
        n_box = count_boxed(sol)
        shapes = [name for name, fn in SHAPE_RULES if gold_boxed and fn(gold_boxed)]

        checks = {}
        # 1. a model that echoes the gold boxed answer, post-think
        resp_ok = "</think>\n" + r"\boxed{" + gold_boxed + "}"
        checks["verbatim"] = score_math_response(resp_ok, gold_boxed)

        # 2. an equivalent-looking variant (fraction/decimal/int)
        probe = equivalent_probe(gold_boxed)
        if probe is not None:
            checks["equivalent_variant"] = math_equiv(probe, gold_boxed)

        # 3. a wrong answer must be rejected
        bad = wrong_probe(gold_boxed)
        checks["wrong_rejected"] = (bad == gold_boxed) or (
            not math_equiv(bad, gold_boxed))

        # 4. multiple boxed: only the last counts; put a wrong one then gold
        resp_multi = ("</think>\n" + r"\boxed{" + bad + "}"
                      " ... " + r"\boxed{" + gold_boxed + "}")
        checks["multi_boxed_takes_last"] = score_math_response(resp_multi, gold_boxed)
        # and gold-then-wrong must NOT count (last is wrong)
        resp_multi_bad = ("</think>\n" + r"\boxed{" + gold_boxed + "}"
                          " ... " + r"\boxed{" + bad + "}")
        checks["multi_boxed_last_wrong_rejected"] = not score_math_response(
            resp_multi_bad, gold_boxed)

        # 5. no boxed answer after </think> -> 0
        checks["no_boxed_rejected"] = not score_math_response(
            "</think>\nThe answer is something else.", gold_boxed)

        row_fail = [k for k, ok in checks.items() if not ok]
        detail.append({
            "order": rank,
            "prompt_id": i,
            "subject": r["type"],
            "level": r["level"],
            "n_boxed_in_solution": n_box,
            "shapes": shapes,
            "gold_extracted": gold_boxed,
            "equivalent_probe": probe,
            "wrong_probe": bad if bad != gold_boxed else None,
            "checks": checks,
            "fail": row_fail,
            "problem_head": r["problem"][:120],
            "solution_tail": sol[-160:],
        })
        if row_fail:
            failures.append((i, r["type"], gold_boxed, row_fail))

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(detail, fh, ensure_ascii=False, indent=2)

    for rec in detail:
        flag = "OK " if not rec["fail"] else "FAIL"
        print(f"[{flag}] id={rec['prompt_id']:>4} {rec['subject']:<24} "
              f"{rec['level']} shapes={','.join(rec['shapes']) or '-':<14} "
              f"nbox={rec['n_boxed_in_solution']} gold={rec['gold_extracted']!r}")
        if rec["equivalent_probe"] is not None:
            print(f"        variant {rec['equivalent_probe']!r} -> "
                  f"equiv={rec['checks'].get('equivalent_variant')}")
    print(f"\nrows={len(rows)} failures={len(failures)} -> {args.out}")
    if failures:
        for i, subj, gold, fs in failures:
            print(f"FAIL id={i} {subj} gold={gold!r} checks={fs}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
