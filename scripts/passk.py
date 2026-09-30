#!/usr/bin/env python3
"""Pass@k (Chen et al. 2021 unbiased estimator) from an eval jsonl.

Each row carries ``q`` (question id) and ``lenient``/``strict`` correctness.
With n samples per question and c correct:

    pass@k = 1 - C(n-c, k) / C(n, k)

computed in log/comb space to stay exact. We report both the combinatorial
unbiased estimator over all n=32 draws (pass@32 is then exactly 1 for any
c>=1) -- this is the standard HumanEval-style pass@k, where pass@1 equals the
mean single-sample accuracy.
"""

import argparse
import json
import math
from collections import defaultdict


def _comb_le_one(a: int, b: int) -> float:
    """C(a, b) for 0<=b<=a, returned via lgamma (a,b can be large-ish)."""
    if b < 0 or b > a:
        return 0.0
    if b == 0 or b == a:
        return 1.0
    return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.exp(_comb_le_one(n - c, k) - _comb_le_one(n, k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--ks", default="1,2,4,8,16,32")
    ap.add_argument("--field", default="lenient", choices=["lenient", "strict"])
    args = ap.parse_args()

    per_q = defaultdict(list)
    with open(args.jsonl) as fh:
        for line in fh:
            r = json.loads(line)
            per_q[r["q"]].append(1 if r.get(args.field) else 0)

    ks = [int(x) for x in args.ks.split(",")]
    ns = {len(v) for v in per_q.values()}
    print(f"questions={len(per_q)} samples_per_q={sorted(ns)} field={args.field}")
    print(f"{'k':>4} {'pass@k':>8}")
    for k in ks:
        vals = []
        for q, v in per_q.items():
            n = len(v)
            c = sum(v)
            vals.append(pass_at_k(n, c, k))
        print(f"{k:>4} {sum(vals)/len(vals):>8.4f}")
    # single-sample accuracy reference (=pass@1)
    allc = sum(sum(v) for v in per_q.values())
    alln = sum(len(v) for v in per_q.values())
    print(f"raw_correct={allc}/{alln} single_acc={allc/alln:.4f}")
    # questions with zero correct (capability ceiling at n=32)
    zero = sum(1 for v in per_q.values() if sum(v) == 0)
    print(f"never_correct_questions={zero}/{len(per_q)}")


if __name__ == "__main__":
    main()
