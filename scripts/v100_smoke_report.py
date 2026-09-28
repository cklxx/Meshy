#!/usr/bin/env python3
"""Extract the 2-step smoke report from trajectories.jsonl + logs + TB."""
import json
import statistics
import sys
from collections import defaultdict

RT = sys.argv[1]
rows = [json.loads(l) for l in open(RT + "/trajectories.jsonl")]

by_round = defaultdict(list)
for r in rows:
    by_round[r["round"]].append(r)

print("== per-round rollout ==")
for rnd in sorted(by_round):
    rs = by_round[rnd]
    groups = defaultdict(list)
    for r in rs:
        q = next(m["content"] for m in r["trajectory"] if m["role"] == "user")
        groups[q].append(r)
    n_zero = 0
    adv_zero = 0
    for g in groups.values():
        vals = {x["reward"] for x in g}
        if len(vals) == 1:
            n_zero += 1
        if abs(statistics.pstdev(x["advantage"] for x in g)) < 1e-9:
            adv_zero += 1
    rew = [r["reward"] for r in rs]
    toks = [r["response_tokens"] for r in rs]
    trunc = sum(r["truncated"] for r in rs)
    solve = sum(1 for r in rs if r["reward"] == 1.0)
    print(f"round {rnd} (wv {rs[0]['weight_version']}): n={len(rs)} "
          f"reward_mean={statistics.mean(rew):.4f} solve={solve}/{len(rs)} "
          f"trunc={trunc} ({trunc/len(rs)*100:.1f}%) "
          f"tok_mean={statistics.mean(toks):.0f} "
          f"p50={statistics.quantiles(toks, n=100)[49]:.0f} "
          f"p90={statistics.quantiles(toks, n=100)[89]:.0f} max={max(toks)} "
          f"zero-var-groups={n_zero}/{len(groups)} adv-std0={adv_zero}")
    # repetition / mixed flags
    rep = sum(r["repetition"] for r in rs)
    mixed = sum(r["mixed_version"] for r in rs)
    print(f"  repetition={rep} mixed_version={mixed} "
          f"finish={dict((fr, sum(1 for r in rs if r['finish_reason']==fr)) for fr in {r['finish_reason'] for r in rs})}")

print("\n== round-2 generation samples (post-update, weight v1) ==")
r2 = by_round[2]
for r in r2[:3]:
    text = next(m["content"] for m in r["trajectory"] if m["role"] == "assistant")
    q = next(m["content"] for m in r["trajectory"] if m["role"] == "user")
    print(f"--- reward={r['reward']} gt={r['ground_truth']} trunc={r['truncated']} toks={r['response_tokens']}")
    print("Q:", q[:120])
    print("A head:", repr(text[:200]))
    print("A tail:", repr(text[-200:]))
