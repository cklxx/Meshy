import json
import statistics as st
import sys

from meshy.backend.titan.plan import PlannerConfig, build_plan, plan_stats

rt = sys.argv[1]
rows = [json.loads(l) for l in open(rt)]
latest = sorted({r["weight_version"] for r in rows})[-1]
win = [r for r in rows if r["weight_version"] == latest][:512]
if len(win) < 512:
    win = rows[-512:]
lengths = [min(r["response_tokens"] + 200, 5120) for r in win]
loss_t = [r["response_tokens"] for r in win]
print("window wv", latest, "n", len(lengths),
      "mean_len", int(st.mean(lengths)), "p50", sorted(lengths)[len(lengths)//2],
      "p95", sorted(lengths)[int(len(lengths)*.95)], "max", max(lengths))


def run(budget):
    cfg = PlannerConfig(mini_batch_size=64, micro_batch_size=1, seq_len=5120,
                        align=1024, layout="padded", max_tokens_per_micro=budget)
    plan = build_plan(lengths, loss_t, 1, cfg)
    minis = plan.per_rank[0]
    micros = [mc for mn in minis for mc in mn.micros if mc.sample_idx]
    stats = plan_stats(minis, "padded")
    max_rows = max(len(mc.sample_idx) for mc in micros)
    max_seq = max(mc.seq_len for mc in micros)
    print(f"budget={budget} n_mini={len(minis)} n_micro={len(micros)} "
          f"pad={stats['padding_ratio']:.3f} max_rows={max_rows} max_seqlen={max_seq}")


for b in (4096, 8192, 16384, 24576):
    run(b)

cfg0 = PlannerConfig(mini_batch_size=64, micro_batch_size=1, seq_len=5120,
                     align=1024, layout="padded", max_tokens_per_micro=None)
p0 = build_plan(lengths, loss_t, 1, cfg0)
s0 = plan_stats(p0.per_rank[0], "padded")
n0 = sum(1 for mn in p0.per_rank[0] for mc in mn.micros if mc.sample_idx)
print(f"baseline micro=1: n_micro={n0} pad={s0['padding_ratio']:.3f}")
