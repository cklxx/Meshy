#!/usr/bin/env python3
"""Dump per-step scalar metrics from a Meshy smoke run's tensorboard events."""
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

rt = sys.argv[1]
ea = EventAccumulator(rt + "/tensorboard", size_guidance={"scalars": 0})
ea.Reload()
want = sys.argv[2:] or [
    "rollout/raw_reward_mean", "train/pg_loss", "train/loss_scale",
    "train/grad_norm", "time/train/forward", "time/train/backward",
    "time/train/optim_step", "time/train/old_logprobs",
]
for tag in ea.Tags()["scalars"]:
    if want and not any(w in tag for w in want):
        continue
    vals = [(s.step, s.value) for s in ea.Scalars(tag)]
    print(tag, " ".join(f"{s}:{v:.4g}" for s, v in vals))
