"""Rollout per-window stats reach the trainer metrics dict (T11d).

The rollout appends rollout_window_stats.jsonl; TitanTrainer joins it by
weight_version and emits TensorBoard scalar names. Also pins the
dynamic/partial mutual-exclusion guard.
"""

from __future__ import annotations

import json
import os

import pytest


def _make_trainer(stats_path):
    # Construct only the metrics-mixin state without the heavy torch backend.
    from meshy.backend.titan.metrics import TitanTrainer

    t = TitanTrainer.__new__(TitanTrainer)
    t._rollout_window_stats_path = stats_path
    return t


def test_dynamic_window_stats_join_by_version(tmp_path):
    p = tmp_path / "stats.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({
            "kind": "dynamic", "window": 1, "weight_version": 5,
            "prompts_drawn": 11, "valid_groups": 8, "valid_samples": 64,
            "groups_dropped_zero_variance": 3, "refill_count": 3,
            "filtered_ratio": 3 / 11,
        }) + "\n")
    t = _make_trainer(str(p))
    m: dict = {}
    t._attach_window_stats(m, 5)
    assert m["grpo_metrics/filtered_ratio"] == pytest.approx(3 / 11)
    assert m["grpo_metrics/refill_count"] == 3
    assert m["grpo_metrics/groups_dropped_zero_variance"] == 3
    assert m["rollout/window_prompts_drawn"] == 11
    assert m["rollout/window_valid_groups"] == 8


def test_partial_window_stats_join(tmp_path):
    p = tmp_path / "stats.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({
            "kind": "partial", "window": 1, "weight_version": 2,
            "groups_closed": 7, "groups_deferred": 1,
            "tail_threshold_samples": 8, "max_windows": 2,
        }) + "\n")
    t = _make_trainer(str(p))
    m: dict = {}
    t._attach_window_stats(m, 2)
    assert m["rollout/partial_groups_deferred"] == 1
    assert m["rollout/partial_groups_closed"] == 7


def test_missing_or_unmatched_version_is_noop(tmp_path):
    t = _make_trainer(str(tmp_path / "absent.jsonl"))
    m: dict = {}
    t._attach_window_stats(m, 1)
    assert m == {}
    p = tmp_path / "stats.jsonl"
    p.write_text(json.dumps({"kind": "dynamic", "weight_version": 9,
                             "filtered_ratio": 0.5, "refill_count": 0,
                             "groups_dropped_zero_variance": 0,
                             "prompts_drawn": 8, "valid_groups": 8}) + "\n")
    t._rollout_window_stats_path = str(p)
    t._attach_window_stats(m, 10)  # no matching version
    assert m == {}


def test_dynamic_and_partial_are_mutually_exclusive():
    from meshy.worker.rollout import RolloutWorker
    with pytest.raises(ValueError, match="cannot be enabled together"):
        RolloutWorker(
            engine=object(), endpoints_ref="x", model_path="m", dataset="d",
            dataset_kwargs=None, partition_id="p", group_size=8,
            train_batch_size=64, sampling_params={}, reward=lambda s: 0.0,
            dynamic_sampling=True, partial_rollout=True,
        )
