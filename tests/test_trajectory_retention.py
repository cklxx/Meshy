"""TrajectoryLogger two-tier slim audit log (T5i-slim).

Every sample is logged with scalar/tag fields but no full dialogue (slim);
XRL_TRAJ_FULL_SAMPLES per window also keep the full trajectory, sampled to
span the reward range. The full 40-window audit chain stays ~3 MB instead of
hundreds of GB, and writes stay append-only.
"""

from __future__ import annotations

import asyncio
import json
import os


class _S:
    def __init__(self, i: int, reward: float):
        self.messages = [{"role": "user", "content": f"p{i}"},
                         {"role": "assistant", "content": f"a{i}"}]
        self.ground_truth = i
        self.reward = reward
        self.advantage = reward - 0.5
        self.finish_reason = 1
        self.truncated = False
        self.repetition = False
        self.mixed_version = False
        self.masks = [1, 1, 1]
        self.tokens = [1, 2, 3]
        self.logprobs = [0.1, 0.2, 0.3]


def _logger(tmp_path, monkeypatch, env: dict):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    from meshy.worker.rollout import TrajectoryLogger
    return TrajectoryLogger(os.path.join(tmp_path, "trajectories.jsonl"))


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _load(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(ln) for ln in fh.read().splitlines() if ln]


def test_every_sample_logged_slim_with_scalars_only(tmp_path, monkeypatch) -> None:
    log = _logger(tmp_path, monkeypatch,
                  {"XRL_TRAJ_SLIM": "1", "XRL_TRAJ_FULL_SAMPLES": "0"})
    _run(log.write([_S(i, float(i % 2)) for i in range(512)], version=0))
    recs = _load(log.path)
    assert len(recs) == 512                      # full audit chain, one per sample
    for r in recs:
        assert "trajectory" not in r             # no full dialogue
        for key in ("round", "weight_version", "reward", "advantage",
                    "truncated", "repetition", "mixed_version",
                    "response_tokens", "ground_truth", "finish_reason"):
            assert key in r                      # all audit scalars retained
    assert recs[0]["weight_version"] == 0 and recs[0]["round"] == 1


def test_full_samples_keep_dialogue_and_are_flagged(tmp_path, monkeypatch) -> None:
    log = _logger(tmp_path, monkeypatch,
                  {"XRL_TRAJ_SLIM": "1", "XRL_TRAJ_FULL_SAMPLES": "16"})
    samples = [_S(i, 1.0 if i < 256 else 0.0) for i in range(512)]
    _run(log.write(samples, version=3))
    recs = _load(log.path)
    assert len(recs) == 512
    full = [r for r in recs if r.get("full")]
    slim = [r for r in recs if not r.get("full")]
    assert len(full) == 16
    assert all("trajectory" in r for r in full)
    assert all("trajectory" not in r for r in slim)
    # Spans both reward classes (high and low), not only the top.
    assert {r["reward"] for r in full} == {0.0, 1.0}
    assert all(r["weight_version"] == 3 for r in full)


def test_full_sampling_is_deterministic(tmp_path, monkeypatch) -> None:
    def picked():
        log = _logger(tmp_path, monkeypatch,
                      {"XRL_TRAJ_SLIM": "1", "XRL_TRAJ_FULL_SAMPLES": "8"})
        samples = [_S(i, float(i % 3) / 2) for i in range(100)]
        _run(log.write(samples, version=0))
        return {r["ground_truth"] for r in _load(log.path) if r.get("full")}
    a = picked()
    b = picked()
    assert a == b and len(a) == 8


def test_slim_off_logs_full_dialogue_for_all(tmp_path, monkeypatch) -> None:
    log = _logger(tmp_path, monkeypatch,
                  {"XRL_TRAJ_SLIM": "0", "XRL_TRAJ_FULL_SAMPLES": "0"})
    _run(log.write([_S(i, 1.0) for i in range(10)], version=0))
    recs = _load(log.path)
    assert len(recs) == 10 and all("trajectory" in r for r in recs)
    # Full path doesn't set the slim-only flag.
    assert all(not r.get("full") for r in recs)


def test_append_only_no_rewrite_in_slim_mode(tmp_path, monkeypatch) -> None:
    # With keep_lines=0 (default) successive windows strictly append; the file
    # is never rewritten, so a crash between windows cannot lose prior data.
    log = _logger(tmp_path, monkeypatch,
                  {"XRL_TRAJ_SLIM": "1", "XRL_TRAJ_FULL_SAMPLES": "4",
                   "XRL_TRAJ_KEEP_LINES": "0"})
    _run(log.write([_S(i, 0.0) for i in range(512)], version=0))
    size1 = os.path.getsize(log.path)
    _run(log.write([_S(i, 1.0) for i in range(512)], version=1))
    assert len(_load(log.path)) == 1024         # both windows retained
    assert os.path.getsize(log.path) > size1
