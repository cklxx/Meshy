from __future__ import annotations

import asyncio
import os

from meshy.dataset.gsm8k import lenient_gsm8k_answer
from meshy.worker.rollout import RolloutWorker
import recipe.v100_inloop as inloop


def test_lenient_answer_cases():
    cases = [
        ("</think>\n\\(\\boxed{\\$18}\\)", 18.0),
        ("</think>\n#### 260", 260.0),
        ("</think>\nSo the answer is 3.", 3.0),
        ("</think>\n\\boxed{260\\text{ sheep}}", 260.0),
        ("</think>\nno numbers here", None),
        # reasoning number must not leak; only post-think span counts
        ("long 9 in reasoning </think>\nanswer #### 7", 7.0),
    ]
    for text, want in cases:
        assert lenient_gsm8k_answer(text) == want, text


def test_prune_keeps_last_three(tmp_path, monkeypatch):
    root = tmp_path / "run"
    wdir = root / "weights" / "actor_train-0"
    wdir.mkdir(parents=True)
    for k in range(0, 8):
        (wdir / f"v{k}").mkdir()
    monkeypatch.setenv("XRL_RUNTIME_DIR", str(root))
    removed = inloop._prune(str(root), 7)
    remaining = sorted(int(n[1:]) for n in os.listdir(wdir))
    assert remaining == [5, 6, 7]
    assert sorted(removed) == ["v0", "v1", "v2", "v3", "v4"]


def test_worker_hook_runs_once_per_version():
    seen = []

    async def hook(*, version, engine, model_path, **kw):
        seen.append(version)

    w = object.__new__(RolloutWorker)
    w.version_hook_fn = hook
    w.version_hook_kwargs = {}
    w.engine = object()
    w.model_path = "/m"
    w._hooks_run = set()
    w._version_hook_lock = asyncio.Lock()

    async def drive():
        await w._run_version_hook(0)
        await w._run_version_hook(0)   # same version, concurrent groups
        await w._run_version_hook(1)

    asyncio.run(drive())
    assert seen == [0, 1]


def test_worker_hook_absent_is_noop():
    w = object.__new__(RolloutWorker)
    w.version_hook_fn = None
    asyncio.run(w._run_version_hook(3))
