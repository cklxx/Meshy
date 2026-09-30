"""DAPO dynamic sampling window-fill semantics (T10).

Drives the real ``RolloutWorker._run_dynamic_rollouts`` with a stub engine,
stub dataset and captured TQ writes -- no TransferQueue, no GPU.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from meshy.worker.rollout import RolloutWorker


class _PromptDataset:
    """Scripted prompt stream.

    ``spec[i]`` is True when prompt i yields a zero-variance (all-wrong) group,
    False for an informative (mixed 0/1) group. Every drawn id is recorded so a
    test can assert no prompt is served twice.
    """

    def __init__(self, spec: list[bool], windows: int = 100):
        self.spec = spec
        self.index = 0
        self.drawn: list[int] = []
        self.windows_left = windows

    def begin_window(self) -> bool:
        if self.windows_left <= 0:
            return False
        self.windows_left -= 1
        return True

    def take_prompts(self, builder, n, *, window_start=False):
        if window_start and not self.begin_window():
            return []
        prompts = []
        for _ in range(n):
            if self.index >= len(self.spec):
                break
            i = self.index
            self.index += 1
            self.drawn.append(i)
            prompts.append({"id": i, "zero": self.spec[i]})
        return prompts


def _worker(
    *,
    target: int,
    max_prompts: int,
    group_size: int = 8,
    max_windows: int = 1,
):
    w = RolloutWorker.__new__(RolloutWorker)
    w._worker_stop = threading.Event()
    w.model_path = "unused/model"
    w.num_epochs = 1
    w.group_size = group_size
    w.max_running_requests = -1
    w.dynamic_target_groups = target
    w.dynamic_max_prompts = max_prompts
    w.weight_version = 0
    w.samples_started = 0
    w.groups_seen = 0
    w.groups_filtered = 0
    w.dynamic_windows = 0
    w.dynamic_prompts_drawn_total = 0
    w.dynamic_groups_dropped_total = 0
    w.dataset_kwargs = {}
    w.window_stats_log = None
    w.window_cursor = None
    w.written: list[tuple[int, list]] = []
    return w


def _install(worker: RolloutWorker, dataset: _PromptDataset, monkeypatch):
    worker.dataset_factory = lambda **kw: dataset
    monkeypatch.setattr(
        "meshy.utils.sample.SampleBuilder", lambda model_path: object()
    )

    async def acquire(num_samples):
        worker.samples_started += num_samples
        return 0

    async def rollout_group(builder, prompt, version):
        worker.groups_seen += 1
        if prompt["zero"]:
            rewards = [0.0] * worker.group_size
        else:
            rewards = [0.0 if j % 2 else 1.0 for j in range(worker.group_size)]
        return [
            SimpleNamespace(reward=r, masks=[1] * 10, truncated=False) for r in rewards
        ]

    async def write(group, version):
        worker.written.append((version, group))

    worker.acquire_generation_slot = acquire
    worker.rollout_group = rollout_group
    worker._write_rollout_group = write


def _run(worker):
    asyncio.run(worker._run_dynamic_rollouts())


def test_zero_variance_groups_dropped_and_window_refilled(monkeypatch):
    # First two prompts are all-wrong, then plenty of informative ones.
    spec = [True, True] + [False] * 20
    ds = _PromptDataset(spec, windows=1)
    w = _worker(target=4, max_prompts=12)
    _install(w, ds, monkeypatch)

    _run(w)

    assert len(w.written) == 4                      # exactly target valid groups
    assert all(any(s.reward == 1.0 for s in g) for _, g in w.written)
    assert w.groups_filtered == 2
    assert ds.index == 6                            # advanced past the dropped ids
    assert len(set(ds.drawn)) == len(ds.drawn)      # no prompt served twice
    # pacing slots refunded: only kept groups' samples remain counted
    assert w.samples_started == 4 * w.group_size
    assert w.dynamic_windows == 1


def test_oversample_cap_keeps_remaining_groups_to_fill_window(monkeypatch):
    # Every prompt is zero-variance. target 4, replacement budget = 6-4 = 2:
    # two drops, then the cap forces the next four groups to be kept.
    ds = _PromptDataset([True] * 20, windows=1)
    w = _worker(target=4, max_prompts=6)
    _install(w, ds, monkeypatch)

    _run(w)

    assert len(w.written) == 4                      # trainer still gets a full window
    assert w.groups_filtered == 2
    assert ds.index == 6                            # hard cap, not an infinite loop
    assert w.samples_started == 4 * w.group_size


def test_dropped_prompts_advance_position_across_windows(monkeypatch):
    # Two windows: one dropped prompt each, never repeated.
    spec = [True, False, False, False, False, False, True, False, False, False]
    ds = _PromptDataset(spec, windows=2)
    w = _worker(target=3, max_prompts=9)
    _install(w, ds, monkeypatch)

    _run(w)

    assert len(w.written) == 6                      # 3 valid groups x 2 windows
    assert w.groups_filtered == 2
    assert len(set(ds.drawn)) == len(ds.drawn) == ds.index


def test_run_ends_when_dataset_exhausted_mid_window(monkeypatch):
    # Only two informative groups exist for a target of four: one window with
    # two groups is written, then the stream ends (no third draw).
    ds = _PromptDataset([True, False, False], windows=1)
    w = _worker(target=4, max_prompts=12)
    _install(w, ds, monkeypatch)

    _run(w)

    assert len(w.written) == 2
