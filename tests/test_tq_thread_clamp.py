"""TQ worker-thread clamp: never exceed physical cores unless the user opted in."""

from __future__ import annotations

import os

from meshy.transferqueue.client import _clamp_tq_threads


def test_clamp_caps_to_physical_cores(monkeypatch) -> None:
    monkeypatch.delenv("TQ_NUM_THREADS", raising=False)
    import psutil

    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    _clamp_tq_threads()
    assert int(os.environ["TQ_NUM_THREADS"]) == int(physical)
    assert int(os.environ["TQ_NUM_THREADS"]) <= 8  # the old default on <=8-core boxes


def test_clamp_respects_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("TQ_NUM_THREADS", "2")
    _clamp_tq_threads()
    assert os.environ["TQ_NUM_THREADS"] == "2"
