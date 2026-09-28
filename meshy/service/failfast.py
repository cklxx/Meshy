"""Run-wide fail-fast coordination over the bootstrap store.

A worker subprocess (trainer / rollout / critic) can hit a fatal error on one
card while every other process keeps running: its TQ thread logs the error but
the engine command loop stays alive, so torchrun never exits and a long
unattended run hangs for hours holding the GPU.

This module gives the whole process group one shared "something died" key in
the bootstrap store:

* the failing worker :meth:`publish_fatal` records ``{source, error}`` and then
  terminates its own process;
* every Ignitor rank :meth:`watch_for_fatal`-polls that key (and the liveness
  of its local engine subprocesses) while joining, and on the first fatal it
  tears down its local services and exits non-zero so torchrun / launch.py
  return non-zero as well.

The first recorded cause is preserved verbatim for the launcher to print.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable

from meshy.service import bootstrap

_FATAL_KEY = "fatal|{root}"
_FATAL_POLL_INTERVAL = float(os.environ.get("XRL_FATAL_POLL_INTERVAL", "0.5"))


def _key() -> str:
    root = os.environ.get("XRL_RUNTIME_DIR", "")
    return _FATAL_KEY.format(root=root)


def fatal_published() -> bool:
    """True iff any process in the run has already published a fatal error."""
    try:
        return bootstrap.check([_key()])
    except Exception:
        # Store unreachable during teardown is not itself a fatal condition.
        return False


def fatal_info() -> dict | None:
    """The first recorded fatal payload, or None."""
    try:
        if not bootstrap.check([_key()]):
            return None
        return bootstrap.get_json(_key())
    except Exception:
        return None


def publish_fatal(source: str, error: BaseException | str) -> None:
    """Record the first fatal cause. Best-effort; never raises."""
    payload = {
        "source": str(source),
        "error": f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error),
        "pid": os.getpid(),
        "rank": os.environ.get("RANK"),
    }
    try:
        # set() overwrites; compare-add would be nicer but TCPStore offers no
        # atomic put-if-absent in all versions. The first writer wins in
        # practice (fatal is a rare one-shot) and every payload is a fatal.
        if not bootstrap.check([_key()]):
            bootstrap.set_json(_key(), payload)
    except Exception:
        pass


def watch_for_fatal(
    is_fatal: Callable[[], bool | str | BaseException],
    *,
    interval: float = _FATAL_POLL_INTERVAL,
    stop_after: float | None = None,
) -> str | None:
    """Block until a run-wide fatal appears or ``is_fatal()`` reports one.

    ``is_fatal`` lets the caller fold local liveness (e.g. an engine child
    exited unexpectedly) into the same wait; a truthy return is treated as a
    fatal (a non-empty string/exception becomes its description). Returns the
    fatal description, or None on timeout/stop.
    """
    deadline = None if stop_after is None else time.monotonic() + stop_after
    while True:
        info = fatal_info()
        if info is not None:
            return f"{info.get('source')}: {info.get('error')}"
        local = is_fatal()
        if local:
            return str(local) if not isinstance(local, BaseException) else f"{type(local).__name__}: {local}"
        if deadline is not None and time.monotonic() >= deadline:
            return None
        time.sleep(interval)
