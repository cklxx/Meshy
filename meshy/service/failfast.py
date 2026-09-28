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
_DONE_KEY_PREFIX = "done|"
_FATAL_POLL_INTERVAL = float(os.environ.get("XRL_FATAL_POLL_INTERVAL", "0.5"))

#: Participants that must each report completion before the run is "done":
#: the rollout producer is exhausted AND the trainer has run its final step and
#: published the last weights. Override with XRL_DONE_SOURCES (comma list) for
#: flows with a different role set (e.g. a critic-only pipeline).
_DONE_SOURCES_DEFAULT = ("rollout", "titan")


def _done_sources() -> tuple[str, ...]:
    raw = os.environ.get("XRL_DONE_SOURCES", "")
    if raw.strip():
        return tuple(s.strip() for s in raw.split(",") if s.strip())
    return _DONE_SOURCES_DEFAULT


def _key(kind: str) -> str:
    root = os.environ.get("XRL_RUNTIME_DIR", "")
    return (f"{kind}|{{root}}").format(root=root)


def _fatal_key() -> str:
    return _key("fatal")


def _done_key(source: str) -> str:
    root = os.environ.get("XRL_RUNTIME_DIR", "")
    return f"{_DONE_KEY_PREFIX}{source}|{root}"


def fatal_published() -> bool:
    """True iff any process in the run has already published a fatal error."""
    try:
        return bootstrap.check([_fatal_key()])
    except Exception:
        # Store unreachable during teardown is not itself a fatal condition.
        return False


def fatal_info() -> dict | None:
    """The first recorded fatal payload, or None."""
    try:
        if not bootstrap.check([_fatal_key()]):
            return None
        return bootstrap.get_json(_fatal_key())
    except Exception:
        return None


def done_published() -> bool:
    """True iff *every* expected completion source has reported done."""
    try:
        return bootstrap.check([_done_key(s) for s in _done_sources()])
    except Exception:
        return False


def publish_done(source: str = "rollout") -> None:
    """Record that one completion source (e.g. rollout/titan) is finished.

    The run only exits 0 once all expected sources have published. Best-effort.
    """
    payload = {"source": str(source), "pid": os.getpid(),
               "rank": os.environ.get("RANK")}
    try:
        k = _done_key(source)
        if not bootstrap.check([k]):
            bootstrap.set_json(k, payload)
    except Exception:
        pass


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
        if not bootstrap.check([_fatal_key()]):
            bootstrap.set_json(_fatal_key(), payload)
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


def watch_for_completion(
    is_fatal: Callable[[], bool | str | BaseException],
    *,
    interval: float = _FATAL_POLL_INTERVAL,
) -> tuple[bool, str | None]:
    """Block until either clean completion (done) or a fatal appears.

    Returns ``(done, cause)``: ``(True, None)`` on clean completion,
    ``(False, cause)`` on a fatal. ``is_fatal`` folds in local child liveness
    exactly as :func:`watch_for_fatal`.
    """
    while True:
        info = fatal_info()
        if info is not None:
            return False, f"{info.get('source')}: {info.get('error')}"
        if done_published():
            return True, None
        local = is_fatal()
        if local:
            cause = str(local) if not isinstance(local, BaseException) else f"{type(local).__name__}: {local}"
            return False, cause
        time.sleep(interval)
