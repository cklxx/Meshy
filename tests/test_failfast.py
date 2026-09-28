"""Fail-fast: a fatal worker error must terminate the whole process group.

Regression for the hang where a worker's TQ thread logged a fatal error but the
engine process, ignitors, torchrun and launch.py all stayed alive for hours.

Process-level guarantees (spawn children, CPU-only, no GPU). Parent and child
share one TCPStore via XRL_BOOTSTRAP_ADDR:

* ``TQWorker.on_tq_fatal`` ends the process non-zero and publishes the cause;
* a passive Ignitor rank (no local engine) exits non-zero promptly once a
  sibling publishes a fatal marker.
"""

from __future__ import annotations

import multiprocessing
import os
import time

import pytest

from meshy.service import bootstrap

_TIMEOUT_S = 15.0


def _start_store() -> str:
    # Each test needs its own store; the module caches one per process, so
    # reset it explicitly rather than silently reusing the previous test's.
    bootstrap._store = None
    port = bootstrap._free_port()
    bootstrap.host_store("127.0.0.1", port)
    return f"127.0.0.1:{port}"


def _worker_fatal_child(addr: str) -> None:
    os.environ["XRL_BOOTSTRAP_ADDR"] = addr
    os.environ.setdefault("XRL_RUNTIME_DIR", "/tmp/failfast-runtime")
    from meshy.worker.tq import TQWorker

    w = TQWorker()
    w.configure_tq(endpoints_ref="stub", terminate_process_on_fatal=True)
    w.on_tq_fatal(AssertionError("scheduled LR step exceeded max: boom"))
    # Reaching here means the process did not die; linger so the test fails.
    time.sleep(30)


def _ignitor_passive_child(addr: str) -> None:
    os.environ["XRL_BOOTSTRAP_ADDR"] = addr
    os.environ.setdefault("XRL_RUNTIME_DIR", "/tmp/failfast-runtime")
    from meshy.service.ignite import Ignitor

    Ignitor([]).join()  # passive rank; must os._exit(1) on fatal marker
    os._exit(0)


def _wait_exit(proc, timeout=_TIMEOUT_S) -> int:
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail("process did not exit after a fatal within timeout")
    return proc.exitcode


def test_on_tq_fatal_terminates_process_nonzero() -> None:
    ctx = multiprocessing.get_context("spawn")
    addr = _start_store()
    proc = ctx.Process(target=_worker_fatal_child, args=(addr,))
    proc.start()
    code = _wait_exit(proc)
    assert code != 0, f"fatal worker exited cleanly ({code}); should be non-zero"
    # The fatal cause must reach the shared store.
    assert bootstrap.check([f"fatal|/tmp/failfast-runtime"])


def test_passive_ignitor_exits_on_fatal_marker() -> None:
    ctx = multiprocessing.get_context("spawn")
    addr = _start_store()
    proc = ctx.Process(target=_ignitor_passive_child, args=(addr,))
    proc.start()
    time.sleep(1.5)  # let the child enter join() and start watching
    bootstrap.set_json(
        "fatal|/tmp/failfast-runtime",
        {"source": "TitanWorker", "error": "AssertionError: boom", "pid": 1, "rank": "0"},
    )
    code = _wait_exit(proc)
    assert code == 1, f"passive ignitor should os._exit(1), got {code}"
