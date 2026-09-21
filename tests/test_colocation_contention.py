"""Two ON_DEMAND consumers contending for one card.

The JustRL-II ring is ``actor_infer(FALLBACK) -> critic -> actor_train``, and
for the whole of the critic's cold start only *one* consumer ever asks for the
card: the critic. ``actor_train`` first contends at the moment the critic stops
clearing its windows and starts publishing ``values``, which is what releases
the trainer's TQ fetch. That transition deadlocked the ring, so it gets a test.

The mechanism was mutual recursion. ``_reconcile`` starts a preempt whenever a
FALLBACK owner sees an open request; ``_transfer``'s phase-1 wait calls
``_reconcile`` to notice the next ring member arriving; and at that point the
card has not moved, so the same branch fires for the same request. With one
consumer phase 1 never loops (the next ring member *is* the one asking), so the
recursion needs a second consumer to appear. It ended in ``RecursionError``
inside the arbiter thread, which left every waiter blocked in
``wait_for_grant`` with no owner able to hand the card on.
"""

from __future__ import annotations

import threading
import time

import pytest

from meshy.service.colocation import (
    ColocationManager,
    ColocationRing,
    RequestKind,
    SchedulingMode,
    issue_genesis,
)
from meshy.transferqueue.colocation import RequestHandle, RequestRecord


class MemoryLedger:
    def __init__(self) -> None:
        self.rows: dict[str, list] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def create_request(self, request):
        with self._lock:
            self._counter += 1
            handle = RequestHandle(request, self._counter, 0)
            self.rows[request.request_id] = [handle, "open", 0, None]
            return handle

    def scan_requests(self):
        with self._lock:
            return [
                RequestRecord(handle, state, version, grant)
                for handle, state, version, grant in self.rows.values()
            ]

    def grant_request(self, handle, grant) -> None:
        with self._lock:
            row = self.rows[handle.request.request_id]
            row[1] = "granted"
            row[2] += 1
            row[3] = grant
            handle.version = row[2]

    def close_request(self, handle) -> None:
        with self._lock:
            row = self.rows[handle.request.request_id]
            row[1] = "closed"
            row[2] += 1
            handle.version = row[2]

    def purge_request(self, handle) -> None:
        with self._lock:
            self.rows.pop(handle.request.request_id, None)

    def close(self) -> None:
        pass


def wait_until(predicate, timeout: float = 10.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{what} was not reached within {timeout}s")
        time.sleep(0.002)


def _ring() -> ColocationRing:
    return ColocationRing(
        "actor_card",
        (
            ("actor_infer", SchedulingMode.FALLBACK),
            ("critic", SchedulingMode.ON_DEMAND),
            ("actor_train", SchedulingMode.ON_DEMAND),
        ),
        poll_interval=0.002,
        # Short enough to keep the test quick; long enough that phase 1 really
        # does loop, which is the condition that used to recurse.
        next_owner_timeout=1.0,
    )


def test_both_consumers_get_the_card_when_they_contend() -> None:
    """The publish-transition case: critic and trainer ask at the same time."""
    config = _ring()
    ledger = MemoryLedger()
    infer = ColocationManager(config, "actor_infer", lambda: ledger)
    critic = ColocationManager(config, "critic", lambda: ledger)
    train = ColocationManager(config, "actor_train", lambda: ledger)
    for m in (infer, critic, train):
        m.start()
    try:
        issue_genesis(config, ledger)
        wait_until(lambda: infer.owns_gpu, what="genesis owner")

        # Both consumers post before either is served -- the trainer's TQ fetch
        # unblocks the instant the critic writes `values`, so in the live run
        # these land within the same poll interval.
        train_req = train.request_gpu("actor_train:step:1")
        critic_req = critic.request_gpu("critic:window:32")

        # Ring distance decides: critic (index 1) is closer to the owner
        # (index 0) than actor_train (index 2).
        critic.wait_for_grant(critic_req, timeout=15)
        assert critic.owns_gpu

        critic.release(transition="critic-window-complete", payload_ref="/w/v0")
        train.wait_for_grant(train_req, timeout=15)
        assert train.owns_gpu

        train.release(transition="step-complete", payload_ref="/w/v1")
        wait_until(lambda: infer.owns_gpu, what="card back to fallback owner")

        for m in (infer, critic, train):
            assert m.fatal_error is None, f"{m.service_id} arbiter died: {m.fatal_error}"
    finally:
        for m in (infer, critic, train):
            m.stop()


def test_preempt_does_not_re_enter_itself() -> None:
    """The recursion, isolated: phase 1 loops while the owner still owns."""
    config = _ring()
    ledger = MemoryLedger()
    infer = ColocationManager(config, "actor_infer", lambda: ledger)
    issue_genesis(config, ledger)
    # Drive the arbiter by hand so the reconcile -> transfer edge is the only
    # thing under test.
    transport = ledger
    infer._reconcile(transport.scan_requests(), transport)
    assert infer.owns_gpu

    # Only the *far* member asks, so phase 1 waits for `critic` (which never
    # comes) and re-enters _reconcile while `actor_infer` still holds the card.
    from meshy.service.colocation import GpuRequest

    pending = GpuRequest(
        "actor_card",
        "actor_train",
        "actor_train:step:1",
        time.time_ns(),
        ring_index=config.ring_index_for("actor_train"),
        priority=0,
        kind=RequestKind.ON_DEMAND,
    )
    ledger.create_request(pending)

    # Must terminate, must hand the card to the only asker, must not recurse.
    infer._reconcile(transport.scan_requests(), transport)
    assert infer._transferring is False
    assert infer._owner_service == "actor_train"


def test_transfer_rejects_reentry_loudly() -> None:
    config = _ring()
    ledger = MemoryLedger()
    infer = ColocationManager(config, "actor_infer", lambda: ledger)
    issue_genesis(config, ledger)
    infer._reconcile(ledger.scan_requests(), ledger)
    assert infer.owns_gpu

    infer._transferring = True
    with pytest.raises(RuntimeError, match="re-entered"):
        infer._transfer(ledger, transition="preempt")
