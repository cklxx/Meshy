from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx

from meshy.engine.sglang import SGLangEngine
from meshy.engine.titan import _publish_weights_to_inference
from meshy.config import InferenceServiceConfig, TrainingServiceConfig
from meshy.service.base import GPU, ServiceGroup
from meshy.service.colocation import GpuGrant
from meshy.service.colocation import ColocationRing, SchedulingMode
from meshy.service.topology import build_topology
from meshy.worker.titan import TitanWorker


def test_publisher_updates_only_inference_outside_colocation_ring(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self, **kwargs):
            # trust_env must be disabled so httpx ignores the broken IPv6 NO_PROXY
            assert kwargs.get("trust_env") is False
            self.timeout = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url: str, *, json: dict | None = None, timeout: float | None = None):
            calls.append((url, json, self.timeout if timeout is None else timeout))
            return Response()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    _publish_weights_to_inference(
        [
            {"name": "actor-infer-0", "endpoint": "http://infer:30000"},
            {"name": "standalone-infer-0", "endpoint": "http://infer:30001/"},
        ],
        "/runtime/weights/titan/v3",
        3,
        "titan",
        managed_names={"actor-infer-0"},
    )

    # Weight push, then mandatory L3 invalidation on the same replica.
    assert [(u, j) for u, j, _ in calls] == [
        (
            "http://infer:30001/update_weights_from_disk",
            {"model_path": "/runtime/weights/titan/v3"},
        ),
        ("http://infer:30001/clear_hicache_storage_backend", None),
    ]
    assert calls[0][2] == 1800.0


def test_publisher_tolerates_replica_without_hicache_backend(monkeypatch) -> None:
    import httpx as _httpx

    class Ok:
        def raise_for_status(self) -> None:
            return None

    class NotFound(_httpx.HTTPStatusError):
        def __init__(self) -> None:
            resp = SimpleNamespace(status_code=404)
            super().__init__("no backend", request=SimpleNamespace(), response=resp)

    sequence = ["update", "clear"]

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs.get("trust_env") is False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url: str, *, json: dict | None = None, timeout: float | None = None):
            stage = sequence.pop(0)
            if stage == "clear":
                raise NotFound()
            return Ok()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    # Must not raise: 404 on the clear endpoint just means L3 is disabled.
    _publish_weights_to_inference(
        [{"name": "s", "endpoint": "http://infer:30001/"}],
        "/w/v1", 1, "titan", managed_names=set(),
    )
    assert sequence == []


def test_load_weights_clears_hicache_l3(monkeypatch) -> None:
    """A weight swap must invalidate token-keyed L3 KV (regression: stale old-
    policy KV otherwise survives SGLang's GPU-radix-only flush and is read
    under the new weights). Fails if the clear endpoint is not called."""
    import requests

    posted: list[str] = []

    class Resp:
        def __init__(self, status: int = 200) -> None:
            self.status_code = status

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError("boom", response=self)

    def fake_request(url: str, **kwargs):
        posted.append(f"POST {url}")
        return Resp(200)

    monkeypatch.setattr(requests, "post", fake_request)
    monkeypatch.setattr(requests, "get", fake_request)

    engine = SGLangEngine(["http://infer:30000"])
    try:
        engine.load_weights("/weights/v9")
    finally:
        asyncio.run(engine.close())

    assert any(url.endswith("/update_weights_from_disk") for _, url in
               (p.split(" ", 1) for p in posted))
    assert any(url.endswith("/clear_hicache_storage_backend") for _, url in
               (p.split(" ", 1) for p in posted)), posted


def test_sglang_engine_tolerates_ipv6_no_proxy(monkeypatch) -> None:
    # The V100 login environment puts bare ::1 and IPv6 CIDRs (fe80::/10) in
    # NO_PROXY, which httpx 0.28 cannot parse (InvalidURL: Invalid port ':').
    # Local/in-cluster clients must build anyway.
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "localhost,127.0.0.1,::1,fe80::/10,fd00::/8")
    engine = SGLangEngine(["http://127.0.0.1:30000"])
    try:
        assert engine.client.is_closed is False
    finally:
        asyncio.run(engine.close())


def test_publisher_tolerates_ipv6_no_proxy(monkeypatch) -> None:
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "localhost,127.0.0.1,::1,fe80::/10,fd00::/8")

    class Response:
        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs.get("trust_env") is False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, *, json=None, **kwargs):
            return Response()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    # must not raise on the unparseable NO_PROXY
    _publish_weights_to_inference(
        [{"name": "infer-0", "endpoint": "http://127.0.0.1:30001/"}],
        "/w/v1",
        1,
        "titan",
        managed_names=set(),
    )


def test_sglang_colocate_acquire_uses_checkpoint_from_grant() -> None:
    engine = SGLangEngine(["http://infer:30000"])
    restored: list[str] = []
    engine.model_path = "/models/base"
    engine.restore_for_colocate = restored.append  # type: ignore[method-assign]
    try:
        engine.on_colocate_acquire(
            GpuGrant("cards", 4, "titan", "actor-infer", payload_ref="/weights/v4")
        )
    finally:
        asyncio.run(engine.close())

    assert restored == ["/weights/v4"]


def test_sglang_non_genesis_acquire_rejects_missing_checkpoint() -> None:
    engine = SGLangEngine(["http://infer:30000"])
    engine.model_path = "/models/base"
    try:
        try:
            engine.on_colocate_acquire(
                GpuGrant("cards", 4, "titan", "actor-infer", payload_ref=None)
            )
        except RuntimeError as exc:
            assert "latest checkpoint" in str(exc)
        else:
            raise AssertionError("missing checkpoint path was accepted")
    finally:
        asyncio.run(engine.close())


def test_titan_worker_attaches_checkpoint_to_colocation_release() -> None:
    class Colocation:
        def __init__(self) -> None:
            self.releases: list[dict] = []

        def release(self, **kwargs) -> None:
            self.releases.append(kwargs)

    worker = TitanWorker.__new__(TitanWorker)
    worker.engine = SimpleNamespace(
        name="titan",
        step_index=2,
        weight_version=2,
        stream_minibatch=False,
        step=lambda samples, sync: SimpleNamespace(
            step=3, weights_path="/weights/v3"
        ),
    )
    worker.colocation = Colocation()
    worker._colocation_request = object()
    worker.batch_size = 1
    worker.trained_since_sync = 0
    worker.stream_minibatch = False
    worker._gate_output = lambda *, step: {"gate": step}

    assert worker.process_tq_batch([SimpleNamespace()]) == {"gate": 3}
    assert worker.colocation.releases == [
        {"transition": "train-step-complete", "payload_ref": "/weights/v3"}
    ]
    assert worker._colocation_request is None


def test_colocation_elects_one_manager_per_ring_member_group() -> None:
    groups = [
        ServiceGroup(
            id="actor-infer",
            n_replicas=2,
            n_gpus_per_replica=1,
            config=InferenceServiceConfig(model_path="/models/base"),
        ),
        ServiceGroup(
            id="actor-train",
            n_replicas=1,
            n_gpus_per_replica=2,
            colocate_with="actor-infer",
            config=TrainingServiceConfig(
                model_path="/models/base",
                trainer_config=SimpleNamespace(),
                batch_size=1,
            ),
        ),
    ]
    topology = build_topology(
        groups,
        [GPU("127.0.0.1", 0, 0, 0), GPU("127.0.0.1", 1, 0, 1)],
        [
            ColocationRing(
                "actor-card",
                (("actor-infer", SchedulingMode.FALLBACK), ("actor-train", SchedulingMode.ON_DEMAND)),
                poll_interval=0.001,
            )
        ],
    )

    assert [s.is_colocation_leader for s in topology.group("actor-infer")] == [True, False]
    assert [s.is_colocation_leader for s in topology.group("actor-train")] == [True]
