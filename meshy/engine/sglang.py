"""SGLang HTTP generation and colocate-management engine.

The engine is intentionally a client.  :class:`meshy.service.inference.SGLangService`
owns the SGLang server subprocess; this class only talks to an already running
server and exposes the API consumed by rollout code.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Callable, Iterable, Iterator

import httpx
from loguru import logger


@dataclass
class _GenerationChunk:
    tokens: list[int]
    logprobs: list[float]
    finished: bool
    finish_reason: str | None = None


@dataclass
class Generation:
    """One completed generation.

    Iterating yields ``(tokens, logprobs)`` so existing
    ``tokens, logprobs = await engine.generate(...)`` call sites keep working
    while the metadata rides along for callers that want it.
    """

    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    #: SGLang ``finish_reason.type`` of the final request: ``"stop"`` (EOS or
    #: stop string) or ``"length"`` (``max_new_tokens`` reached).
    finish_reason: str | None = None
    #: number of abort/resume cycles; each one means the request was paused
    #: for a colocated training step and resumed on the *next* weight version.
    continuations: int = 0

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    def __iter__(self) -> Iterator[list[Any]]:
        yield self.tokens
        yield self.logprobs


class SGLangEngine:
    """Round-robin SGLang client with retry and abort continuation support."""

    def __init__(
        self,
        endpoints: Iterable[str] | str,
        sampling_params: dict[str, Any] | None = None,
        *,
        max_connections: int = 1024,
        attempts: int = 60,
        backoff: float = 1.0,
        backoff_max: float = 5.0,
        max_continuations: int = 128,
        hicache_storage_enabled: bool = False,
    ) -> None:
        if isinstance(endpoints, str):
            endpoints = [endpoints]
        normalized = [str(endpoint).rstrip("/") for endpoint in endpoints]
        if not normalized:
            raise ValueError("SGLangEngine requires at least one endpoint")
        self._cycle_values = tuple(normalized)
        self.endpoints = cycle(self._cycle_values)
        self._endpoint_lock = threading.Lock()
        self.sampling_params = dict(sampling_params or {})
        self.model_path: str | None = None
        # Whether this server was launched with an HiCache L3 backend. Only
        # then is /clear_hicache_storage_backend valid; when L3 is off SGLang
        # answers 400 ("Hierarchical cache is not enabled"), so we must decide
        # from config and never probe the endpoint.
        self.hicache_storage_enabled = bool(hicache_storage_enabled)
        self.attempts = max(1, int(attempts))
        self.backoff = max(0.0, float(backoff))
        self.backoff_max = max(0.0, float(backoff_max))
        self.max_continuations = max(1, int(max_continuations))
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
        timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=None)
        # SGLang endpoints are local / in-cluster hosts. trust_env=False avoids
        # both needless proxy use and httpx choking on IPv6 CIDRs (e.g. ::1,
        # fe80::/10) that the login environment puts in NO_PROXY
        # (httpx.InvalidURL: Invalid port ':').
        self.client = httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False)

    def _next_endpoint(self) -> str:
        with self._endpoint_lock:
            return next(self.endpoints)

    async def _post(self, url: str, payload: dict[str, Any], attempts: int) -> httpx.Response:
        last_exc: Exception | None = None
        for index in range(max(1, int(attempts))):
            try:
                response = await self.client.post(url, json=payload)
                if response.status_code != 200:
                    raise httpx.HTTPError(
                        f"request failed with status {response.status_code}"
                    )
                return response
            except Exception as exc:  # noqa: BLE001 - transport failures are retryable
                last_exc = exc
                if index + 1 < max(1, int(attempts)):
                    delay = min(self.backoff * (index + 1), self.backoff_max)
                    if index == 0 or (index + 1) % 10 == 0:
                        logger.warning(
                            "SGLang request {} failed (attempt {}/{}): {}; retrying in {:.1f}s",
                            url,
                            index + 1,
                            attempts,
                            exc,
                            delay,
                        )
                    await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _finish_type(meta_info: dict[str, Any]) -> str | None:
        reason = meta_info.get("finish_reason") or {}
        return reason.get("type") if isinstance(reason, dict) else None

    async def _generate_once(
        self,
        input_ids: Iterable[int],
        sampling_params: dict[str, Any] | None = None,
        *,
        attempts: int | None = None,
    ) -> _GenerationChunk:
        endpoint = self._next_endpoint()
        response = await self._post(
            f"{endpoint}/generate",
            {
                "input_ids": list(input_ids),
                "return_logprob": True,
                "sampling_params": self.sampling_params
                if sampling_params is None
                else dict(sampling_params),
            },
            self.attempts if attempts is None else attempts,
        )
        meta = response.json().get("meta_info") or {}
        rows = meta.get("output_token_logprobs") or []
        tokens = [int(token_id) for _, token_id, _ in rows]
        logprobs = [float(logprob) for logprob, _, _ in rows]
        bad = [i for i, lp in enumerate(logprobs) if not math.isfinite(lp)]
        if bad:
            # A non-finite behaviour-policy log-prob would poison the PPO
            # ratio of the whole sample; fail the request (the caller's retry /
            # group-drop path handles it) instead of shipping NaN to training.
            raise RuntimeError(
                f"SGLang returned {len(bad)} non-finite output logprob(s) "
                f"(first at output position {bad[0]}: {logprobs[bad[0]]!r})"
            )
        # An abort is a valid HTTP response, but it is not a completed sample.
        finish_reason = self._finish_type(meta)
        finished = finish_reason in ("stop", "length")
        return _GenerationChunk(
            tokens=tokens, logprobs=logprobs, finished=finished, finish_reason=finish_reason
        )

    async def generate(
        self,
        input_ids: Iterable[int],
        sampling_params: dict[str, Any] | None = None,
        *,
        attempts: int | None = None,
    ) -> Generation:
        """Generate until SGLang reports a terminal finish reason.

        If colocate training aborts a request, SGLang returns the partial output
        with ``finish_reason.type == "abort"``.  The prompt and partial output
        are resubmitted automatically after the service resumes generation;
        every such resume is counted in :attr:`Generation.continuations`.

        The result unpacks as ``(tokens, logprobs)`` and additionally carries
        the terminal ``finish_reason`` (``"length"`` means the response was
        truncated at ``max_new_tokens``).
        """
        prompt = list(input_ids)
        params = dict(self.sampling_params if sampling_params is None else sampling_params)
        result = Generation()
        original_budget = params.get("max_new_tokens")
        budget = int(original_budget) if original_budget is not None else None
        for continuation in range(self.max_continuations + 1):
            if budget is not None:
                remaining = budget - len(result.tokens)
                if remaining <= 0:
                    # The abort landed exactly on the budget boundary: the
                    # response is as long as ``max_new_tokens`` allows.
                    result.finish_reason = "length"
                    break
                request_params = dict(params)
                request_params["max_new_tokens"] = remaining
            else:
                request_params = params
            chunk = await self._generate_once(
                prompt + result.tokens,
                request_params,
                attempts=attempts,
            )
            result.tokens.extend(chunk.tokens)
            result.logprobs.extend(chunk.logprobs)
            if chunk.finished:
                result.finish_reason = chunk.finish_reason
                return result
            if continuation >= self.max_continuations:
                raise RuntimeError(
                    "SGLang generation did not finish after "
                    f"{self.max_continuations + 1} requests"
                )
            result.continuations += 1
        return result

    @property
    def _endpoint_values(self) -> tuple[str, ...]:
        return self._cycle_values

    def _management(self, method: str, path: str, *, timeout: float = 600.0, **kwargs: Any) -> Any:
        import requests

        last_exc: Exception | None = None
        for endpoint in self._endpoint_values:
            try:
                request_fn = getattr(requests, method.lower())
                response = request_fn(f"{endpoint}{path}", timeout=timeout, **kwargs)
                response.raise_for_status()
            except requests.RequestException as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc

    def wait_ready(self, *, liveness: Callable[[], None] | None = None, timeout: float = 1800.0) -> None:
        deadline = time.monotonic() + timeout
        import requests

        while time.monotonic() < deadline:
            if liveness is not None:
                liveness()
            try:
                if all(
                    requests.get(f"{endpoint}/health_generate", timeout=60).status_code == 200
                    for endpoint in self._endpoint_values
                ):
                    return
            except requests.RequestException:
                pass
            time.sleep(2.0)
        raise TimeoutError(f"SGLang endpoints not ready within {timeout}s: {self._endpoint_values}")

    def wait_until_idle(self, *, timeout: float = 300.0, interval: float = 0.5) -> None:
        import requests

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                idle = True
                for endpoint in self._endpoint_values:
                    response = requests.get(f"{endpoint}/v1/loads", timeout=10)
                    response.raise_for_status()
                    loads = response.json().get("loads") or []
                    idle &= sum(int(x.get("num_running_reqs", 0)) + int(x.get("num_waiting_reqs", 0)) for x in loads) == 0
                if idle:
                    return
            except requests.RequestException:
                pass
            time.sleep(interval)
        raise TimeoutError(f"SGLang endpoints did not become idle within {timeout}s")

    def pause_generation(self) -> None:
        self._management("POST", "/pause_generation", json={"mode": "abort"})

    def continue_generation(self) -> None:
        # Recent SGLang versions validate a request body for this endpoint.
        self._management("POST", "/continue_generation", json={})

    def release_memory(self, tags: Iterable[str] = ("kv_cache", "weights")) -> None:
        self._management("POST", "/release_memory_occupation", json={"tags": list(tags)})

    def resume_memory(self, tags: Iterable[str] = ("kv_cache", "weights")) -> None:
        self._management("POST", "/resume_memory_occupation", json={"tags": list(tags)})

    def load_weights(self, model_path: str) -> None:
        self._management("POST", "/update_weights_from_disk", json={"model_path": model_path}, timeout=1800.0)
        # Weight updates invalidate KV computed under the old policy. SGLang's
        # flush_cache (on by default) only resets the GPU radix/host pool; it
        # does NOT touch the L3 storage backend, whose keys are token-only and
        # carry no weight version. Without this, a same-prompt rollout on the
        # new weights would read stale L3 KV and silently diverge from the
        # recorded logprobs. Only call when this server actually has L3.
        self.clear_hicache_storage()

    def clear_hicache_storage(self) -> None:
        """Invalidate the HiCache L3 backend after a weight swap.

        No-op unless the server was configured with an L3 backend. When it is,
        a clear failure is fatal (stale KV would corrupt the next rollout).
        """
        if not self.hicache_storage_enabled:
            return
        import requests

        for endpoint in self._endpoint_values:
            requests.post(
                f"{endpoint}/clear_hicache_storage_backend", timeout=300.0
            ).raise_for_status()
            logger.info("SGLang {} cleared HiCache L3 after weight update", endpoint)

    def _sm70_colocate_tags(self) -> tuple[str, ...] | None:
        """Release tags that also pause the CUDA-graph pool on sm70, else None.

        On V100 the decode graph pool is registered with the torch memory saver;
        omitting its tag leaves ~1 GB pinned after release and OOMs the colocate
        trainer. On any other GPU this returns None and the standard
        (kv_cache, weights) tags are used.
        """
        try:
            import torch

            if not (torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (7, 0)):
                return None
            from meshy.backend.sglang_sm70 import release_tags

            return release_tags()
        except Exception:
            return None

    def release_for_colocate(self) -> None:
        import time

        t0 = time.monotonic()
        self.pause_generation()
        t_pause = time.monotonic()
        self.wait_until_idle()
        t_idle = time.monotonic()
        tags = self._sm70_colocate_tags() or ("kv_cache", "weights")
        self.release_memory(tags)
        t_release = time.monotonic()
        logger.info(
            "SGLang release_for_colocate: pause {:.2f}s wait_idle {:.2f}s "
            "release_memory {} {:.2f}s (total {:.2f}s)",
            t_pause - t0, t_idle - t_pause, tags, t_release - t_idle,
            t_release - t0,
        )

    def restore_for_colocate(self, weights_path: str) -> None:
        import time

        t0 = time.monotonic()
        tags = self._sm70_colocate_tags()
        if tags is None:
            self.resume_memory(("weights",))
            self.load_weights(weights_path)
            self.resume_memory(("kv_cache",))
        else:
            # sm70 graph resume: graph, weights, kv_cache order; weights load
            # after the weights region is resumed (as in the standard path).
            if "cuda_graph" in tags:
                self.resume_memory(("cuda_graph",))
            self.resume_memory(("weights",))
            self.load_weights(weights_path)
            self.resume_memory(("kv_cache",))
        t_weights = time.monotonic()
        self.continue_generation()
        logger.info(
            "SGLang restore_for_colocate: resume+load_weights {:.2f}s "
            "continue_generation {:.2f}s (total {:.2f}s)",
            t_weights - t0, time.monotonic() - t_weights,
            time.monotonic() - t0,
        )

    def on_colocate_release(self, target: str) -> None:
        del target
        self.release_for_colocate()

    def on_colocate_acquire(self, grant: Any) -> None:
        weights_path = getattr(grant, "payload_ref", None)
        # Genesis has no trainer checkpoint and the server was booted from the
        # configured base model. Every later grant must carry the checkpoint
        # produced by the training step that just released the GPU; falling
        # back to ``model_path`` there would silently restore stale weights.
        if not weights_path and int(getattr(grant, "sequence", 0)) == 0:
            weights_path = self.model_path
        if not weights_path:
            raise RuntimeError(
                "SGLang colocate acquire requires the latest checkpoint path "
                "in grant.payload_ref"
            )
        logger.info("SGLang restoring colocate weights from {}", weights_path)
        self.restore_for_colocate(weights_path)

    async def close(self) -> None:
        await self.client.aclose()
