"""Gate-driven GRPO rollout Worker."""

from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
import json
import math
import os
import time
from typing import Any, Callable

from loguru import logger

from meshy.config import GRPO_TRAINER_FIELDS
from meshy.engine.sglang import Generation, SGLangEngine
from meshy.utils.metric import has_repetition
from meshy.worker.tq import TQInput, TQOutput, TQWorker
from meshy.service.colocation import ColocationManager, RequestKind

#: Columns written per sample. Must equal what the trainer fetches
#: (``meshy.config.GRPO_TRAINER_FIELDS``): the fetch is an AND-filter.
GRPO_FIELDS = tuple(GRPO_TRAINER_FIELDS)

#: The same columns minus ``advantage``, written when a critic Service owns the
#: advantage instead. Because the trainer's fetch is an AND-filter, leaving the
#: column unwritten is exactly what holds each row back until the critic fills
#: it in -- no extra column and no schema change on either side. See
#: :class:`meshy.worker.critic.CriticWorker`.
GRPO_FIELDS_NO_ADVANTAGE = tuple(f for f in GRPO_FIELDS if f != "advantage")

#: Added when reward shaping is configured: ``reward`` then carries the shaped
#: ``R`` and this carries the unshaped task reward (:mod:`meshy.reward`).
RAW_REWARD_FIELD = "raw_reward"


def resolve_callable(value: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(value):
        return value
    if not isinstance(value, str) or ":" not in value:
        raise TypeError("callable config must be a callable or 'module:attribute' path")
    module_name, attr_path = value.split(":", 1)
    obj: Any = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    if not callable(obj):
        raise TypeError(f"configured object {value!r} is not callable")
    return obj


def grpo_advantage(samples: list[Any]) -> None:
    import torch

    if len(samples) == 1:
        samples[0].advantage = samples[0].reward
        return
    rewards = torch.tensor([sample.reward for sample in samples], dtype=torch.float32)
    advantages = (rewards - rewards.mean()) / rewards.std().clamp(min=1e-8)
    for sample, advantage in zip(samples, advantages):
        sample.advantage = float(advantage.item())


def is_zero_variance_group(samples: list[Any], *, tolerance: float = 1e-8) -> bool:
    """Return whether all raw rewards in a completed group are equal.

    Deliberately the *raw* reward, not the shaped one: the question dynamic
    sampling asks is whether the group's outcomes were informative about the
    task. Length shaping perturbs every sample a little, so keying on the
    shaped reward would report an all-correct group as having variance and
    defeat the filter.
    """
    if not samples:
        return True
    rewards = [
        float(sample.reward if getattr(sample, "raw_reward", None) is None else sample.raw_reward)
        for sample in samples
    ]
    return max(rewards) - min(rewards) <= tolerance


def sample_to_tensordict(sample: Any, weight_version: int):
    import torch
    from tensordict import TensorDict

    raw_reward = getattr(sample, "raw_reward", None)
    return TensorDict(
        {
            "tokens": torch.tensor(list(sample.tokens), dtype=torch.long),
            "logprobs": torch.tensor(list(sample.logprobs), dtype=torch.float32),
            "mask_assistant": torch.tensor(list(sample.masks), dtype=torch.float32),
            # ``None`` when a critic Service owns the advantage: the column is
            # excluded from the write (GRPO_FIELDS_NO_ADVANTAGE), so this
            # placeholder never reaches TQ.
            "advantage": torch.tensor(
                0.0 if sample.advantage is None else float(sample.advantage),
                dtype=torch.float32,
            ),
            "weight_version": torch.tensor(int(weight_version), dtype=torch.int64),
            "reward": torch.tensor(float(sample.reward), dtype=torch.float32),
            # Equal to ``reward`` when no shaping is configured; the column is
            # only written when the worker declares it.
            RAW_REWARD_FIELD: torch.tensor(
                float(sample.reward if raw_reward is None else raw_reward),
                dtype=torch.float32,
            ),
            "truncated": torch.tensor(int(bool(getattr(sample, "truncated", False))), dtype=torch.int64),
            "repetition": torch.tensor(int(bool(getattr(sample, "repetition", False))), dtype=torch.int64),
            "mixed_version": torch.tensor(
                int(bool(getattr(sample, "mixed_version", False))), dtype=torch.int64
            ),
        },
        batch_size=[],
    )


def _scalar(value: Any) -> int:
    if getattr(value, "is_nested", False):
        value = value[0]
    return int(value.reshape(-1)[0].item())


class TrajectoryLogger:
    """Per-sample audit log for rollout windows.

    Two retention tiers (both append-only, no rewrite — a crash never loses a
    window):

    * **slim** (``XRL_TRAJ_SLIM=1``, default): every sample is logged with all
      scalar/tag fields (round, weight_version, reward, advantage, truncation
      flags, response_tokens, ground_truth, finish_reason, ...) but WITHOUT the
      full ``trajectory`` dialogue. That dialogue is ~95% of a record's bytes
      and is not needed for the quantitative audits (batch counts, crash-resend
      vs resampling). 40 windows x 512 x ~70 B ≈ 1.4 MB/run.
    * **full sample** (``XRL_TRAJ_FULL_SAMPLES``, default 16): that many
      samples per window additionally keep the full ``trajectory`` (flagged
      ``"full": true``), chosen deterministically to span the reward range for
      qualitative inspection. ~1 MB/run.

    ``XRL_TRAJ_KEEP_LINES`` remains as a backstop: if >0 the file is rolled to
    its last N records (default 0 = off; with slim logging the bound is so low
    this should not be needed).
    """

    def __init__(self, path: str, *, verbose: bool = False) -> None:
        self.path = os.path.abspath(path)
        self.verbose = bool(verbose)
        self._lock = asyncio.Lock()
        self.slim = os.environ.get("XRL_TRAJ_SLIM", "1") == "1"
        self.full_samples = int(os.environ.get("XRL_TRAJ_FULL_SAMPLES", "16"))
        self.keep_lines = int(os.environ.get("XRL_TRAJ_KEEP_LINES", "0"))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        logger.info(
            "RolloutWorker: trajectory logging -> {} (verbose={}, slim={}, "
            "full_samples/window={}, keep_lines={})",
            self.path, self.verbose, self.slim, self.full_samples, self.keep_lines,
        )

    def _scalar_fields(self, sample: Any, version: int) -> dict[str, Any]:
        """Everything except the full dialogue; present for every sample."""
        return {
            "timestamp": time.time(),
            # Historical trajectory consumers group samples by training round.
            # The rollout version is the exact source value used by TQ; keep
            # both fields so newer consumers can use the staleness tag directly.
            "round": version + 1,
            "weight_version": version,
            "ground_truth": sample.ground_truth,
            "reward": sample.reward,
            "advantage": sample.advantage,
            "finish_reason": getattr(sample, "finish_reason", None),
            "truncated": bool(getattr(sample, "truncated", False)),
            "repetition": bool(getattr(sample, "repetition", False)),
            "mixed_version": bool(getattr(sample, "mixed_version", False)),
            "response_tokens": sum(1 for m in sample.masks if m),
        }

    def _full_indices(self, samples: list[Any]) -> set[int]:
        """Deterministically pick up to full_samples indices spanning the
        reward distribution (high/mid/low), so qualitative checks cover both
        successes and failures. Stable across crashes: depends only on order."""
        n = self.full_samples
        if n <= 0:
            return set()
        if len(samples) <= n:
            return set(range(len(samples)))
        # Rank by reward (stable on ties via index), then take evenly spaced
        # ranks so the sample spans the full reward range rather than only the
        # top.
        order = sorted(range(len(samples)),
                       key=lambda i: (float(samples[i].reward), i))
        if n == 1:
            return {order[len(order) // 2]}
        step = (len(order) - 1) / (n - 1)
        return {order[round(k * step)] for k in range(n)}

    def _record(self, sample: Any, version: int, *,
                full: bool, with_dialogue: bool) -> dict[str, Any]:
        record = self._scalar_fields(sample, version)
        if full:
            # Flag marks a slim-run sampled full dialogue.
            record["full"] = True
        if with_dialogue:
            record["trajectory"] = list(sample.messages)
        if self.verbose:
            record.update(
                num_tokens=len(list(sample.tokens)),
                tokens=list(sample.tokens),
                logprobs=list(sample.logprobs),
                masks=list(sample.masks),
            )
        return record

    async def write(self, samples: list[Any], version: int) -> None:
        full_idx = self._full_indices(samples) if self.slim else set(range(len(samples)))
        lines = []
        for i, sample in enumerate(samples):
            is_full_sample = i in full_idx
            if self.slim:
                # slim: only sampled records carry the dialogue, flagged full.
                lines.append(json.dumps(
                    self._record(sample, version,
                                 full=is_full_sample, with_dialogue=is_full_sample),
                    ensure_ascii=False, default=str,
                ) + "\n")
            else:
                # legacy full logging for every sample, no slim "full" flag.
                lines.append(json.dumps(
                    self._record(sample, version,
                                 full=False, with_dialogue=True),
                    ensure_ascii=False, default=str,
                ) + "\n")
        blob = "".join(lines)
        async with self._lock:
            await asyncio.to_thread(self._append, blob)

    def _tail_lines(self, n: int) -> list[str]:
        """Return the last n newline-terminated records without reading the
        whole (multi-GB) file: scan backwards in 1 MiB blocks from the end."""
        block = 1 << 20
        with open(self.path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            pos = fh.tell()
            chunks: list[bytes] = []
            nl = 0
            while pos > 0 and nl <= n:
                read = min(block, pos)
                pos -= read
                fh.seek(pos)
                data = fh.read(read)
                chunks.append(data)
                nl += data.count(b"\n")
            tail = b"".join(reversed(chunks))
        lines = tail.splitlines(keepends=True)
        # Unless the scan started at the file head, the first line is a partial
        # record straddling a block boundary — drop it.
        if pos > 0 and lines:
            lines = lines[1:]
        return [ln.decode("utf-8", errors="replace") for ln in lines[-n:]]

    def _append(self, blob: str) -> None:
        if self.keep_lines <= 0:
            with open(self.path, "a", encoding="utf-8") as output:
                output.write(blob)
            return
        # Backstop rolling retention: keep only the last keep_lines records
        # across the existing file plus this window's records.
        new_lines = [ln + "\n" for ln in blob.splitlines()]
        spare = self.keep_lines - len(new_lines)
        kept = (
            self._tail_lines(spare)
            if spare > 0 and os.path.exists(self.path)
            else []
        )
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as output:
            output.writelines(kept)
            output.writelines(new_lines)
        os.replace(tmp, self.path)



def append_window_stats(path: str | None, record: dict[str, Any]) -> None:
    """Append one per-window rollout statistic line (best-effort, low rate).

    The trainer correlates the line by ``weight_version`` and surfaces the
    scalars (filtered_ratio, refill/drop counts) in its log + TensorBoard.
    """
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("RolloutWorker: failed to write window stats {}", path)


class RolloutWorker(TQWorker):
    """Read gen gates, generate grouped trajectories, and append them to TQ."""

    def __init__(
        self,
        *,
        engine: SGLangEngine,
        endpoints_ref: str,
        model_path: str,
        dataset: str | Callable[..., Any],
        dataset_kwargs: dict[str, Any] | None,
        partition_id: str,
        group_size: int,
        train_batch_size: int,
        sampling_params: dict[str, Any] | None,
        reward: str | Callable[[Any], float],
        reward_shaping: str | Callable[..., float] | None = None,
        reward_shaping_kwargs: dict[str, Any] | None = None,
        advantage: str | Callable[[list[Any]], Any] | None = None,
        advantage_kwargs: dict[str, Any] | None = None,
        filter_zero_std_groups: bool = False,
        oversample_factor: float = 1.0,
        dynamic_sampling: bool = False,
        dynamic_target_groups: int | None = None,
        dynamic_max_prompts: int | None = None,
        num_epochs: int = 1,
        pacing_window: int | str | None = 1,
        max_running_requests: int = -1,
        poll_interval: float = 2.0,
        trajectory_log: str | None = None,
        verbose_trajectory_log: bool = False,
        window_stats_log: str | None = None,
        external_advantage: bool = False,
        version_hook: str | Callable[..., Any] | None = None,
        version_hook_kwargs: dict[str, Any] | None = None,
        client_factory=None,
        colocation: ColocationManager | None = None,
    ) -> None:
        super().__init__()
        if group_size <= 0 or train_batch_size <= 0:
            raise ValueError("group_size and train_batch_size must be positive")
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if pacing_window == "auto":
            pacing_window = 1
        if pacing_window is not None and int(pacing_window) < 1:
            raise ValueError("pacing_window must be >= 1, 'auto', or None")
        # NOTE: the partial_rollout feature is not in this integration tree, so
        # env's partial_rollout+dynamic_sampling mutual-exclusion guard is
        # intentionally omitted here. Re-add it if/when partial rollout lands.
        self.engine = engine
        self.colocation = colocation
        self._colocation_request = None
        self.model_path = model_path
        self.dataset_factory = resolve_callable(dataset)
        self.dataset_kwargs = dict(dataset_kwargs or {})
        self.group_size = int(group_size)
        self.train_batch_size = int(train_batch_size)
        self.sampling_params = dict(sampling_params or engine.sampling_params)
        self.reward_fn = resolve_callable(reward)
        # Shaping is part of the reward, so it is applied here rather than in
        # the advantage pipeline -- which ``external_advantage`` skips
        # entirely. See :mod:`meshy.reward`.
        self.reward_shaping_fn = (
            resolve_callable(reward_shaping) if reward_shaping is not None else None
        )
        self.reward_shaping_kwargs = dict(reward_shaping_kwargs or {})
        # A critic Service computes the advantage from its own value net, so
        # the rollout must neither compute nor publish the column: writing a
        # group-normalised placeholder would satisfy the trainer's AND-filter
        # and let rows through before the critic ever saw them.
        self.external_advantage = bool(external_advantage)
        if self.external_advantage:
            if advantage is not None:
                raise ValueError(
                    "external_advantage=True hands the advantage to a critic "
                    "Service; an inline advantage callable would be ignored"
                )
            self.advantage_fn = None
            fields = GRPO_FIELDS_NO_ADVANTAGE
        else:
            self.advantage_fn = grpo_advantage if advantage is None else resolve_callable(advantage)
            fields = GRPO_FIELDS
        # Declared only when there is something to distinguish it from
        # ``reward``: the trainer's fetch is an AND-filter, so a column no
        # consumer asked for is harmless but one nobody writes is fatal.
        if self.reward_shaping_fn is not None:
            fields = fields + (RAW_REWARD_FIELD,)
        self.rollout_fields = fields
        self.advantage_kwargs = dict(advantage_kwargs or {})
        self.dynamic_sampling = bool(dynamic_sampling)
        # Dynamic sampling and the legacy per-epoch filter both call
        # is_zero_variance_group but enforce different refill/cap semantics;
        # they never run together (dynamic drives its own window loop).
        self.filter_zero_std_groups = bool(filter_zero_std_groups) and not self.dynamic_sampling
        groups_per_window, rem = divmod(self.train_batch_size, self.group_size)
        if rem:
            raise ValueError(
                "train_batch_size must be a multiple of group_size "
                f"({self.train_batch_size} % {self.group_size} = {rem})"
            )
        self.dynamic_target_groups = (
            int(dynamic_target_groups) if dynamic_target_groups is not None else groups_per_window
        )
        if self.dynamic_target_groups <= 0:
            raise ValueError("dynamic_target_groups must be positive")
        # Hard ceiling on prompts drawn in one window (kept + dropped); default
        # 3x the target. Once reached the worker keeps zero-variance groups so
        # the window still fills and the trainer is never starved.
        self.dynamic_max_prompts = (
            int(dynamic_max_prompts)
            if dynamic_max_prompts is not None
            else 3 * self.dynamic_target_groups
        )
        if self.dynamic_max_prompts < self.dynamic_target_groups:
            raise ValueError("dynamic_max_prompts must be >= dynamic_target_groups")
        if oversample_factor < 1.0:
            raise ValueError("oversample_factor must be >= 1")
        # Replacement budget for groups dropped by ``filter_zero_std_groups``,
        # in multiples of the dataset. See :meth:`_run_rollouts`.
        self.oversample_factor = float(oversample_factor)
        self.groups_seen = 0
        self.groups_filtered = 0
        # Dynamic-sampling per-window accounting (logged at each window close).
        self.dynamic_windows = 0
        self.dynamic_prompts_drawn_total = 0
        self.dynamic_groups_dropped_total = 0
        self.num_epochs = int(num_epochs)
        self.pacing_window = None if pacing_window is None else int(pacing_window)
        self.max_running_requests = int(max_running_requests)
        self.poll_interval = float(poll_interval)
        self.gates_seen = 0
        self.weight_version = 0
        self.samples_started = 0
        self.window_stats_log = window_stats_log
        if self.window_stats_log:
            os.makedirs(os.path.dirname(os.path.abspath(self.window_stats_log)), exist_ok=True)
        self.trajectory_logger = (
            TrajectoryLogger(trajectory_log, verbose=verbose_trajectory_log)
            if trajectory_log
            else None
        )
        self.version_hook_fn = (
            resolve_callable(version_hook) if version_hook is not None else None
        )
        self.version_hook_kwargs = dict(version_hook_kwargs or {})
        self._hooks_run: set[int] = set()
        self._version_hook_lock = asyncio.Lock()

        gen_gate_fields = ("gate_step", "weight_version")
        gen_gate_partition = "gen_gate"
        gen_gate_task = "gen_gate"

        self.configure_tq(
            endpoints_ref=endpoints_ref,
            input=None,
            controls={
                "gen_gate": TQInput(
                    partition=gen_gate_partition,
                    fields=gen_gate_fields,
                    batch_size=1,
                    consumer=gen_gate_task,
                    clear_after_success=True,
                )
            },
            outputs={
                "rollouts": TQOutput(
                    fields=self.rollout_fields,
                    new_rows=True,
                    partition=partition_id,
                )
            },
            poll_interval=poll_interval,
            client_factory=client_factory,
        )

    def run(self) -> None:
        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        await self.open_tq()
        try:
            if self.dynamic_sampling:
                await self._run_dynamic_rollouts()
            else:
                await self._run_rollouts()
        finally:
            await self.close_tq()

    def _generation_budget(self) -> int:
        if self.gates_seen == 0:
            return 0
        assert self.pacing_window is not None
        return (self.gates_seen - 1 + self.pacing_window) * self.train_batch_size

    async def try_advance_gate(self) -> bool:
        data = await self.read_tq_control("gen_gate")
        if data is None:
            return False
        step = _scalar(data["gate_step"])
        version = _scalar(data["weight_version"])
        expected = self.gates_seen
        if step != expected:
            logger.warning(
                "gen_gate stream out of order: expected {}, got {} (weight v{})",
                expected,
                step,
                version,
            )
        previous = self.weight_version
        self.gates_seen += 1
        self.weight_version = version
        logger.info(
            "RolloutWorker: gate {} up, version {} -> {} ({} samples started)",
            step,
            previous,
            version,
            self.samples_started,
        )
        return True

    async def drain_gates(self) -> None:
        while await self.try_advance_gate():
            pass

    async def acquire_generation_slot(self, num_samples: int) -> int:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        # The initial gate establishes the version that inference is allowed to
        # use.  An ungated pacing policy may run ahead only after that gate.
        while self.gates_seen == 0:
            if self.stopped:
                raise asyncio.CancelledError
            if not await self.try_advance_gate():
                await asyncio.sleep(min(self.poll_interval, 2.0))
        if self.pacing_window is None:
            await self.drain_gates()
            self.samples_started += num_samples
            return self.weight_version
        while self.samples_started + num_samples > self._generation_budget():
            if self.stopped:
                raise asyncio.CancelledError
            if not await self.try_advance_gate():
                await asyncio.sleep(min(self.poll_interval, 2.0))
        self.samples_started += num_samples
        return self.weight_version

    async def _ensure_colocation(self) -> None:
        """Ensure the fallback rollout role currently owns the GPU token."""
        if self.colocation is None or self.colocation.owns_gpu:
            return
        if self._colocation_request is None:
            request_id = f"rollout:fallback:{self.gates_seen}"
            self._colocation_request = self.colocation.request_gpu(
                request_id=request_id, kind=RequestKind.FALLBACK
            )
        await asyncio.to_thread(self.colocation.wait_for_grant, self._colocation_request)

    async def _run_version_hook(self, version: int) -> None:
        """Run the version hook exactly once per weight version.

        The caller is the first group of a new GPU grant, so the engine has
        already resumed and loaded ``version`` and the trainer is blocked
        waiting for this window's data. Other groups of the same version wait
        on the lock rather than skipping ahead.
        """
        if self.version_hook_fn is None or version in self._hooks_run:
            return
        async with self._version_hook_lock:
            if version in self._hooks_run:
                return
            result = self.version_hook_fn(
                version=version,
                engine=self.engine,
                model_path=self.model_path,
                **self.version_hook_kwargs,
            )
            if inspect.isawaitable(result):
                await result
            self._hooks_run.add(version)

    async def rollout_group(self, builder: Any, prompt: Any, weight_version: int) -> list[Any] | None:
        await self._ensure_colocation()
        await self._run_version_hook(weight_version)
        group = [copy.deepcopy(prompt) for _ in range(self.group_size)]

        async def rollout_one(sample: Any) -> Any:
            generation = await self.engine.generate(
                list(sample.tokens),
                sampling_params=self.sampling_params,
            )
            tokens, logprobs = generation
            builder.append_tokens(sample, "assistant", tokens, logprobs)
            self._stamp_sample(sample, generation)
            reward = float(self.reward_fn(sample))
            if self.reward_shaping_fn is not None:
                # ``reward`` becomes the shaped R the critic and the GAE
                # recursion consume; the unshaped value rides along so the
                # solve-rate metrics stay about correctness.
                sample.raw_reward = reward
                sample.reward = float(
                    self.reward_shaping_fn(reward, sample, **self.reward_shaping_kwargs)
                )
            else:
                sample.reward = reward
            return sample

        group = await asyncio.gather(*(rollout_one(sample) for sample in group))
        self.groups_seen += 1
        if self.filter_zero_std_groups:
            rewards = [float(sample.reward) for sample in group]
            if is_zero_variance_group(group):
                self.groups_filtered += 1
                logger.debug(
                    "RolloutWorker: dropping zero-variance group {} (reward={})",
                    self.groups_seen,
                    rewards[0] if rewards else None,
                )
                return None
        if self.advantage_fn is not None:
            advantages = self.advantage_fn(group, **self.advantage_kwargs)
            if advantages is not None:
                values = list(advantages)
                if len(values) != len(group):
                    raise ValueError("advantage function result length must match rollout group")
                for sample, value in zip(group, values):
                    sample.advantage = float(value)
        return group

    @staticmethod
    def _stamp_sample(sample: Any, generation: Any) -> None:
        """Record the per-sample quality flags the trainer turns into metrics.

        ``generation`` is normally an :class:`~meshy.engine.sglang.Generation`;
        a bare ``(tokens, logprobs)`` tuple (legacy engines, test doubles)
        leaves the flags at their defaults.
        """
        finish_reason = getattr(generation, "finish_reason", None)
        sample.finish_reason = finish_reason
        # SGLang reports ``length`` when the response hit ``max_new_tokens``.
        sample.truncated = bool(
            getattr(generation, "truncated", finish_reason == "length")
        )
        # Every abort/resume cycle happened because a colocated training step
        # took the card; the resumed tokens come from the *next* weights.
        sample.mixed_version = int(getattr(generation, "continuations", 0) or 0) > 0
        response = sample.messages[-1]["content"] if sample.messages else ""
        sample.repetition = has_repetition(response)

    async def _write_rollout_group(self, group: list[Any], version: int) -> None:
        from meshy.transferqueue import adapter

        rows = [sample_to_tensordict(sample, version) for sample in group]
        data = adapter.samples_to_td(rows, self.rollout_fields)
        if self.trajectory_logger is not None:
            await self.trajectory_logger.write(group, version)
        await self.write_tq_output("rollouts", data)

    def _oversample_budget(self, dataset_prompts: int | None) -> int | None:
        """Extra prompts this epoch may spend replacing filtered groups.

        ``filter_zero_std_groups`` drops a group after it has been generated,
        so without replacement an epoch that filters heavily yields fewer
        windows than it has prompts for. Replacement here is *implicit* rather
        than a refill buffer: the rollout does not drain a window before
        starting the next (``pacing_window=None``), and every consumer takes a
        fixed row count off TQ, so a dropped group is simply made up by the
        prompts already in flight behind it. All this budget does is bound how
        far the epoch may over-run its nominal length -- which is what stops a
        stretch of uniformly-solved prompts from burning the dataset.

        ``None`` when no bound applies: the filter is off, no replacement was
        asked for, or the dataset cannot report its own size (in which case
        there is nothing to take a fraction of, and an unbounded epoch is the
        pre-existing behaviour rather than a new risk).
        """
        if not self.filter_zero_std_groups or self.oversample_factor <= 1.0:
            return None
        if not dataset_prompts:
            logger.warning(
                "RolloutWorker: oversample_factor={} ignored -- the dataset does "
                "not expose ``n_prompts``",
                self.oversample_factor,
            )
            return None
        return int(dataset_prompts * (self.oversample_factor - 1.0))

    async def _rollout_groups_concurrent(
        self, builder: Any, prompts: list[Any], semaphore: Any
    ) -> list[list[Any]]:
        """Generate one group per prompt, concurrently, in prompt order.

        Every group is returned (the legacy filter is off on this path); the
        caller decides whether a zero-variance group is kept or replaced.
        """

        async def one(prompt: Any) -> tuple[int, list[Any]]:
            version = await self.acquire_generation_slot(self.group_size)
            if semaphore is not None:
                await semaphore.acquire()
            try:
                group = await self.rollout_group(builder, prompt, version)
                return version, group
            finally:
                if semaphore is not None:
                    semaphore.release()

        return await asyncio.gather(*(one(p) for p in prompts))

    def _refund_generation_slots(self, num_groups: int) -> None:
        """Return pacing slots for groups that will never become training rows.

        A dropped zero-variance group generated no data, so the samples it
        reserved must not count against the gate budget -- otherwise the
        replacement prompts could not be launched and the window would come out
        short, starving the trainer under ``pacing_window=1``.
        """
        self.samples_started = max(0, self.samples_started - num_groups * self.group_size)

    async def _run_dynamic_rollouts(self) -> None:
        """Per-window DAPO dynamic sampling.

        Each window draws until it holds exactly ``dynamic_target_groups``
        informative groups. A zero-variance group is dropped and its pacing
        slots refunded so replacement prompts launch inside the same gate. The
        anti-death-loop bound is a replacement budget of
        ``dynamic_max_prompts - target`` dropped groups per window (3x target at
        the default): once spent, zero-variance groups are kept and the window
        fills unfiltered, so the trainer always receives a full window while
        data remains and filtering can never spin forever.
        """
        from meshy.utils.sample import SampleBuilder

        builder = SampleBuilder(self.model_path)
        max_groups = (
            max(1, math.ceil(self.max_running_requests / self.group_size))
            if self.max_running_requests > 0
            else None
        )
        semaphore = asyncio.Semaphore(max_groups) if max_groups is not None else None
        replacement_budget = self.dynamic_max_prompts - self.dynamic_target_groups

        for epoch in range(self.num_epochs):
            if self.stopped:
                break
            dataset = self.dataset_factory(**self.dataset_kwargs)
            while not self.stopped:
                kept: list[tuple[int, list[Any]]] = []
                window_dropped = 0
                window_drawn = 0
                first_draw = True
                while len(kept) < self.dynamic_target_groups:
                    deficit = self.dynamic_target_groups - len(kept)
                    prompts = dataset.take_prompts(
                        builder, deficit, window_start=first_draw
                    )
                    first_draw = False
                    if not prompts:
                        break
                    results = await self._rollout_groups_concurrent(builder, prompts, semaphore)
                    results = [r for r in results if r[1] is not None]
                    window_drawn += len(results)
                    for version, group in results:
                        may_drop = window_dropped < replacement_budget
                        if may_drop and is_zero_variance_group(group):
                            window_dropped += 1
                            self.groups_filtered += 1
                            self._refund_generation_slots(1)
                            continue
                        kept.append((version, group))

                if not kept:
                    logger.info(
                        "RolloutWorker dataset exhausted (dynamic epoch {}); "
                        "{} groups filtered of {} seen",
                        epoch, self.groups_filtered, self.groups_seen,
                    )
                    from meshy.service.failfast import publish_done

                    publish_done("rollout")
                    return

                for version, group in kept:
                    await self._write_rollout_group(group, version)

                self.dynamic_windows += 1
                self.dynamic_prompts_drawn_total += window_drawn
                self.dynamic_groups_dropped_total += window_dropped
                filtered_ratio = window_dropped / window_drawn if window_drawn else 0.0
                window_version = max(v for v, _ in kept)
                append_window_stats(self.window_stats_log, {
                    "kind": "dynamic",
                    "window": self.dynamic_windows,
                    "weight_version": window_version,
                    "prompts_drawn": window_drawn,
                    "valid_groups": len(kept),
                    "valid_samples": len(kept) * self.group_size,
                    "groups_dropped_zero_variance": window_dropped,
                    "refill_count": window_dropped,
                    "filtered_ratio": round(filtered_ratio, 6),
                })
                logger.info(
                    "RolloutWorker dynamic window {}: {} valid groups / {} prompts "
                    "({} samples), dropped {} zero-variance groups, filtered_ratio={:.3f}"
                    .format(
                        self.dynamic_windows, len(kept), window_drawn,
                        len(kept) * self.group_size, window_dropped, filtered_ratio,
                    )
                )

    async def _run_rollouts(self) -> None:
        from meshy.utils.sample import SampleBuilder

        builder = SampleBuilder(self.model_path)
        max_groups = (
            max(1, math.ceil(self.max_running_requests / self.group_size))
            if self.max_running_requests > 0
            else None
        )
        semaphore = asyncio.Semaphore(max_groups) if max_groups is not None else None
        tasks: set[asyncio.Task[Any]] = set()

        async def execute(prompt: Any, version: int) -> None:
            try:
                group = await self.rollout_group(builder, prompt, version)
                if group is not None:
                    await self._write_rollout_group(group, version)
            except Exception as exc:
                logger.exception("RolloutWorker group failed: {}", exc)
            finally:
                if semaphore is not None:
                    semaphore.release()

        for epoch in range(self.num_epochs):
            if self.stopped:
                break
            dataset = self.dataset_factory(**self.dataset_kwargs)
            budget = self._oversample_budget(getattr(dataset, "n_prompts", None))
            filtered_at_epoch_start = self.groups_filtered
            exhausted = False
            while not self.stopped:
                prompts = dataset.next_batch(builder)
                if not prompts:
                    logger.info(
                        "RolloutWorker dataset exhausted (epoch {}); {} of {} groups "
                        "filtered as zero-variance",
                        epoch,
                        self.groups_filtered,
                        self.groups_seen,
                    )
                    exhausted = True
                    break
                if budget is not None:
                    spent = self.groups_filtered - filtered_at_epoch_start
                    if spent > budget:
                        logger.info(
                            "RolloutWorker: oversample budget exhausted (epoch {}): "
                            "{} groups filtered, budget {}",
                            epoch,
                            spent,
                            budget,
                        )
                        exhausted = True
                        break
                for prompt in prompts:
                    version = await self.acquire_generation_slot(self.group_size)
                    if semaphore is not None:
                        await semaphore.acquire()
                    task = asyncio.create_task(execute(prompt, version))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
            if tasks:
                # Let every in-flight group finish and land in TQ before we say
                # the producer side is done, so the trainer cannot see a
                # half-written final window.
                await asyncio.gather(*tasks)
            if exhausted:
                from meshy.service.failfast import publish_done

                logger.info("RolloutWorker finished all {} epoch(s); publishing done",
                            self.num_epochs)
                publish_done("rollout")
                break


__all__ = [
    "GRPO_FIELDS",
    "GRPO_FIELDS_NO_ADVANTAGE",
    "RAW_REWARD_FIELD",
    "RolloutWorker",
    "grpo_advantage",
    "is_zero_variance_group",
    "resolve_callable",
    "sample_to_tensordict",
]
