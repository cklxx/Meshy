"""SPMD wrapper around :class:`CriticEngine` — one replica, one command loop.

:class:`CriticEngine` is a bare compute object (model + optimiser + value
step). This module gives it the same replica shape the actor trainer has:
rank 0 originates work, the window is broadcast so every rank executes the same
collectives, and only rank 0 returns a result. It is the critic's counterpart
to :class:`meshy.engine.opd.OPDTeacherEngine`.

The `CriticEngine` is built in :meth:`setup`, not ``__init__``, so
:meth:`SpmdEngine._init_process_group` has already created the process group by
the time it runs. ``torchtitan``'s ``init_distributed`` is idempotent (it warns
and returns when a group already exists), so the inner engine attaches to that
group rather than trying to build a second one.
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

from meshy.engine.spmd import SpmdEngine


class CriticSpmdEngine(SpmdEngine):
    """A critic value net driven through the replica command loop."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        local_device_id: str,
        master_addr: str,
        master_port: int,
        runtime_root: str,
        name: str,
        model_path: str,
        trainer_config: Any,
        train_backbone: bool = True,
        lr: float = 5e-6,
        warmup_steps: int = 10,
        value_epochs: int = 1,
        gamma: float = 1.0,
        alpha: float = 1.5,
        max_norm: float = 1.0,
        micro_rows: int = 1,
        publish_mode: str = "values",
        is_colocate: bool = False,
        bind_host: str = "127.0.0.1",
        endpoint_port: int = 0,
    ) -> None:
        super().__init__(
            rank=rank,
            world_size=world_size,
            local_device_id=local_device_id,
            master_addr=master_addr,
            master_port=master_port,
            bind_host=bind_host,
            endpoint_port=endpoint_port,
            runtime_root=runtime_root,
            name=name,
        )
        self.model_path = model_path
        self.trainer_config = trainer_config
        self.train_backbone = bool(train_backbone)
        self.lr = float(lr)
        self.warmup_steps = int(warmup_steps)
        self.value_epochs = int(value_epochs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.max_norm = float(max_norm)
        #: rows per critic forward; long sequences need this small
        self.micro_rows = int(micro_rows)
        if publish_mode not in ("values", "advantage"):
            raise ValueError(
                f"publish_mode must be 'values' or 'advantage', got {publish_mode!r}"
            )
        #: ``"values"`` writes the per-token value function back to the rollout
        #: row and leaves GAE to the trainer (``enable_gae``); ``"advantage"``
        #: is the original arrangement where the critic reduces to one scalar
        #: per sequence. See ``docs/critic_engine.md``.
        self.publish_mode = publish_mode
        self.is_colocate = bool(is_colocate)
        self.critic: Any = None
        #: set by rank 0 in ``score_and_train`` just before the command is
        #: broadcast; other ranks receive it over the wire.
        self._publish = True
        #: the actor weight version these advantages were produced against;
        #: reported on the cold-start gate so the rollout keeps generating
        #: against the weights it already holds.
        self.weight_version = 0

    # ── SpmdEngine hooks ────────────────────────────────────────────────
    def setup(self) -> None:
        from torchtitan.config.configs import (
            ActivationCheckpointConfig,
            CompileConfig,
            DebugConfig,
            ParallelismConfig,
            TrainingConfig,
        )

        from meshy.backend.titan.critic.engine import CriticEngine
        from meshy.backend.titan.config import _get_model_spec

        cfg = self.trainer_config
        spec = _get_model_spec(
            cfg.model_name, cfg.model_flavor, getattr(cfg, "attn_backend", "sdpa")
        )
        parallelism = ParallelismConfig(
            data_parallel_shard_degree=cfg.dp_shard_degree,
            data_parallel_replicate_degree=cfg.dp_replicate_degree,
            tensor_parallel_degree=cfg.tp_degree,
            context_parallel_degree=cfg.cp_degree,
        )
        training = TrainingConfig(
            seq_len=cfg.seq_len,
            local_batch_size=self.micro_rows,
            global_batch_size=self.micro_rows * self.world_size,
            max_norm=self.max_norm,
            steps=cfg.steps,
            dtype=cfg.dtype,
        )
        self.critic = CriticEngine(
            spec.model,
            seq_len=cfg.seq_len,
            parallelism=parallelism,
            training=training,
            compile=CompileConfig(enable=False, components=[]),
            # Long-context critics need real activation checkpointing: the
            # retained per-layer activations scale with the CP-local sequence
            # length, and with train_backbone the backward keeps them all.
            ac_config=ActivationCheckpointConfig(
                mode=getattr(cfg, "activation_checkpoint_mode", "selective")
            ),
            debug=DebugConfig(),
            dump_folder=cfg.dump_folder,
            lr=self.lr,
            warmup_steps=self.warmup_steps,
            max_norm=self.max_norm,
            train_backbone=self.train_backbone,
            model_spec=spec,
        )
        # Cold start (recipe §2): backbone from the base model, value head
        # random. Never the actor's current weights -- the critic must not
        # inherit the policy's drift.
        n = self.critic.load_backbone(self.model_path)
        logger.info(
            "Critic {} rank {}: loaded {:,} backbone params from {}; value head random",
            self.name, self.rank, n, self.model_path,
        )
        if self.is_colocate:
            self.offload_to_cpu()

    def execute(self, payload: dict[str, Any], samples: list[Any] | None) -> Any:
        action = payload.get("action")
        if action == "score_and_train":
            return self._score_and_train_impl(samples)
        # GPU residency is broadcast like any other command so every rank moves
        # together; the colocation manager only runs on rank 0.
        if action == "colocate_acquire":
            self.restore_to_gpu()
            return None
        if action == "colocate_release":
            self.offload_to_cpu()
            return None
        raise ValueError(f"unknown critic command {action!r}")

    # ── public API (rank 0) ─────────────────────────────────────────────
    def score_and_train(self, samples: list[Any], *, publish: bool = True) -> Any:
        """Score one window, publish the result, then update the value net.

        What is published depends on :attr:`publish_mode`: ``"values"`` returns
        the per-token value function per row (the trainer turns it into GAE
        advantages), ``"advantage"`` returns one reduced scalar per sequence.

        ``publish=False`` during the critic's cold start: the window still
        trains the value net, but nothing is returned, so the actor sees
        nothing (recipe §2).
        """
        # Read by rank 0 only, immediately before the broadcast inside
        # ``_score_and_train_impl`` puts it on the wire for every rank.
        self._publish = bool(publish)
        return self.submit_command("score_and_train", samples=samples)

    # ── implementation (all ranks) ──────────────────────────────────────
    def _score_and_train_impl(self, samples: list[Any] | None) -> dict[str, Any] | None:
        from meshy.backend.titan.critic.data import samples_from_rows
        from meshy.backend.titan.critic.gae import compute_returns
        from meshy.backend.titan.critic.metrics import critic_diagnostics

        if self.critic is None:
            raise RuntimeError("CriticSpmdEngine.score_and_train() called before init()")

        broadcast = self._broadcast(
            {"samples": samples, "publish": self._publish} if self.is_master else None
        )
        rows = broadcast["samples"]
        publish = bool(broadcast["publish"])
        batch_samples = samples_from_rows(rows)

        # ---- 1. score with the CURRENT value function -------------------
        # Before any update, so the values the actor receives come from the
        # value function that did not see this window.
        per_seq: list[float] = []
        per_row_values: list[torch.Tensor] = []
        values, returns, masks, row_rewards, group_ids = [], [], [], [], []
        micro_batches: list[tuple[Any, ...]] = []
        for lo in range(0, len(batch_samples), self.micro_rows):
            chunk = batch_samples[lo : lo + self.micro_rows]
            n_docs = len(chunk)
            # One build per chunk, on CPU: the scoring forward takes a device
            # copy and the value step below consumes the CPU tensors directly.
            # Building it twice (once per phase) doubled the padding and the
            # host-side tensor construction for every row of the window.
            b = self.critic.make_batch(chunk, device="cpu")
            on_device = {
                key: tensor.to(self.critic.device, non_blocking=True)
                for key, tensor in b.items()
            }
            if self.publish_mode == "advantage":
                _, adv_seq, v = self.critic.predict_vapo_gae(
                    on_device["input_ids"], on_device["positions"],
                    on_device["rewards"], on_device["mask"],
                    on_device["doc_ids"], n_docs, gamma=self.gamma, alpha=self.alpha,
                )
                per_seq.extend(float(x) for x in adv_seq.detach().float().cpu().tolist())
            else:
                # The trainer runs GAE itself, so the critic owes it only the
                # value function. Running the recursion here too would be a
                # second full-length scan per row for a number nobody reads.
                v = self.critic.predict_values(
                    on_device["input_ids"], on_device["positions"]
                )
            # ``v`` is the pre-update value function -- the same forward the
            # gauges describe and, in ``values`` mode, exactly what is
            # published, so the actor's advantage cannot be contaminated by
            # this window's own value step.
            v_host = v.detach().float().cpu()
            # Published per row and trimmed to the row's own length: the padded
            # tail is not part of the sample and the trainer slices by ``L``.
            per_row_values.extend(
                v_host[j, : len(chunk[j].tokens)].clone() for j in range(n_docs)
            )
            targets = compute_returns(
                b["rewards"], b["mask"], b["doc_ids"], gamma=self.gamma
            )
            values.append(v_host)
            returns.append(targets)
            masks.append(b["mask"])
            row_rewards.append(b["row_rewards"])
            group_ids.append(b["group_ids"])
            micro_batches.append(
                (b["input_ids"], b["positions"], targets, b["mask"], None)
            )

        # ---- 2. gauges, from those same pre-update values ---------------
        diagnostics = critic_diagnostics(
            _cat_padded(values), _cat_padded(returns), _cat_padded(masks),
            torch.cat(row_rewards), torch.cat(group_ids),
        )

        # ---- 3. now update the value net --------------------------------
        # One optimiser step per window, not per row. ``micro_rows`` bounds the
        # rows resident on the GPU for a single forward/backward; it must not
        # also set the update granularity. Stepping per chunk would make a
        # 480-row window 480 updates, put the whole ``warmup_steps`` warmup
        # inside the first window's first few rows, and give every update the
        # gradient of a single sequence -- none of which is recipe §2, which
        # counts critic iterations in rollout steps ("~25 steps before
        # value_loss enters its normal range", "10 iter lr warmup").
        #
        # ``train_value_accumulated`` takes the whole window's micro-batches,
        # normalises every one of them by the window-global assistant-token
        # count across DP and CP, and applies a single warmup + clip + step
        # after they have all contributed gradients -- mathematically one fit
        # on the concatenated window at micro-batch peak memory. Queued
        # micro-batches stay on CPU; the engine moves one at a time to the
        # device immediately before its forward (same pattern as
        # ``scripts/critic_train_trajectory.py``). The micro-batches were built
        # by the scoring loop above, which is also what guarantees the value
        # step regresses onto exactly the targets the gauges reported.
        losses: list[float] = [
            self.critic.train_value_accumulated(micro_batches)
            for _ in range(max(1, self.value_epochs))
        ]

        if not self.is_master:
            return None
        result: dict[str, Any] = {
            "value_loss": sum(losses) / max(1, len(losses)),
            "rows": len(batch_samples),
            "diagnostics": diagnostics,
        }
        if publish:
            from tensordict import TensorDict

            if self.publish_mode == "values":
                result["values"] = [
                    TensorDict({"values": v}, batch_size=[]) for v in per_row_values
                ]
            else:
                result["advantage"] = [
                    TensorDict(
                        {"advantage": torch.tensor(float(a), dtype=torch.float32)},
                        batch_size=[],
                    )
                    for a in per_seq
                ]
        return result

    # ── colocation ──────────────────────────────────────────────────────
    def offload_to_cpu(self) -> None:
        if self.critic is None:
            return
        self.critic.model.to("cpu")
        # ``model.to`` does not reach the optimizer: AdamW keeps two fp32
        # tensors per trainable parameter in ``optimizer.state``, keyed by the
        # parameter object. Left behind they hold ~2x the sharded parameter
        # bytes on the card for the rest of the run -- see
        # ``TitanTrainer._move_optimizer_states``.
        self._move_optimizer_states("cpu")
        self._sync_rope_cache()
        torch.cuda.empty_cache()

    def restore_to_gpu(self) -> None:
        if self.critic is None:
            return
        self.critic.model.to(self.critic.device)
        self._move_optimizer_states(self.critic.device)
        self._sync_rope_cache()

    def _sync_rope_cache(self) -> None:
        """Re-point ``rope.cache`` at the ``freqs_cis`` buffer after a move.

        ``RoPE.cache`` is a plain attribute, not a registered buffer, so
        ``model.to(...)`` rebinds ``freqs_cis`` (the tensor the forward reads)
        and leaves ``cache`` referencing the old device's storage -- keeping a
        full RoPE cache alive on the card across every hand-off. Same fixup as
        ``TitanTrainer.restore_to_gpu``.
        """
        model = getattr(self.critic, "model", None)
        rope = getattr(model, "rope", None)
        if rope is not None and torch.is_tensor(getattr(model, "freqs_cis", None)):
            rope.cache = model.freqs_cis

    def _move_optimizer_states(self, device: Any) -> None:
        optimizer = getattr(self.critic, "optimizer", None)
        if optimizer is None:
            return
        target = torch.device(device)
        for state in optimizer.state.values():
            for key, value in state.items():
                # ``step`` is a scalar whose device belongs to the AdamW
                # variant in use (fused/capturable); moving it breaks those
                # kernels' assumptions.
                if key == "step" or not torch.is_tensor(value):
                    continue
                if value.device.type != target.type:
                    state[key] = value.to(target)

    def on_colocate_acquire(self, grant: Any) -> None:
        del grant
        # The colocation manager runs on rank 0 only, so the move has to go
        # through the command broadcast or the other ranks stay on CPU while
        # their inputs arrive on the GPU.
        if self.world_size > 1 and self.is_master:
            self.submit_command("colocate_acquire")
        else:
            self.restore_to_gpu()

    def on_colocate_release(self, target: str) -> None:
        del target
        if self.world_size > 1 and self.is_master:
            self.submit_command("colocate_release")
        else:
            self.offload_to_cpu()

    def health_info(self) -> dict[str, Any]:
        return {"train_steps": getattr(self.critic, "train_steps", 0)}


def _cat_padded(chunks: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate ``[rows, S_i]`` chunks along rows, right-padding to max S."""
    if len(chunks) == 1:
        return chunks[0]
    width = max(c.shape[1] for c in chunks)
    return torch.cat(
        [
            c
            if c.shape[1] == width
            else torch.cat([c, c.new_zeros(c.shape[0], width - c.shape[1])], dim=1)
            for c in chunks
        ],
        dim=0,
    )


__all__ = ["CriticSpmdEngine"]
