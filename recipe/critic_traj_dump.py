"""Bounded MiniCPM5-2B rollout that writes a **verbose** trajectory dump.

The critic's offline tooling (``scripts/critic_train_trajectory.py``,
``meshy/backend/titan/critic/data.py``) consumes a ``trajectory.jsonl`` that
carries per-token ``tokens`` / ``masks`` / ``logprobs``. ``TrajectoryLogger``
only writes those in verbose mode, and none of the shipped recipes turn it on,
so there is no such dump to test against. This recipe produces one.

It is ``recipe.justrl_minicpm5_2b_4gpu`` shrunk to a fixture: 4 colocated
cards, two 16-prompt batches at group size 4 (128 samples), 512 new tokens,
and ``verbose_trajectory_log=True``. It exists to generate critic input, not
to train anything — the two GRPO steps it runs are incidental.

    MINICPM5_LOCAL_PATH=/path/to/minicpm5-2b \\
        CUDA_VISIBLE_DEVICES=0,1,2,3 \\
        python scripts/launch.py --recipe recipe.critic_traj_dump

The dump lands at ``<runtime-dir>/trajectories.jsonl``.
"""

from __future__ import annotations

import os
from pathlib import Path

from meshy.config import (
    InferenceServiceConfig,
    RolloutServiceConfig,
    SamplingParams,
    TrainerConfig,
    TrainerParamsConfig,
    TrainingServiceConfig,
)
from meshy.dataset.math import MATH
from meshy.service.base import ServiceGroup
from meshy.service.colocation import ColocationRing, SchedulingMode
from meshy.service.ignite import Ignitor

try:
    MODEL_PATH = str(Path(os.environ["MINICPM5_LOCAL_PATH"]).expanduser().resolve())
except KeyError as exc:
    raise RuntimeError(
        "MINICPM5_LOCAL_PATH must be set to the MiniCPM5 checkpoint path"
    ) from exc

NUM_INFERENCE_ENGINES = 4
ROLLOUT_BATCH = 16
GROUP_SIZE = 4
BATCH_SIZE = ROLLOUT_BATCH * GROUP_SIZE  # 64
NUM_EPOCHS = 1
MAX_NEW_TOKENS = 512
SEQ_LEN = 2048

# The entire point of this recipe.
VERBOSE_TRAJECTORY_LOG = True


class BoundedMATH(MATH):
    """Expose exactly two prompt batches, so the run terminates on its own."""

    def __init__(self, batch_size: int, seed: int | None = None, **kwargs):
        super().__init__(batch_size=batch_size, seed=seed, **kwargs)
        self._batches_left = 2

    def next_batch(self, builder):
        if self._batches_left <= 0:
            return []
        self._batches_left -= 1
        return super().next_batch(builder)


def _validate_local_model() -> None:
    model_dir = Path(MODEL_PATH)
    required_files = ("config.json", "tokenizer.json", "model.safetensors.index.json")
    missing = [name for name in required_files if not (model_dir / name).is_file()]
    if not model_dir.is_dir() or missing or not any(model_dir.glob("*.safetensors")):
        detail = f"; missing files: {missing}" if missing else ""
        raise FileNotFoundError(
            f"MiniCPM5 must be loaded from a complete local model directory: "
            f"{model_dir}{detail}"
        )


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        model_name="minicpm5",
        model_flavor="2B",
        seq_len=SEQ_LEN,
        lr=1e-6,
        weight_decay=0.1,
        warmup_steps=0,
        max_norm=1.0,
        steps=2,
        dtype="bfloat16",
        compile_model=False,
        dp_shard_degree=-1,
        dp_replicate_degree=1,
        tp_degree=1,
        cp_degree=1,
        enable_checkpoint=False,
        dump_folder="./outputs/critic_traj_dump",
    )


def _trainer_params() -> TrainerParamsConfig:
    return TrainerParamsConfig(
        mini_batch_size=8,
        micro_batch_size=1,
        ppo_clip_eps_low=0.2,
        ppo_clip_eps_high=0.28,
        old_logprobs_source="train",
    )


def _inference_group() -> ServiceGroup:
    return ServiceGroup(
        id="actor_infer",
        n_replicas=NUM_INFERENCE_ENGINES,
        n_gpus_per_replica=1,
        config=InferenceServiceConfig(
            model_path=MODEL_PATH,
            server_args={
                "model_path": MODEL_PATH,
                "tp_size": 1,
                "enable_memory_saver": True,
                "mem_fraction_static": 0.6,
            },
        ),
    )


def _training_group() -> ServiceGroup:
    return ServiceGroup(
        id="actor_train",
        n_replicas=1,
        n_gpus_per_replica=NUM_INFERENCE_ENGINES,
        colocate_with="actor_infer",
        wait_until=["actor_infer"],
        config=TrainingServiceConfig(
            model_path=MODEL_PATH,
            trainer_config=_trainer_config(),
            trainer_params=_trainer_params(),
            batch_size=BATCH_SIZE,
            timer_enabled=True,
        ),
    )


def _rollout_group() -> ServiceGroup:
    return ServiceGroup(
        id="rollout",
        n_replicas=1,
        n_gpus_per_replica=0,
        wait_until=["actor_train", "actor_infer"],
        config=RolloutServiceConfig(
            model_path=MODEL_PATH,
            dataset="recipe.critic_traj_dump:BoundedMATH",
            dataset_kwargs={"batch_size": ROLLOUT_BATCH, "seed": 42},
            reward="meshy.dataset.math:MATH.reward",
            sampling_params=SamplingParams(
                temperature=1.0, top_p=1.0, top_k=-1, max_new_tokens=MAX_NEW_TOKENS
            ).as_dict(),
            group_size=GROUP_SIZE,
            poll_interval=1.0,
            pacing_window=1,
            num_epochs=NUM_EPOCHS,
            verbose_trajectory_log=VERBOSE_TRAJECTORY_LOG,
        ),
    )


def build_service_groups() -> list[ServiceGroup]:
    return [_inference_group(), _training_group(), _rollout_group()]


SERVICE_GROUPS = build_service_groups()
COLOCATIONS = [
    ColocationRing(
        group_id="actor_card",
        ring=(
            ("actor_infer", SchedulingMode.FALLBACK),
            ("actor_train", SchedulingMode.ON_DEMAND),
        ),
    )
]


def main() -> None:
    _validate_local_model()
    Ignitor(SERVICE_GROUPS, COLOCATIONS).run()


if __name__ == "__main__":
    main()
