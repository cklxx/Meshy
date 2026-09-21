"""32K MiniCPM5-2B-SFT critic cold start: 6 inference GPUs + CP2 critic.

Set MINICPM5_LOCAL_PATH to a local OpenBMB/MiniCPM5-2B-SFT snapshot,
S9_DATASET_PATH to s9_no_advanced_math.jsonl, and HF_HOME to the HF cache.
Launch with ``python scripts/launch.py --recipe recipe.critic_coldstart_32k``.
See docs/critic_coldstart_plan.md for the launch command and metric semantics.
"""

from __future__ import annotations

import os
from pathlib import Path

from meshy.config import (
    CriticServiceConfig,
    InferenceServiceConfig,
    RolloutServiceConfig,
    SamplingParams,
    TrainerConfig,
)
from meshy.service.base import ServiceGroup
from meshy.service.ignite import Ignitor

try:
    MODEL_PATH = str(Path(os.environ["MINICPM5_LOCAL_PATH"]).expanduser().resolve())
except KeyError as exc:
    raise RuntimeError("MINICPM5_LOCAL_PATH must be set") from exc

try:
    DATASET_PATH = str(Path(os.environ["S9_DATASET_PATH"]).expanduser().resolve())
except KeyError as exc:
    raise RuntimeError("S9_DATASET_PATH must point to the S9 JSONL file") from exc

SEQ_LEN = 32768
MAX_NEW_TOKENS = 28672
NUM_INFERENCE_CARDS = 6
NUM_CRITIC_CARDS = 2
ROLLOUT_BATCH = 16
GROUP_SIZE = 8
NUM_EPOCHS = int(os.environ.get("XRL_NUM_EPOCHS", "100"))
CRITIC_SCORE_BATCH = int(os.environ.get("XRL_CRITIC_SCORE_BATCH", "64"))

if CRITIC_SCORE_BATCH <= 0 or CRITIC_SCORE_BATCH % GROUP_SIZE:
    raise ValueError("XRL_CRITIC_SCORE_BATCH must be a positive multiple of GROUP_SIZE (8)")
if NUM_EPOCHS <= 0:
    raise ValueError("XRL_NUM_EPOCHS must be positive")

# Let the launcher resolve endpoints after establishing the runtime directory.
os.environ.setdefault("XRL_TQ_PRE_ALLOC", str(4 * CRITIC_SCORE_BATCH))
if int(os.environ["XRL_TQ_PRE_ALLOC"]) < CRITIC_SCORE_BATCH:
    raise ValueError("XRL_TQ_PRE_ALLOC must hold at least one critic window")


def _validate_local_model() -> None:
    model_dir = Path(MODEL_PATH)
    required = ("config.json", "tokenizer.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if not model_dir.is_dir() or missing or not any(model_dir.glob("*.safetensors")):
        detail = f"; missing files: {missing}" if missing else ""
        raise FileNotFoundError(
            f"MiniCPM5 must be loaded from a complete local model directory: "
            f"{model_dir}{detail}"
        )
    if not Path(DATASET_PATH).is_file():
        raise FileNotFoundError(f"S9 dataset does not exist: {DATASET_PATH}")


def _sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1, max_new_tokens=MAX_NEW_TOKENS
    )


def _inference_group() -> ServiceGroup:
    return ServiceGroup(
        id="actor_infer",
        n_replicas=NUM_INFERENCE_CARDS,
        n_gpus_per_replica=1,
        config=InferenceServiceConfig(
            model_path=MODEL_PATH,
            server_args={
                "model_path": MODEL_PATH,
                "tp_size": 1,
                "context_length": SEQ_LEN,
                "attention_backend": "fa3",
                "mem_fraction_static": 0.90,
                "max_running_requests": 64,
                "max_total_tokens": 1440000,
                "schedule_conservativeness": 1.2,
            },
        ),
    )


def _critic_trainer_config() -> TrainerConfig:
    return TrainerConfig(
        model_name="minicpm5",
        model_flavor="2B",
        seq_len=SEQ_LEN,
        max_norm=1.0,
        steps=10**6,
        # fp32 master weights. ``dtype`` sets the *storage* dtype of the
        # parameters, gradients and Adam moments; the matmuls stay bf16 via
        # FSDP's ``mixed_precision_param`` (default bfloat16, see
        # ``critic/parallel.py``). Under bf16 storage an AdamW step of ~lr=5e-6
        # is ~0.08 ULP at |w|≈0.02 and rounds away entirely, so the value head
        # never moves.
        dtype="float32",
        compile_model=False,
        dp_shard_degree=1,
        dp_replicate_degree=1,
        tp_degree=1,
        cp_degree=2,
        activation_checkpoint_mode="full",
        enable_checkpoint=False,
        dump_folder="./outputs/critic_coldstart_32k/critic",
    )


def _critic_group() -> ServiceGroup:
    return ServiceGroup(
        id="critic",
        n_replicas=1,
        n_gpus_per_replica=NUM_CRITIC_CARDS,
        wait_until=["actor_infer"],
        config=CriticServiceConfig(
            model_path=MODEL_PATH,
            trainer_config=_critic_trainer_config(),
            score_batch_size=CRITIC_SCORE_BATCH,
            train_backbone=True,
            lr=5e-6,
            warmup_steps=10,
            cold_start_windows=-1,
            publish_gate_zero=True,
            value_epochs=1,
            gamma=1.0,
            alpha=1.5,
            max_norm=1.0,
        ),
    )


def _rollout_group() -> ServiceGroup:
    return ServiceGroup(
        id="rollout",
        n_replicas=1,
        n_gpus_per_replica=0,
        wait_until=["actor_infer", "critic"],
        config=RolloutServiceConfig(
            model_path=MODEL_PATH,
            dataset="meshy.dataset.s9_math:S9Math",
            dataset_kwargs={"path": DATASET_PATH, "batch_size": ROLLOUT_BATCH, "seed": 42},
            reward="meshy.dataset.s9_math:S9Math.reward",
            sampling_params=_sampling_params().as_dict(),
            group_size=GROUP_SIZE,
            num_epochs=NUM_EPOCHS,
            poll_interval=2.0,
            pacing_window=None,
            # TQ bounds stored rows; this also bounds generation and pending puts.
            async_max_running_request=NUM_INFERENCE_CARDS * 64,
            verbose_trajectory_log=True,
            external_advantage=True,
            filter_zero_std_groups=False,
        ),
    )


def build_service_groups() -> list[ServiceGroup]:
    # Declaration order assigns inference GPUs 0-5 and critic GPUs 6-7.
    return [_inference_group(), _critic_group(), _rollout_group()]


SERVICE_GROUPS = build_service_groups()


def main() -> None:
    _validate_local_model()
    Ignitor(SERVICE_GROUPS).run()


if __name__ == "__main__":
    main()
