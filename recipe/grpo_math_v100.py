"""GRPO vs DAPO on Hendrycks MATH for Qwen3-0.6B on one 32 GB V100 (sm70).

Derived from :mod:`recipe.grpo_gsm8k_v100` (same colocate/memory/sm70 setup
and the same verified window geometry: 40 windows x 512, mini64 = 8 updates,
AC full, mem_fraction 0.60, decode graph bs64, DCP 10/2). Only the dataset /
reward / holdout differ:

* dataset: HendrycksMATH (EleutherAI mirror, seven subjects concatenated,
  7500 train rows); answers are the final post-think ``\\boxed{}`` expression,
  scored by math_verify equivalence (normalised + SymPy fallback).
* reward: 0/1 :func:`HendrycksMATH.reward`; DAPO's overlong shaping is added
  only when ``XRL_OVERLONG_SHAPING=1``.
* in-loop holdout: MATH-500 (``XRL_EVAL_DATASET=math500``), 200x1 every 6
  windows, 4096 max new tokens.

The two comparison runs differ only by the DAPO switches, sharing the same
seed and prompt order (BoundedHendrycksMATH seeks with XRL_START_WINDOW / the
DCP-resolved window exactly like GSM8K):

* math_grpo40: both switches off.
* math_dapo40: XRL_DYNAMIC_SAMPLING=1 XRL_OVERLONG_SHAPING=1.
"""

from __future__ import annotations

import os

from meshy.config import (
    RolloutServiceConfig,
    InferenceServiceConfig,
    TrainerConfig,
    TrainerParamsConfig,
    TrainingServiceConfig,
)
from meshy.dataset.hendrycks_math import HendrycksMATH
from meshy.service.base import ServiceGroup
from meshy.service.colocation import ColocationRing, SchedulingMode
from meshy.service.ignite import Ignitor

# In-loop eval for this recipe is MATH-500 regardless of the process default.
os.environ.setdefault("XRL_EVAL_DATASET", "math500")


class BoundedHendrycksMATH(HendrycksMATH):
    """HendrycksMATH restricted to this run's slice of the prompt curriculum.

    Mirrors BoundedGSM8K: stop after ``batches_this_run`` prompt windows and
    seek to ``start_window * batch_size`` on a warm/DCP start so prompts never
    replay. Counts *windows* (not draws) on the first draw so the DAPO
    dynamic-sampling refill loop does not consume run length.
    """

    def __init__(
        self,
        batch_size: int,
        seed: int | None = None,
        start_window: int | None = None,
        batches_this_run: int | None = None,
        **kwargs,
    ):
        if start_window is None or batches_this_run is None:
            plan = WINDOW_PLAN
            start_window = plan.start_window if start_window is None else start_window
            if batches_this_run is None:
                batches_this_run = plan.batches_this_run
        super().__init__(
            batch_size=batch_size,
            seed=seed,
            start_index=int(start_window) * batch_size,
            wrap_epochs=True,
            **kwargs,
        )
        self._batches_left = int(batches_this_run)

    def next_batch(self, builder):
        if self._batches_left <= 0:
            return []
        self._batches_left -= 1
        return super().next_batch(builder)

    def begin_window(self) -> bool:
        if self._batches_left <= 0:
            return False
        self._batches_left -= 1
        return True


MODEL_PATH = os.environ.get("XRL_MODEL", "/data00/meshy/models/Qwen3-0.6B")
from recipe.v100_windows import resume_inference_model_path

INFERENCE_MODEL_PATH = resume_inference_model_path(MODEL_PATH)
MODEL_NAME = os.environ.get("XRL_MODEL_NAME", "qwen3")
MODEL_FLAVOR = os.environ.get("XRL_MODEL_FLAVOR", "0.6B")
SEQ_LEN = int(os.environ.get("XRL_SEQ_LEN", "5120"))
MAX_NEW_TOKENS = int(os.environ.get("XRL_MAX_NEW_TOKENS", "4096"))

STORAGE_ROOT = os.environ.get("XRL_STORAGE_ROOT", "/data00/meshy/store")
CKPT_DIR = os.environ.get("XRL_CKPT_DIR", os.path.join(STORAGE_ROOT, "ckpt"))
ROLLOUT_DIR = os.environ.get("XRL_ROLLOUT_DIR", os.path.join(STORAGE_ROOT, "rollout"))

ROLLOUT_BATCH = int(os.environ.get("XRL_ROLLOUT_BATCH", "8"))
GROUP_SIZE = int(os.environ.get("XRL_GROUP_SIZE", "8"))
BATCH_SIZE = ROLLOUT_BATCH * GROUP_SIZE
RL_STEPS = int(os.environ.get("XRL_STEPS", "40"))

from recipe.v100_windows import resolve_start_window

WINDOW_PLAN = resolve_start_window(RL_STEPS)

TRAIN_DTYPE = os.environ.get("XRL_TRAIN_DTYPE", "float32")
TRAIN_MIXED_PARAM = os.environ.get("XRL_TRAIN_MIXED_PARAM", "float16")


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        model_name=MODEL_NAME,
        model_flavor=MODEL_FLAVOR,
        seq_len=SEQ_LEN,
        steps=RL_STEPS,
        dtype=TRAIN_DTYPE,
        mixed_precision_param=TRAIN_MIXED_PARAM,
        lr=float(os.environ.get("XRL_LR", "1e-6")),
        weight_decay=0.1,
        max_norm=1.0,
        warmup_steps=0,
        dp_shard_degree=-1,
        dp_replicate_degree=1,
        tp_degree=1,
        cp_degree=1,
        enable_checkpoint=os.environ.get("XRL_ENABLE_DCP_CKPT", "1") == "1",
        checkpoint_interval=int(os.environ.get("XRL_DCP_CKPT_INTERVAL", "10")),
        checkpoint_keep=int(os.environ.get("XRL_DCP_CKPT_KEEP", "2")),
        # The final-window DCP must resume a crashed arm, so keep optimizer/LR/
        # train state (torchtitan default True writes weights-only). Both GRPO
        # and DAPO arms use this one recipe, so this covers both; the full final
        # DCP costs ~8.8 GB extra across the two arms.
        last_save_model_only=os.environ.get("XRL_LAST_SAVE_MODEL_ONLY", "0") == "1",
        dump_folder=os.path.join(
            CKPT_DIR,
            "grpo_math_v100",
            os.environ.get(
                "XRL_RUN_TAG",
                os.path.basename(os.environ.get("XRL_RUNTIME_DIR", "").rstrip("/"))
                or "default",
            ),
        ),
        compile_model=False,
        activation_checkpoint_mode=os.environ.get(
            "XRL_ACTIVATION_CHECKPOINT", "full"
        ),
    )


def _trainer_params() -> TrainerParamsConfig:
    # Pinned to the values clean40b actually trained with (its 2026-09-29
    # 17:02 TitanTrainer init): max_tokens_per_micro=4096 (per-token micro
    # packing), mini_batch_size=64 (512/64 = 8 optimizer updates/window),
    # seq_bucket=64 -> seq_align=64. Both GRPO and DAPO arms inherit identical
    # values from this one recipe; the runbook launch exports them too so the
    # shared v100_run_rl.sh defaults (gsm8k mini=8/seq=1024) cannot leak in.
    mtpm = os.environ.get("XRL_MAX_TOKENS_PER_MICRO", "4096")
    params = dict(
        mini_batch_size=int(os.environ.get("XRL_MINI_BATCH", "64")),
        micro_batch_size=1,
        ppo_clip_eps_low=0.2,
        ppo_clip_eps_high=0.28,
        old_logprobs_source="train",
    )
    if mtpm:
        params["batch_layout"] = "padded"
        params["max_tokens_per_micro"] = int(mtpm)
    params["seq_bucket"] = int(os.environ.get("XRL_SEQ_BUCKET", "64"))
    return TrainerParamsConfig(**params)


def _inference_config() -> InferenceServiceConfig:
    return InferenceServiceConfig(
        model_path=INFERENCE_MODEL_PATH,
        server_args={
            "model_path": INFERENCE_MODEL_PATH,
            "tp_size": 1,
            "enable_memory_saver": True,
            "mem_fraction_static": float(os.environ.get("XRL_MEM_FRACTION", "0.60")),
            "dtype": "half",
            "attention_backend": "triton",
            "sampling_backend": "pytorch",
        },
    )


def _training_config() -> TrainingServiceConfig:
    return TrainingServiceConfig(
        model_path=MODEL_PATH,
        trainer_config=_trainer_config(),
        trainer_params=_trainer_params(),
        batch_size=BATCH_SIZE,
        timer_enabled=True,
    )


def _rollout_group() -> ServiceGroup:
    overlong_shaping = os.environ.get("XRL_OVERLONG_SHAPING", "0") == "1"
    rollout_kwargs = dict(
        model_path=MODEL_PATH,
        dataset="recipe.grpo_math_v100:BoundedHendrycksMATH",
        dataset_kwargs={"batch_size": ROLLOUT_BATCH, "split": "train", "seed": 42},
        reward="meshy.dataset.hendrycks_math:HendrycksMATH.reward",
        sampling_params={
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        group_size=GROUP_SIZE,
        poll_interval=2.0,
        pacing_window=1,
        num_epochs=int(os.environ.get("XRL_EPOCHS", "1")),
        version_hook="recipe.v100_inloop:version_hook",
    )
    if os.environ.get("XRL_DYNAMIC_SAMPLING", "0") == "1":
        rollout_kwargs["dynamic_sampling"] = True
        # This hard cap (kept+dropped prompts in one window), not
        # oversample_factor, is the real refill ceiling in the dynamic path
        # (rollout.py caps the replacement budget at max-target). 192 = 3x the
        # 64-prompt target; the §3.2 decision to "cut oversample to 2" lowers
        # this to 128. Overshoot keeps zero-variance groups so the window fills.
        rollout_kwargs["dynamic_max_prompts"] = int(
            os.environ.get("XRL_DYNAMIC_MAX_PROMPTS", "192")
        )
    if overlong_shaping:
        rollout_kwargs["reward_shaping"] = "meshy.reward:dapo_overlong_penalty"
        rollout_kwargs["reward_shaping_kwargs"] = {
            "max_response_len": int(os.environ.get("XRL_OVERLONG_L_MAX", str(MAX_NEW_TOKENS))),
            "cache_len": int(os.environ.get("XRL_OVERLONG_L_CACHE", "1024")),
        }
    return ServiceGroup(
        id="rollout",
        n_replicas=1,
        n_gpus_per_replica=0,
        wait_until=["actor_train", "actor_infer"],
        config=RolloutServiceConfig(**rollout_kwargs),
    )


def build_service_groups() -> list[ServiceGroup]:
    return [
        ServiceGroup(id="actor_infer", config=_inference_config(),
                    n_replicas=1, n_gpus_per_replica=1),
        ServiceGroup(id="actor_train", config=_training_config(),
                    n_replicas=1, n_gpus_per_replica=1,
                    colocate_with="actor_infer", wait_until=["actor_infer"]),
        _rollout_group(),
    ]


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
    os.environ.setdefault("XRL_RUNTIME_DIR", ROLLOUT_DIR)
    Ignitor(SERVICE_GROUPS, COLOCATIONS).run()


if __name__ == "__main__":
    main()
