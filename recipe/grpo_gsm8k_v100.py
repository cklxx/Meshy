"""GRPO-on-GSM8K for Qwen3-0.6B on a single 32 GB V100 (sm70, colocate).

Single-card asymmetric colocate: one TP=1 SGLang engine and one FSDP=1
TitanTrainer share the card; the trainer is offloaded to host RAM during
rollout and the engine's weights/KV pool are released during training
(``enable_memory_saver``).

sm70 constraints baked in:
* inference dtype is fp16 (``--dtype half``): V100 has no bf16 tensor cores;
* training keeps fp32 master weights/grads/Adam moments with fp16 FSDP
  all-gather compute and a dynamic GradScaler (``dtype="float32"`` +
  ``mixed_precision_param="float16"``). Uniform fp16 storage
  (``XRL_TRAIN_DTYPE=float16``) is available but unsound at lr=1e-6: with
  |w|~0.02 an AdamW step (~1e-6) is below half of the fp16 ULP (~1.5e-5) and
  rounds away, so the policy does not move. See "Defeating the
  Training-Inference Mismatch via FP16" (arXiv:2510.26788);
* attention/sampling backends are triton/pytorch: flashinfer prebuilt wheels
  ship sm75+ cubins only and sgl-kernel ships sm80+ only, so the default
  flashinfer attention/sampling paths cannot launch on V100;
* decode CUDA graph is on by sm70 default (``sm70_server_defaults``): the
  triton backend supports it on sm70 and measured decode goes 25 -> 208
  tok/s single-stream, 1436 -> 3310 at batch 64; prefill graph stays off
  (variable-shape capture is the path that poisoned the allocator). Only
  the decode graph adds ~3 GiB of capture pool; release/resume of the
  memory saver is verified with capture present (sm70 T1f check).
  ``MESHY_SM70_CUDA_GRAPH=0`` forces eager;
* ``compile_model=False``: inductor on sm70 buys little and risks sm-specific
  codegen.

Memory accounting (Qwen3-0.6B: 596 M unique params, tied embed; the
checkpoint ships an untied lm_head copy so model.safetensors is 0.75 B /
1.40 GiB on disk, which shrinks to 1.11 GiB once weights are tied;
hidden 1024, 28 layers, 16 q heads / 8 KV heads x head_dim 128,
intermediate 3072, vocab 151936)
---------------------------------------------------------------------------
GPU, rollout phase (trainer offloaded):
  SGLang static pool  mem_fraction 0.60   19.2  GiB  (weights sit inside it)
    ├─ engine weights  fp16               1.11 GiB
    ├─ runtime/workspaces + decode graph capture (~3 GiB)  ~5   GiB
    └─ KV pool                            ~13   GiB
  dynamic (torch allocator outside pool)  ~2    GiB
  ----------------------------------------------
  used                                 ~21 GiB of 32, headroom ~11
  KV per token = 2*28*8*128*2 = 114,688 B; 16 GiB => ~140 k KV tokens
  (~32 full 4.3k-token sequences worst case; median completion is ~485
  tokens, so the 64-seq rollout window is KV-bound only on the long tail).

GPU, training phase (KV pool released via memory saver; engine weights may
stay resident, ~1.1 GiB):
  params  fp32 master                 2.22 GiB
  grads   fp32                       2.22 GiB
  AdamW exp_avg/exp_avg_sq fp32      4.44 GiB
  (FSDP all-gather casts the working copy to fp16 for each forward/backward)
  activations (micro_batch=1, seq up to 5120,
    selective AC) + fp32 logits chunk
    (1024 x 151936 x 4) + workspaces  ~3    GiB
  ----------------------------------------------
  peak                                ~12 GiB of 32
  XRL_TRAIN_DTYPE=float16 cuts master+grad+state residency to 4.44 GiB but
  stalls lr=1e-6 updates; use only with a much larger lr.

Host RAM, 31 GiB total:
  rollout phase: offloaded trainer residency (fp32 params+Adam states;
  grads live only inside a training step) 6.66 GiB
                   + SGLang/framework RSS ~4 GiB                     ~11 GiB
  training phase: engine residual RSS ~1.5 + trainer/framework RSS ~3  ~5 GiB
(1) The fp32 master copy is what makes lr=1e-6 AdamW steps representable;
the fp16 all-gather only affects matmul inputs, matching the inference
rollout precision. fp16 RL paper (arXiv:2510.26788): fp16 rollout + fp16
compute plus loss scaling, not fp16 weight storage.

Run::

    PATH=/usr/local/cuda-12.4/bin:$PATH HF_ENDPOINT=https://hf-mirror.com \\
    XRL_MODEL=/data00/meshy/models/Qwen3-0.6B \\
        python scripts/launch.py --recipe recipe.grpo_gsm8k_v100
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
from meshy.dataset.gsm8k import GSM8K
from meshy.service.base import ServiceGroup
from meshy.service.colocation import ColocationRing, SchedulingMode
from meshy.service.ignite import Ignitor


class BoundedGSM8K(GSM8K):
    """:class:`GSM8K` that serves exactly ``RL_STEPS`` prompt batches.

    The stock dataset only signals end when its rows run out; 7473 train rows
    give ~934 batches per epoch, so a short run would overshoot the LR
    scheduler horizon. Mirroring ``recipe.justrl_smoke.SmokeMATH``, this stops
    the rollout (and hence the run) at the configured step count. The bound is
    per dataset *instance*, and the worker rebuilds the dataset every epoch,
    so the recipe pins ``num_epochs=1``: the bound is the total run length.
    """

    def __init__(self, batch_size: int, seed: int | None = None, **kwargs):
        super().__init__(batch_size=batch_size, seed=seed, **kwargs)
        self._batches_left = RL_STEPS

    def next_batch(self, builder):
        if self._batches_left <= 0:
            return []
        self._batches_left -= 1
        return super().next_batch(builder)

MODEL_PATH = os.environ.get("XRL_MODEL", "/data00/meshy/models/Qwen3-0.6B")
MODEL_NAME = os.environ.get("XRL_MODEL_NAME", "qwen3")
MODEL_FLAVOR = os.environ.get("XRL_MODEL_FLAVOR", "0.6B")
SEQ_LEN = int(os.environ.get("XRL_SEQ_LEN", "5120"))
# 4096 covers the p95 sampled completion (3992 tokens; 200-question x4 holdout
# at temp 0.6/top_p 0.95/top_k 20). GSM8K prompts max out at 207 tokens, so
# p95 prompt+completion fits in ~4200; 5120 is the smallest seq_bucket=1024
# multiple above it (micro-batches still pad to the batch's own longest
# bucket, not to seq_len). 1024 replaces the default 2048 bucket so the
# ceiling is not forced onto a 2048 grid.
MAX_NEW_TOKENS = int(os.environ.get("XRL_MAX_NEW_TOKENS", "4096"))

# Shared 3FS root (FUSE-mounted on the V100). Checkpoints/dumps go under
# ckpt/, the runtime root (weights + rollout trajectories) under rollout/.
# Set XRL_STORAGE_ROOT to a local path to run without 3FS; per-dir overrides
# via XRL_CKPT_DIR / XRL_ROLLOUT_DIR take precedence.
STORAGE_ROOT = os.environ.get("XRL_STORAGE_ROOT", "/3fs/stage/meshy")
CKPT_DIR = os.environ.get("XRL_CKPT_DIR", os.path.join(STORAGE_ROOT, "ckpt"))
ROLLOUT_DIR = os.environ.get("XRL_ROLLOUT_DIR", os.path.join(STORAGE_ROOT, "rollout"))

ROLLOUT_BATCH = int(os.environ.get("XRL_ROLLOUT_BATCH", "8"))  # prompts per step
GROUP_SIZE = int(os.environ.get("XRL_GROUP_SIZE", "8"))  # completions per prompt
BATCH_SIZE = ROLLOUT_BATCH * GROUP_SIZE  # trainer trigger threshold: 64
# Outer RL updates. Two things are keyed to this number:
#  1. the LR scheduler horizon (``TrainerConfig.steps``) — one
#     ``lr_scheduler.step()`` per *outer* train_step (the mini-batches inside
#     it only call ``optimizer.step``; the scheduler call is gated by
#     ``step_schedule=True``, which fires once per 64-sample TQ batch);
#  2. the rollout length — :class:`BoundedGSM8K` returns exactly this many
#     prompt batches, so the run ends at the scheduler horizon instead of
#     overrunning it (the framework otherwise stops only on dataset
#     exhaustion; with lr_decay_ratio=0 the scheduler asserts on the first
#     step past ``steps`` — stable_steps == steps+1).
RL_STEPS = int(os.environ.get("XRL_STEPS", "300"))
# GradScaler/LR proof: 8 epochs x 7473 train rows / 8 prompts = ~7473
# possible batches; the bound below is what actually stops the run.

# fp32 master weights + fp16 FSDP compute + dynamic loss scaling (the sm70
# default). XRL_TRAIN_DTYPE=float16 is uniform-fp16 storage: ~2 GiB cheaper
# per arm but lr=1e-6 AdamW steps underflow; see module docstring.
TRAIN_DTYPE = os.environ.get("XRL_TRAIN_DTYPE", "float32")
# Compute dtype for the FSDP working copy. None follows TRAIN_DTYPE
# (float16 -> fp16 compute, else bf16 default); sm70 wants fp16.
TRAIN_MIXED_PARAM = os.environ.get("XRL_TRAIN_MIXED_PARAM", "float16")


def _trainer_config() -> TrainerConfig:
    # One GPU: dp_shard_degree=-1 with nproc=1 still applies FSDP
    # MixedPrecisionPolicy even at degree 1 (torchtitan keeps the fsdp mesh
    # for exactly this), so the fp32 storage / fp16 compute split is real.
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
        enable_checkpoint=False,
        dump_folder=os.path.join(CKPT_DIR, "grpo_gsm8k_v100"),
        compile_model=False,
        # At ~4-5k-token sequences "selective" (per-op) AC still retains every
        # layer's QKV/attention-score activations: the fp32 attention scores
        # alone (16 heads x S^2 x 4 B) are ~1.6 GiB at S=4937 per layer, and
        # 28 resident layers OOM the V100. "full" wraps each transformer
        # block; backward recomputes one block at a time (measured decision
        # for long-context recipes, see TrainerConfig docstring).
        activation_checkpoint_mode="full",
    )


def _trainer_params() -> TrainerParamsConfig:
    # Mirrors recipe/justrl: 8-sample mini-batch, micro 1, asymmetric clip,
    # old logprobs recomputed inside the train pass. XRL_MINI_BATCH scales with
    # the rollout window so optimizer updates/window stay 8 (64/8 or 512/64).
    # XRL_MAX_TOKENS_PER_MICRO switches from per-row to per-token micro
    # sizing (padded/sdpa; packed/varlen needs flash_attn sm75+ + bf16, which
    # V100 lacks). Token budget takes precedence over micro_batch_size and cuts
    # the number of micro-forward passes in a 512 window ~5x (512 -> ~104).
    mtpm = os.environ.get("XRL_MAX_TOKENS_PER_MICRO")
    params = dict(
        mini_batch_size=int(os.environ.get("XRL_MINI_BATCH", "8")),
        micro_batch_size=1,
        seq_bucket=1024,
        ppo_clip_eps_low=0.2,
        ppo_clip_eps_high=0.28,
        old_logprobs_source="train",
    )
    if mtpm:
        params["batch_layout"] = "padded"
        params["max_tokens_per_micro"] = int(mtpm)
    return TrainerParamsConfig(**params)


def _inference_config() -> InferenceServiceConfig:
    return InferenceServiceConfig(
        model_path=MODEL_PATH,
        server_args={
            "model_path": MODEL_PATH,
            "tp_size": 1,
            # Required for the colocate GPU hand-off.
            "enable_memory_saver": True,
            # See module docstring: 19.2 GiB static pool on the rollout card.
            "mem_fraction_static": float(os.environ.get("XRL_MEM_FRACTION", "0.60")),
            # sm70: fp16 weights/KV, triton attention (flashinfer cubins are
            # sm75+), pytorch sampling (sgl-kernel ops are sm80+). Decode
            # CUDA graph comes from sm70_server_defaults (on; MESHY_SM70_CUDA_GRAPH=0
            # forces eager); do not pin disable_cuda_graph here.
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
    return ServiceGroup(
        id="rollout",
        n_replicas=1,
        n_gpus_per_replica=0,
        wait_until=["actor_train", "actor_infer"],
        config=RolloutServiceConfig(
            model_path=MODEL_PATH,
            dataset="recipe.grpo_gsm8k_v100:BoundedGSM8K",
            dataset_kwargs={"batch_size": ROLLOUT_BATCH, "split": "train", "seed": 42},
            reward="meshy.dataset.gsm8k:GSM8K.reward",
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
            # In-loop holdout (200x1 at v0/50/.../250), milestone copy and
            # old-version pruning. Fires on the first group after each new
            # weight grant, so eval runs under the exact version with the
            # trainer blocked. v300 has no rollout window (the bounded dataset
            # serves exactly 300 batches), so its 200x4 eval is standalone.
            version_hook="recipe.v100_inloop:version_hook",
        ),
    )


def build_service_groups() -> list[ServiceGroup]:
    return [
        ServiceGroup(
            id="actor_infer",
            config=_inference_config(),
            n_replicas=1,
            n_gpus_per_replica=1,
        ),
        ServiceGroup(
            id="actor_train",
            config=_training_config(),
            n_replicas=1,
            n_gpus_per_replica=1,
            colocate_with="actor_infer",
            wait_until=["actor_infer"],
        ),
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
    # Point the runtime root (weights + rollout trajectories) at 3FS unless
    # the caller already pinned XRL_RUNTIME_DIR.
    os.environ.setdefault("XRL_RUNTIME_DIR", ROLLOUT_DIR)
    Ignitor(SERVICE_GROUPS, COLOCATIONS).run()


if __name__ == "__main__":
    main()
