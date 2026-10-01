#!/usr/bin/env python3
"""P1 memory probe: one fwd+bwd+optimizer step at seq_len=mtpm=7168, AC full.

Standalone (~10 min incl. model load), does NOT touch any T11 run config.
Builds the production TitanTrainer via the same forge config path as the MATH
recipe but with seq_len/max_tokens_per_micro=7168, feeds ONE synthetic
7168-token sample through one outer train step, and reports GPU/host peaks.

Run on the V100 after the GRPO arm frees the card:
  PYTHONPATH=<tree> python scripts/v100_mem_7168.py
"""
from __future__ import annotations

import os
import resource
import time

# Match the production MATH run except sequence length.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
SEQ = 7168

import torch  # noqa: E402

from meshy.config import TrainerConfig  # noqa: E402
from meshy.engine.titan import build_titan_trainer  # noqa: E402

MODEL = os.environ.get("XRL_MODEL", "/data00/meshy/models/Qwen3-0.6B")


def _host_avail_gb() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")


def _synthetic_sample(vocab: int) -> dict:
    # The trainer consumes TQ-deserialized row DICTS (not the Sample dataclass):
    # build_micro_batch indexes td["tokens"]/["mask_assistant"]/["logprobs"].
    # Deterministic pseudo-token stream; mask the whole assistant tail (worst-
    # case activation), 7168 tokens.
    ids = torch.tensor([(i * 2654435761) % vocab for i in range(SEQ)], dtype=torch.long)
    n = ids.shape[0]
    return {
        "tokens": ids,
        "mask_assistant": torch.ones(n, dtype=torch.long),
        "logprobs": torch.full((n,), -1.0, dtype=torch.float32),
        "advantage": 0.5,
        "reward": 1.0,
        "ground_truth": 0,
        "weight_version": 0,
        "truncated": False,
        "repetition": False,
        "mixed_version": False,
    }


def main() -> None:
    # torchtitan ForgeEngine reads torchrun env directly (no elastic launch in
    # this standalone probe); provide a single-rank world.
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")

    host_before = _host_avail_gb()
    torch.cuda.reset_peak_memory_stats()

    cfg = TrainerConfig(
        model_name="qwen3",
        model_flavor="0.6B",
        seq_len=SEQ,
        dtype="float32",               # fp32 master + GradScaler (forge picks fp16 compute), as MATH recipe
        dp_shard_degree=-1,
        tp_degree=1,
        cp_degree=1,
        enable_checkpoint=False,
        checkpoint_interval=10,
        checkpoint_keep=2,
        last_save_model_only=False,
        activation_checkpoint_mode="full",
    )
    # Single sample -> one micro batch of 1 row; mtpm=7168 packs the full row.
    params = {
        "mini_batch_size": 1,
        "micro_batch_size": 1,
        "ppo_clip_eps_low": 0.2,
        "ppo_clip_eps_high": 0.28,
        "old_logprobs_source": "train",
        "batch_layout": "padded",
        "max_tokens_per_micro": SEQ,
        "seq_bucket": SEQ,
    }

    t0 = time.time()
    trainer = build_titan_trainer(MODEL, cfg, params, timer_enabled=False)
    print(f"[mem7168] trainer built in {time.time()-t0:.1f}s", flush=True)

    # vocab size from the loaded model config (fallback 151936 for Qwen3-0.6B).
    vocab = getattr(getattr(trainer, "model", None), "config", None)
    vocab = getattr(vocab, "vocab_size", 151936)
    samples = [_synthetic_sample(int(vocab))]

    # Move model to GPU (the trainer normally toggles via colocation; standalone
    # probe restores it explicitly for the forward/backward).
    t0 = time.time()
    if hasattr(trainer, "restore_to_gpu"):
        trainer.restore_to_gpu()
    else:
        trainer.model.to("cuda")
    torch.cuda.synchronize()
    print(f"[mem7168] restored to GPU in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    metrics = trainer.train_step(samples)
    torch.cuda.synchronize()
    step_s = time.time() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    host_min = _host_avail_gb()
    print(f"[mem7168] one fwd+bwd+opt step in {step_s:.1f}s", flush=True)
    print(f"[mem7168] RESULT seq={SEQ} AC=full fp32master/fp16compute")
    print(f"[mem7168] cuda_max_memory_allocated_GiB={peak_alloc:.2f}")
    print(f"[mem7168] cuda_max_memory_reserved_GiB={peak_reserved:.2f}")
    print(f"[mem7168] host_avail_before_GiB={host_before:.1f} host_avail_min_GiB={host_min:.1f}")
    print(f"[mem7168] metrics_keys={len(metrics)} loss={metrics.get('loss', float('nan')):.4f}")
    print(f"[mem7168] VERDICT={'UNDER_26GiB_OK' if peak_alloc < 26.0 else 'OVER_26GiB'}")


if __name__ == "__main__":
    main()
