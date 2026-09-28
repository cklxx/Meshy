"""Measure how many parameters one AdamW step actually moves at lr=1e-6.

Uniform-fp16 storage stores parameters and Adam moments in fp16: at |w| ~=
0.02 the fp16 ULP is ~1.5e-5, so an lr=1e-6 AdamW update (~1e-6) is below
half a ULP and rounds away. This script runs two arms on one GPU:

* ``fp16``        : parameters fp16, native fp16 fwd/bwd (old v100 default);
* ``fp32+fsdp16`` : fp32 master parameters with the real production
                    parallelize path (``parallelize_qwen3``) under a
                    world-size-1 FSDP mesh with MixedPrecisionPolicy
                    param_dtype=fp16 + dynamic GradScaler. This verifies
                    FSDP mixed precision applies at degree 1 and records
                    the dtype the first block actually sees in forward.

Single process, no torchrun. Prints and writes JSON::

    python scripts/sm70_optimizer_update.py --flavor 0.6B --out ulp.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.models.qwen3 import model_registry
from torchtitan.tools import utils as tools_utils

MANTISSA = {torch.float16: 10, torch.float32: 23}


def loc(t):
    return t.to_local() if hasattr(t, "to_local") else t


def build_raw(storage: torch.dtype, flavor: str):
    torch.manual_seed(0)
    spec = model_registry(flavor, attn_backend="sdpa")
    with torch.device("meta"), tools_utils.set_default_dtype(storage):
        model = spec.model.build()
    model.to_empty(device="cuda:0")
    with torch.no_grad():
        model.init_states()
    model.train()
    return model, spec


def build_fsdp_fp16_compute(flavor: str, seq_len: int):
    """Production path: fp32 storage, FSDP degree-1 all-gather cast to fp16."""
    torch.manual_seed(0)
    spec = model_registry(flavor, attn_backend="sdpa")
    cfg = spec.model
    cfg.update_from_config(
        trainer_config=type(
            "TC",
            (),
            {
                "parallelism": ParallelismConfig(),
                "training": TrainingConfig(seq_len=seq_len),
                "debug": None,
            },
        )()
    )
    with torch.device("meta"), tools_utils.set_default_dtype(torch.float32):
        model = cfg.build()

    parallel_dims = ParallelDims.from_config(ParallelismConfig(), 1)
    training = TrainingConfig(
        seq_len=seq_len, dtype="float32", mixed_precision_param="float16"
    )
    model = spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=ParallelismConfig(),
        compile_config=CompileConfig(enable=False, components=[]),
        ac_config=ActivationCheckpointConfig(mode="none"),
        dump_folder="/tmp/sm70_ulp_ckpt",
    )
    model.to_empty(device="cuda:0")
    with torch.no_grad():
        model.init_states()
    model.train()
    return model, spec


def measure(model, before):
    """Compare sharded params in place; works for plain tensors and DTensors."""
    total = changed = finite = 0
    ratios = []
    for n, p in model.named_parameters():
        if n not in before:
            continue
        b = loc(before[n]).float()
        v = loc(p).detach().float()
        if tuple(b.shape) != tuple(v.shape):
            continue
        ok = torch.isfinite(v)
        delta = v - b
        moved = (delta != 0) & ok
        total += delta.numel()
        changed += int(moved.sum().item())
        finite += int(ok.sum().item())
        bits = MANTISSA[loc(before[n]).dtype]
        ulp = torch.where(
            b != 0,
            2.0 ** (torch.floor(b.abs().clamp_min(2.0 ** -30).log2()) - bits),
            torch.full_like(b, 2.0 ** (-30 - bits)),
        )
        ratios.append(((delta.abs() / ulp) * ok).cpu().reshape(-1))
    r = torch.cat(ratios) if ratios else torch.empty(0)
    nz = r[r > 0]
    return {
        "params_total": total,
        "params_changed_finite": changed,
        "changed_finite_fraction": changed / max(1, total),
        "param_finite_fraction": finite / max(1, total),
        "delta_over_ulp_median_of_changed": float(nz.median()) if nz.numel() else 0.0,
    }


def run_raw_fp16(flavor: str, seq: int):
    model, spec = build_raw(torch.float16, flavor)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-6, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8,
    )
    ids = torch.randint(0, spec.model.vocab_size, (1, seq), device="cuda:0")
    tgt = torch.randint(0, spec.model.vocab_size, (1, seq), device="cuda:0")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.zero_grad()
    z = model(ids)
    loss = torch.nn.functional.cross_entropy(
        z.reshape(-1, z.size(-1)).float(), tgt.reshape(-1)
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    out = measure(model, before)
    seen = {}

    def hook(_m, args):
        seen.setdefault("qkv_input_dtype", str(args[0].dtype).replace("torch.", ""))

    model.layers["0"].attention.qkv_linear.register_forward_pre_hook(hook)
    with torch.no_grad():
        model(ids)
    out.update(
        storage_dtype="float16",
        compute="native fp16, no scaler",
        skipped_overflow_steps=0,
        grad_scale=1.0,
        forward_dtype=seen.get("qkv_input_dtype", "?"),
        loss=float(loss.detach().float().item()),
    )
    del model, opt
    torch.cuda.empty_cache()
    return out


def run_fsdp_fp32_fp16(flavor: str, seq: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    model, spec = build_fsdp_fp16_compute(flavor, seq)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-6, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8,
    )
    scaler = torch.cuda.amp.GradScaler()
    ids = torch.randint(0, spec.model.vocab_size, (1, seq), device="cuda:0")
    tgt = torch.randint(0, spec.model.vocab_size, (1, seq), device="cuda:0")

    seen = {}

    def hook(m, args):
        seen.setdefault("qkv_input_dtype", str(args[0].dtype).replace("torch.", ""))
        w = next(
            (sm.weight for sm in m.modules() if hasattr(sm, "weight") and sm.weight is not None),
            None,
        )
        if w is not None:
            seen.setdefault("qkv_weight_dtype", str(w.dtype).replace("torch.", ""))

    model.layers["0"].attention.qkv_linear.register_forward_pre_hook(hook)

    skipped = 0
    before_flat = None
    for _ in range(8):
        before_flat = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt.zero_grad()
        # No autocast: Meshy's train_context only enters loss_parallel; fp16
        # comes solely from FSDP unsharding params into param_dtype.
        z = model(ids)
        loss = torch.nn.functional.cross_entropy(
            z.float().reshape(-1, z.size(-1)), tgt.reshape(-1)
        )
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        old = float(scaler.get_scale())
        scaler.step(opt)
        scaler.update()
        if float(scaler.get_scale()) >= old:
            break
        skipped += 1

    seen.clear()
    with torch.no_grad():
        model(ids)
    storage = {str(p.dtype).replace("torch.", "") for p in model.parameters()}
    out = measure(model, before_flat)
    out.update(
        storage_dtype="float32",
        storage_dtypes_seen=sorted(storage),
        compute="FSDP1 mp.param=fp16, no autocast, GradScaler",
        skipped_overflow_steps=skipped,
        grad_scale=float(scaler.get_scale()),
        forward_input_dtype=seen.get("qkv_input_dtype", "?"),
        forward_weight_dtype=seen.get("qkv_weight_dtype", "?"),
        loss=float(loss.detach().float().item()),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flavor", default="0.6B")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a CUDA GPU"
    cap = torch.cuda.get_device_capability()
    report = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": f"{cap[0]}.{cap[1]}",
        "torch": torch.__version__,
        "lr": 1e-6,
        "arms": [],
    }
    for run in (lambda: run_raw_fp16(args.flavor, args.seq),
                lambda: run_fsdp_fp32_fp16(args.flavor, args.seq)):
        arm = run()
        report["arms"].append(arm)
        print(json.dumps(arm), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2)
    for arm in report["arms"]:
        assert 0.0 <= arm["changed_finite_fraction"] <= 1.0


if __name__ == "__main__":
    main()
