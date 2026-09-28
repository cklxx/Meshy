"""Measure how many parameters one AdamW step actually moves at lr=1e-6.

Motivation: uniform-fp16 storage stores parameters *and* Adam moments in
fp16. At |w| ~= 0.02 the fp16 ULP is ~1.5e-5, so an lr=1e-6 AdamW update
(~1e-6) is smaller than half an ULP and rounds away; small fp16 gradients
can underflow as well. This script quantifies both effects on real
torchtitan Qwen3 weights, two arms:

* ``fp16``  : parameters fp16, plain fp16 fwd/bwd (the old v100 default);
* ``fp32``  : fp32 master parameters, fwd/bwd under ``autocast(fp16)`` with
              a dynamic GradScaler (the mixed-precision path, new default).

Single GPU, no distributed init. Prints and writes JSON::

    python scripts/sm70_optimizer_update.py --flavor 0.6B --out /data00/meshy/kern/ulp.json
"""

from __future__ import annotations

import argparse
import json

import torch

from torchtitan.models.qwen3 import model_registry
from torchtitan.tools import utils as tools_utils


def run_arm(dtype: torch.dtype, *, mixed_fp16: bool, flavor: str) -> dict:
    torch.manual_seed(0)
    spec = model_registry(flavor, attn_backend="sdpa")
    with torch.device("meta"), tools_utils.set_default_dtype(dtype):
        model = spec.model.build()
    device = torch.device("cuda:0")
    model.to_empty(device=device)
    with torch.no_grad():
        model.init_states()
    model.train()

    # Same optimiser settings as recipe/grpo_gsm8k_v100.py.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-6, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_fp16)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    input_ids = torch.randint(0, spec.model.vocab_size, (1, 256), device=device)
    labels = torch.randint(0, spec.model.vocab_size, (1, 256), device=device)

    optimizer.zero_grad()
    if mixed_fp16:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    total = changed = 0
    finite_grads = total_grads = 0
    delta_over_ulp_vals: list[torch.Tensor] = []
    mantissa_bits = {torch.float16: 10, torch.float32: 23}[dtype]
    for n, p in model.named_parameters():
        b = before[n]
        delta = (p.detach() - b).float()
        nz = delta != 0
        total += delta.numel()
        changed += int(nz.sum().item())
        if p.grad is not None:
            total_grads += p.grad.numel()
            finite_grads += int(torch.isfinite(p.grad).sum().item())
        # Per-element ULP of the storage dtype at |b|: 2^(floor(log2|b|) - m).
        with torch.no_grad():
            ulp = torch.where(
                b.float() != 0,
                2.0 ** (torch.floor(torch.log2(b.float().abs().clamp_min(2.0 ** -30))) - mantissa_bits),
                torch.full_like(b.float(), 2.0 ** (-30 - mantissa_bits)),
            )
            delta_over_ulp_vals.append((delta.abs() / ulp).cpu())

    ratio = torch.cat([v.reshape(-1) for v in delta_over_ulp_vals])
    result = {
        "storage_dtype": "float32" if dtype is torch.float32 else "float16",
        "compute": "fp16-autocast+scaler" if mixed_fp16 else "native storage dtype",
        "loss_finite": bool(torch.isfinite(loss).item()),
        "loss": float(loss.detach().float().item()),
        "params_total": total,
        "params_changed": changed,
        "changed_fraction": changed / max(1, total),
        "grad_finite_fraction": finite_grads / max(1, total_grads),
        "delta_over_ulp_median": float(ratio.median().item()),
        "delta_over_ulp_p90": float(torch.quantile(ratio, 0.9).item()),
        "grad_scale": float(scaler.get_scale()),
    }
    del model, optimizer, before
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flavor", default="0.6B")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "this measurement needs the V100"
    cap = torch.cuda.get_device_capability()
    report = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": f"{cap[0]}.{cap[1]}",
        "torch": torch.__version__,
        "arms": [
            run_arm(torch.float16, mixed_fp16=False, flavor=args.flavor),
            run_arm(torch.float32, mixed_fp16=True, flavor=args.flavor),
        ],
    }
    print(json.dumps(report, indent=2))
    for arm in report["arms"]:
        assert 0.0 <= arm["changed_fraction"] <= 1.0
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
