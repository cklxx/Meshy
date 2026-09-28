#!/usr/bin/env python3
"""GPU-boundary check: MATH vs EFFICIENT SDPA log-prob agreement on sm70.

Run ONLY at a step boundary (idle GPU). Builds one fp16 batch of realistic
Qwen3 shapes, runs causal SDPA under each backend, and reports the max abs
diff of the attention output. Gate: must be <= 1e-2 before enabling the
mem-efficient path in training.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(q, k, v, backend):
    with sdpa_kernel(backend):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    return out


def main() -> None:
    assert torch.cuda.get_device_capability(0) == (7, 0), "sm70 only"
    torch.manual_seed(0)
    worst = 0.0
    # (batch, heads, seqlen, head_dim): Qwen3-0.6B = 16 q heads, dim 128
    for b, s in ((1, 5120), (4, 1024), (8, 768), (16, 512)):
        h, d = 16, 128
        q = torch.randn(b, h, s, d, dtype=torch.float16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        o_math = run(q, k, v, SDPBackend.MATH)
        o_eff = run(q, k, v, SDPBackend.EFFICIENT_ATTENTION)
        # compare only non-pad (whole row here), fp16 realistic
        diff = (o_math.float() - o_eff.float()).abs().max().item()
        mean = o_math.float().abs().mean().item()
        print(f"shape b={b} s={s}: max_abs_diff={diff:.5f} ref_mean_abs={mean:.4f}")
        worst = max(worst, diff)
        del q, k, v, o_math, o_eff
        torch.cuda.empty_cache()
    print(f"WORST_MAX_ABS_DIFF={worst:.5f} GATE=1e-2 -> {'PASS' if worst <= 1e-2 else 'FAIL'}")
    sys.exit(0 if worst <= 1e-2 else 1)


if __name__ == "__main__":
    main()
