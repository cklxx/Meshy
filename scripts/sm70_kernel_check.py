"""Correctness + speed check for meshy.kernels sm70 TileLang ops.

Compares against SGLang forward_native semantics (fp32 accumulate,
eps=1e-6, weight multiply in fp32 then cast to fp16):

* rmsnorm / fused_add_rmsnorm on hidden 1024 (Qwen3-0.6B norm width);
* silu_and_mul on D=3072 (Qwen3-0.6B intermediate width);
* token counts 1, 33, 64, 2048, 6400, 8192 (odd sizes exercise the
  bucket padding path).

Reports max abs error and timing vs native and torch.compile. Run on the
V100::

    PATH=/usr/local/cuda-12.4/bin:$PATH \
      /data00/meshy/venv/bin/python scripts/sm70_kernel_check.py --out bench.json

CPU mode runs only the bucket-selection unit checks (no TileLang/CUDA).
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from meshy.kernels import (
    DEFAULT_BUCKETS,
    fused_add_rmsnorm,
    rmsnorm,
    select_bucket,
    silu_and_mul,
)

EPS = 1e-6
TOKENS = (1, 33, 64, 2048, 6400, 8192)


def ref_rmsnorm(x, weight, eps=EPS):
    xf = x.float()
    h = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (h * weight.float()).to(torch.float16)


def ref_fused(x, residual, weight, eps=EPS):
    s = x.float() + residual.float()
    h = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    return (h * weight.float()).to(torch.float16), s.to(torch.float16)


def ref_silu(x):
    d = x.shape[-1] // 2
    a, b = x.float()[..., :d], x.float()[..., d:]
    return (a * torch.sigmoid(a) * b).to(torch.float16)


def bench(fn, args, iters=300, warmup=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def check_rms(n, m):
    torch.manual_seed(0)
    x = torch.randn(m, n, device="cuda", dtype=torch.float16) * 0.5
    w = torch.randn(n, device="cuda", dtype=torch.float16) * 0.1 + 1
    out = rmsnorm(x, w, EPS)
    ref = ref_rmsnorm(x, w)
    err = (out.float() - ref.float()).abs()
    return {
        "op": "rmsnorm", "tokens": m, "n": n,
        "bucket": select_bucket(m),
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "tilelang_ms": bench(rmsnorm, (x, w, EPS)),
        "native_ms": bench(ref_rmsnorm, (x, w)),
    }


def check_fused(n, m):
    torch.manual_seed(1)
    x = torch.randn(m, n, device="cuda", dtype=torch.float16) * 0.5
    r = torch.randn(m, n, device="cuda", dtype=torch.float16) * 0.5
    w = torch.randn(n, device="cuda", dtype=torch.float16) * 0.1 + 1
    ref_x, ref_r = ref_fused(x, r, w)
    xi, ri = x.clone(), r.clone()
    out_x, out_r = fused_add_rmsnorm(xi, ri, w, EPS)
    ex = (out_x.float() - ref_x.float()).abs()
    er = (out_r.float() - ref_r.float()).abs()
    return {
        "op": "fused_add_rmsnorm", "tokens": m, "n": n,
        "bucket": select_bucket(m),
        "max_abs_err": float(max(ex.max(), er.max())),
        "mean_abs_err": float(torch.cat([ex.reshape(-1), er.reshape(-1)]).mean()),
        "tilelang_ms": bench(lambda a, b: fused_add_rmsnorm(a, b, w, EPS), (x, r)),
        "native_ms": bench(lambda a, b: ref_fused(a, b, w), (x, r)),
    }


def check_silu(d, m):
    torch.manual_seed(2)
    x = torch.randn(m, 2 * d, device="cuda", dtype=torch.float16) * 0.5
    out = silu_and_mul(x)
    ref = ref_silu(x)
    err = (out.float() - ref.float()).abs()
    return {
        "op": "silu_and_mul", "tokens": m, "n": d,
        "bucket": select_bucket(m),
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "tilelang_ms": bench(silu_and_mul, (x,)),
        "native_ms": bench(ref_silu, (x,)),
    }


def test_bucket_logic():
    """CPU-runnable self-check for the padding size selection."""
    assert select_bucket(1) == 1
    assert select_bucket(64) == 64
    assert select_bucket(65) == 96
    assert select_bucket(2048) == 2048
    assert select_bucket(8192) == 8192
    for too_big in (8193, 10000):
        try:
            select_bucket(too_big)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError above max bucket")
    assert DEFAULT_BUCKETS == tuple(sorted(DEFAULT_BUCKETS))
    print("BUCKET_LOGIC_PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--cpu", action="store_true",
        help="run only CPU-side bucket logic checks (no CUDA kernels)",
    )
    args = ap.parse_args()
    test_bucket_logic()
    if args.cpu:
        return
    torch.cuda.set_device(0)
    rows = []
    for m in TOKENS:
        rows.append(check_rms(1024, m))
        rows.append(check_fused(1024, m))
        rows.append(check_silu(3072, m))

    c_rms = torch.compile(ref_rmsnorm)
    c_silu = torch.compile(ref_silu)
    for m in TOKENS:
        x = torch.randn(m, 1024, device="cuda", dtype=torch.float16) * 0.5
        w = torch.randn(1024, device="cuda", dtype=torch.float16) * 0.1 + 1
        rows.append({
            "op": "rmsnorm_compile", "tokens": m, "n": 1024,
            "bucket": select_bucket(m),
            "max_abs_err": None, "mean_abs_err": None,
            "tilelang_ms": None, "native_ms": bench(c_rms, (x, w)),
        })
        x2 = torch.randn(m, 6144, device="cuda", dtype=torch.float16) * 0.5
        rows.append({
            "op": "silu_compile", "tokens": m, "n": 6144,
            "bucket": select_bucket(m),
            "max_abs_err": None, "mean_abs_err": None,
            "tilelang_ms": None, "native_ms": bench(c_silu, (x2,)),
        })

    report = {
        "device": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability())),
        "torch": torch.__version__,
        "buckets": list(DEFAULT_BUCKETS),
        "rows": rows,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)

    worst = max(r["max_abs_err"] for r in rows if r["max_abs_err"] is not None)
    assert worst < 0.05, f"max abs error {worst} exceeds fp16 tolerance"
    print("SM70_KERNEL_CHECK_PASS worst_max_abs_err=", worst)


if __name__ == "__main__":
    main()
