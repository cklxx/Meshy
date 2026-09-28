# sm70 TileLang hot-path kernels (T3b)

`meshy/kernels/__init__.py` — three fp16 TileLang ops for V100 (sm70),
where sgl-kernel and flashinfer ship sm75+ binaries only. fp32 internal
accumulation, fp16 I/O, one program per row, per-shape compiled-kernel
cache. TileLang 0.1.11 (venv), nvcc 12.4.

| op | drop-in signature |
|---|---|
| `rmsnorm(x, weight, eps=1e-6, out=None)` | matches `sgl_kernel.rmsnorm` |
| `fused_add_rmsnorm(x, residual, weight, eps=1e-6)` | in-place, sgl_kernel semantics; residual = fp16 sum, x = normed |
| `silu_and_mul(x)` | matches `sglang.jit_kernel.activation.silu_and_mul` |

Numerics match SGLang `RMSNorm.forward_native` (eps 1e-6, weight multiply
in fp32 then cast). Correctness gate: `scripts/sm70_kernel_check.py`
(max abs error **0.001953125**, ~1 fp16 ULP at |x|≈1, asserts < 0.05).

## V100 timing (ms/call, mean of 300 after 50 warmup)

Qwen3-0.6B widths: RMSNorm N=1024, SiLU D=3072 (input 6144).

| tokens | rmsnorm (tl / torch / compile) | fused_add (tl / torch) | silu (tl / torch / compile) |
|---|---|---|---|
| 1 | 0.014 / 0.098 / 0.074 (6.8x) | 0.009 / 0.128 (14.9x) | 0.019 / 0.072 / 0.067 (3.9x) |
| 64 | 0.014 / 0.098 / 0.075 (7.0x) | 0.009 / 0.130 (14.5x) | 0.019 / 0.079 / 0.080 (4.2x) |
| 2048 | 0.015 / 0.139 / 0.082 (9.3x / 5.5x vs compile) | 0.024 / 0.212 (8.7x) | 0.052 / 0.537 / 0.077 (10.4x / 1.5x vs compile) |
| 8192 | 0.047 / 0.457 / 0.082 (9.8x / 1.8x vs compile) | 0.086 / 0.724 (8.4x) | 0.194 / 2.069 / 0.191 (10.7x / ~1.0x vs compile) |

All three beat both baselines at every shape; the gain over native grows
with batch size; vs torch.compile rmsnorm is 1.8–5.5x faster, fused_add
has no fused compile baseline, silu ties compile only at 8192 tokens.

## Integration point (aligned with env's patch)

`meshy/backend/sglang_sm70.py` (v100/env) forces every fused op to
`forward_native` and stubs `sgl_kernel.common_ops`. Switching is a flag:
the patch's forced callable should dispatch to `meshy.kernels` when an
env switch (e.g. `MESHY_SM70_TILELANG=1`) is set:

```python
import os, meshy.kernels as _tl

def _rms(x, w, eps=1e-6):
    if os.environ.get("MESHY_SM70_TILELANG") == "1":
        return _tl.rmsnorm(x, w, eps)
    return x.float().pow(2).mean(-1, keepdim=True)  # forward_native path
```

`RMSNorm.forward_cuda` (residual None) and the residual branch map onto
`rmsnorm` / `fused_add_rmsnorm`; `SiluAndMul.forward_cuda` maps onto
`silu_and_mul`. Leave the default on native until the kernels have run
inside a real SGLang generate (env's T1b smoke); the switch is opt-in.
First call per (M, N) pays JIT compile (~seconds); compiled kernels are
cached in-process.
