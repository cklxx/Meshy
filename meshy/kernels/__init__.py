"""sm70 (Volta) TileLang kernels for the SGLang hot path.

Three fp16 ops whose sgl-kernel/flashinfer implementations ship sm75+
binaries only (see docs/sm70_audit.md):

* :func:`rmsnorm` — out = (x_norm32 * w32).to(fp16), fp32 mean-square
  accumulation, matching SGLang ``RMSNorm.forward_native`` with
  ``cast_x_before_out_mul=False``;
* :func:`fused_add_rmsnorm` — in-place: residual16 = (x + residual).to16,
  x = normalised fused sum times weight; same fp32 accumulation;
* :func:`silu_and_mul` — out = silu(x[..., :D]) * x[..., D:].

One program per row. All shapes are compile-time specialisations (the four
token counts times Qwen3-0.6B widths), so the reduction needs no
cross-block sync; programs are cached per shape. Tensor shapes are declared
as annotation statements inside the body after ``T.const`` (tilelang
prim_func convention), not as Python parameter annotations.
"""

from __future__ import annotations

import tilelang
import tilelang.language as T

_THREADS = 256
_cache: dict[tuple, object] = {}


def _make_rmsnorm(eps: float):
    @tilelang.jit(pass_configs={"tl.disable_tma_lower": True})
    def kernel(X, W, Out):
        M, N = T.const("M, N")
        X: T.Tensor((M, N), T.float16)
        W: T.Tensor((N,), T.float16)
        Out: T.Tensor((M, N), T.float16)
        with T.Kernel(M, threads=_THREADS) as row:
            x16 = T.alloc_fragment((N,), T.float16)
            xf = T.alloc_fragment((N,), T.float32)
            sq = T.alloc_fragment((N,), T.float32)
            powsum = T.alloc_fragment((1,), T.float32)
            T.copy(X[row, :], x16)
            for j in T.Parallel(N):
                xf[j] = T.cast(x16[j], "float32")
                sq[j] = xf[j] * xf[j]
            T.reduce_sum(sq, powsum)
            scale = T.rsqrt(powsum[0] / N + eps)
            for j in T.Parallel(N):
                w = T.cast(W[j], "float32")
                x16[j] = T.cast(xf[j] * scale * w, "float16")
            T.copy(x16, Out[row, :])

    return kernel


def _make_fused_add_rmsnorm(eps: float):
    @tilelang.jit(pass_configs={"tl.disable_tma_lower": True})
    def kernel(X, R, W):
        M, N = T.const("M, N")
        X: T.Tensor((M, N), T.float16)
        R: T.Tensor((M, N), T.float16)
        W: T.Tensor((N,), T.float16)
        with T.Kernel(M, threads=_THREADS) as row:
            x16 = T.alloc_fragment((N,), T.float16)
            r16 = T.alloc_fragment((N,), T.float16)
            s = T.alloc_fragment((N,), T.float32)
            sq = T.alloc_fragment((N,), T.float32)
            powsum = T.alloc_fragment((1,), T.float32)
            T.copy(X[row, :], x16)
            T.copy(R[row, :], r16)
            for j in T.Parallel(N):
                s[j] = T.cast(x16[j], "float32") + T.cast(r16[j], "float32")
                sq[j] = s[j] * s[j]
                r16[j] = T.cast(s[j], "float16")
            T.copy(r16, R[row, :])
            T.reduce_sum(sq, powsum)
            scale = T.rsqrt(powsum[0] / N + eps)
            for j in T.Parallel(N):
                w = T.cast(W[j], "float32")
                x16[j] = T.cast(s[j] * scale * w, "float16")
            T.copy(x16, X[row, :])

    return kernel


def _make_silu_and_mul():
    @tilelang.jit(pass_configs={"tl.disable_tma_lower": True})
    def kernel(X, Out):
        M, D = T.const("M, D")
        X: T.Tensor((M, 2 * D), T.float16)
        Out: T.Tensor((M, D), T.float16)
        with T.Kernel(M, threads=_THREADS) as row:
            o16 = T.alloc_fragment((D,), T.float16)
            for j in T.Parallel(D):
                a = T.cast(X[row, j], "float32")
                g = T.cast(X[row, D + j], "float32")
                o16[j] = T.cast(a * (1.0 / (1.0 + T.exp(-a))) * g, "float16")
            T.copy(o16, Out[row, :])

    return kernel


def _ker(key, **dims):
    ker = _cache.get(key)
    if ker is None or not callable(ker):
        name = key[0]
        if name == "rms":
            ker = _make_rmsnorm(key[-1]).compile(**dims)
        elif name == "fused":
            ker = _make_fused_add_rmsnorm(key[-1]).compile(**dims)
        else:
            ker = _make_silu_and_mul().compile(**dims)
        _cache[key] = ker
    return ker


def rmsnorm(x, weight, eps: float = 1e-6, out=None):
    """FP16 weighted RMSNorm, fp32 accumulation. ``x``: [M, N].

    Drop-in for ``sgl_kernel.rmsnorm`` (``out`` written in place when given).
    """
    import torch

    m, n = x.shape
    ker = _ker(("rms", m, n, float(eps)), M=m, N=n)
    if out is None:
        out = torch.empty_like(x)
    ker(x.contiguous(), weight.contiguous(), out)
    return out


def fused_add_rmsnorm(x, residual, weight, eps: float = 1e-6):
    """In-place fused residual add + RMSNorm (SGLang sgl_kernel semantics).

    After the call ``residual`` holds the fp16-cast pre-norm sum and ``x``
    holds the normalised, weight-multiplied output. Returns ``(x, residual)``.
    """
    m, n = x.shape
    ker = _ker(("fused", m, n, float(eps)), M=m, N=n)
    ker(x.contiguous(), residual.contiguous(), weight.contiguous())
    return x, residual


def silu_and_mul(x):
    """FP16 SiLU(SwiGLU) activation on the last dim split in half."""
    import torch

    m = x.shape[0]
    d = x.shape[-1] // 2
    ker = _ker(("silu", m, d), M=m, D=d)
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    ker(x.contiguous(), out)
    return out
