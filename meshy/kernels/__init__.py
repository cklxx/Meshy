"""sm70 (Volta) TileLang kernels for the SGLang hot path.

Three fp16 ops whose sgl-kernel/flashinfer implementations ship sm75+
binaries only (see docs/sm70_audit.md):

* :func:`rmsnorm` — out = (x_norm32 * w32).to(fp16), fp32 mean-square
  accumulation, matching SGLang ``RMSNorm.forward_native`` with
  ``cast_x_before_out_mul=False``;
* :func:`fused_add_rmsnorm` — in-place: residual16 = (x + residual).to16,
  x = normalised fused sum times weight; same fp32 accumulation;
* :func:`silu_and_mul` — out = silu(x[..., :D]) * x[..., D:].

The compiled kernel shape ABI is fixed at compile time (this tilelang
build cannot launch one kernel with a runtime row count), so callers go
through **row buckets**: the input is padded up to the next bucket M, the
per-bucket kernel runs, and the result is sliced back. Buckets are the
cost model: padding overhead stays below 2x per call while the kernels
are 4-15x faster than native (docs/sm70_tilelang_kernels.md). Compile
every bucket once at server start via :func:`prewarm` so the first
request never pays JIT.

Tensor shapes are annotation statements inside the body after ``T.const``
(tilelang prim_func convention).
"""

from __future__ import annotations

import bisect

# tilelang is imported lazily so CPU-side helpers (select_bucket) and the
# sm70 patch can import this module on hosts without a compiled tilelang.
tilelang = None
T = None

_THREADS = 256


def _tl():
    global tilelang, T
    if T is None:
        import tilelang as _tilelang
        import tilelang.language as _T

        tilelang = _tilelang
        T = _T
    return tilelang, T
#: Row buckets, dense at small decode sizes, sparse above. 15 buckets x
#: 3 ops = 45 JIT kernels per prewarm; above the max bucket the patch
#: falls back to forward_native, so a long chunk never crashes serve.
DEFAULT_BUCKETS: tuple[int, ...] = (
    1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192,
)
_cache: dict[tuple, object] = {}


def select_bucket(m: int, buckets=DEFAULT_BUCKETS) -> int:
    """Smallest bucket >= m; raises when above the largest precompiled size."""
    if m <= 0:
        raise ValueError(f"row count must be positive, got {m}")
    idx = bisect.bisect_left(buckets, m)
    if idx >= len(buckets):
        raise ValueError(
            f"row count {m} exceeds largest TileLang bucket {buckets[-1]}"
        )
    return buckets[idx]


def _make_rmsnorm(eps: float):
    tilelang, T = _tl()
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
    tilelang, T = _tl()
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
    tilelang, T = _tl()
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
    """Return a compiled kernel for key ``(name, bucket_M, N[, eps])``."""
    ker = _cache.get(key)
    if ker is None:
        name = key[0]
        if name == "rms":
            ker = _make_rmsnorm(key[-1]).compile(**dims)
        elif name == "fused":
            ker = _make_fused_add_rmsnorm(key[-1]).compile(**dims)
        else:
            ker = _make_silu_and_mul().compile(**dims)
        _cache[key] = ker
    return ker


def _pad2d(x, m_bucket):
    import torch

    pad = m_bucket - x.shape[0]
    if pad:
        return torch.nn.functional.pad(x, (0, 0, 0, pad))
    return x


def rmsnorm(x, weight, eps: float = 1e-6, out=None):
    """FP16 weighted RMSNorm, fp32 accumulation, row-bucketed.

    Drop-in for ``sgl_kernel.rmsnorm`` (``out`` written in place when given).
    ``x`` must be 2D ``[M, N]`` fp16.
    """
    import torch

    m, n = x.shape
    mb = select_bucket(m)
    ker = _ker(("rms", mb, n, float(eps)), M=mb, N=n)
    xp = x if mb == m else _pad2d(x.contiguous(), mb)
    outp = out if (out is not None and mb == m) else torch.empty(
        (mb, n), dtype=x.dtype, device=x.device
    )
    ker(xp, weight.contiguous(), outp)
    result = outp[:m]
    if out is not None and mb != m:
        out.copy_(result)
        return out
    return result


def fused_add_rmsnorm(x, residual, weight, eps: float = 1e-6):
    """In-place fused residual add + RMSNorm (sgl_kernel semantics).

    After the call ``residual`` holds the fp16-cast pre-norm sum and ``x``
    holds the normalised, weight-multiplied output. Inputs are 2D fp16.
    """
    m, n = x.shape
    mb = select_bucket(m)
    ker = _ker(("fused", mb, n, float(eps)), M=mb, N=n)
    if mb == m:
        ker(x.contiguous(), residual.contiguous(), weight.contiguous())
        return x, residual
    # Padded work buffers; write only the valid rows back (in-place ABI).
    import torch

    xp = _pad2d(x.contiguous(), mb)
    rp = _pad2d(residual.contiguous(), mb)
    ker(xp, rp, weight.contiguous())
    x.copy_(xp[:m])
    residual.copy_(rp[:m])
    return x, residual


def silu_and_mul(x, out=None):
    """FP16 SiLU(SwiGLU) on the last dim split in half, row-bucketed."""
    import torch

    m = x.shape[0]
    d = x.shape[-1] // 2
    mb = select_bucket(m)
    ker = _ker(("silu", mb, d), M=mb, D=d)
    xp = x if mb == m else torch.nn.functional.pad(x.contiguous(), (0, 0, 0, mb - m))
    outp = torch.empty((mb, d), dtype=x.dtype, device=x.device)
    ker(xp, outp)
    result = outp[:m]
    if out is not None:
        out.copy_(result)
        return out
    return result.contiguous()


def prewarm(*, norm_widths, silu_halves, buckets=DEFAULT_BUCKETS, eps: float = 1e-6):
    """Compile every (op, bucket, width) kernel; call before serving."""
    compiled = []
    for n in norm_widths:
        for mb in buckets:
            _ker(("rms", mb, int(n), float(eps)), M=mb, N=int(n))
            _ker(("fused", mb, int(n), float(eps)), M=mb, N=int(n))
            compiled.append(("rmsnorm", mb, int(n)))
            compiled.append(("fused_add_rmsnorm", mb, int(n)))
    for d in silu_halves:
        for mb in buckets:
            _ker(("silu", mb, int(d)), M=mb, D=int(d))
            compiled.append(("silu_and_mul", mb, int(d)))
    return compiled
