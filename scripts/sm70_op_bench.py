"""sm70 native-op microbenchmark at Qwen3-0.6B shapes.

Times the pure-torch fused ops that replace sgl_kernel on sm70, so kern can
prioritise TileLang replacements by wall-clock share. Shapes from
Qwen3-0.6B: hidden=1024, ffn=3072, 28 layers, head_dim=128, 16 q / 8 kv heads.

Usage: python sm70_op_bench.py [tokens]
"""
import sys
import torch

torch.manual_seed(0)
dev = "cuda"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
H, F_, HD, NH, NKV = 1024, 3072, 128, 16, 8
DT = torch.float16
ITERS = 200


def bench(name, fn, *args):
    for _ in range(20):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / ITERS
    return ms


# RMSNorm (matches SGLang RMSNorm.forward_native)
w = torch.randn(H, device=dev, dtype=torch.float32)
x = torch.randn(N, H, device=dev, dtype=DT)


def rmsnorm(x, w, eps=1e-6):
    xf = x.float()
    v = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(v + eps) * w).to(DT)


# SiLUAndMul
g = torch.randn(N, 2 * F_, device=dev, dtype=DT)


def silu_mul(g):
    d = g.shape[-1] // 2
    return torch.nn.functional.silu(g[..., :d]) * g[..., d:]


# fused_add_rmsnorm equivalent (residual path)
res = torch.randn(N, H, device=dev, dtype=DT)


def fused_add_rmsnorm(x, res, w, eps=1e-6):
    xf = x.float() + res.float()
    v = xf.pow(2).mean(-1, keepdim=True)
    xo = (xf * torch.rsqrt(v + eps) * w).to(DT)
    return xo, xf.to(DT)


# rotary (native): rotate_half + cos/sin on q,k at decode-ish and prefill shape
pos = torch.arange(N, device=dev)
cos = torch.randn(N, HD, device=dev, dtype=DT)
sin = torch.randn(N, HD, device=dev, dtype=DT)
q = torch.randn(N, NH, HD, device=dev, dtype=DT)
k = torch.randn(N, NKV, HD, device=dev, dtype=DT)


def rotate_half(t):
    return torch.stack((-t[..., 1::2], t[..., ::2]), dim=-1).flatten(-2)


def rope(q, k, cos, sin):
    return q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1), \
           k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)


res_table = []
res_table.append(("rmsnorm", bench("rmsnorm", rmsnorm, x, w)))
res_table.append(("silu_and_mul", bench("silu", silu_mul, g)))
res_table.append(("fused_add_rmsnorm", bench("fused", fused_add_rmsnorm, x, res, w)))
res_table.append(("rope(q+k)", bench("rope", rope, q, k, cos, sin)))

# SDPA decode attention: single query token over growing KV is the hot path;
# benchmark a representative decode step KV len = N
q1 = torch.randn(1, NH, 1, HD, device=dev, dtype=DT)
k1 = torch.randn(1, NKV, N, HD, device=dev, dtype=DT).repeat_interleave(NH // NKV, 1)
v1 = torch.randn(1, NKV, N, HD, device=dev, dtype=DT).repeat_interleave(NH // NKV, 1)


def sdpa_decode(q1, k1, v1):
    return torch.nn.functional.scaled_dot_product_attention(q1, k1, v1, is_causal=False)


res_table.append(("sdpa_decode_kv%d" % N, bench("sdpa", sdpa_decode, q1, k1, v1)))

# SDPA prefill attention N tokens (causal)
qp = torch.randn(1, NH, N, HD, device=dev, dtype=DT)
kp = torch.randn(1, NH, N, HD, device=dev, dtype=DT)
vp = torch.randn(1, NH, N, HD, device=dev, dtype=DT)


def sdpa_prefill(qp, kp, vp):
    return torch.nn.functional.scaled_dot_product_attention(qp, kp, vp, is_causal=True)


res_table.append(("sdpa_prefill_n%d" % N, bench("sdpa_p", sdpa_prefill, qp, kp, vp)))

total = sum(ms for _, ms in res_table)
print(f"tokens/shape N={N}, fp16, {ITERS} iters, ms per call")
for name, ms in res_table:
    print(f"  {name:22s} {ms:8.4f} ms   {100*ms/total:5.1f}% of listed")
print("OP_BENCH_DONE")
