# sm70 (V100) SGLang compatibility — patch, results, operator profile

Tesla V100-SXM2-32GB, compute capability **7.0 (sm_70)**, host driver 535/CUDA 12.2,
nvcc 12.4. Env: torch 2.13.0+cu126 (arch_list sm_50..sm_90, **sm_70 present**),
sglang **0.5.18**, sglang-kernel **0.4.6.post1**, Python 3.12.14. Run 2026-09-28.

**Status: serving works.** A runtime patch (`meshy/backend/sglang_sm70.py`,
applied automatically by `SGLangService` on capability (7,0)) makes SGLang
0.5.18 serve Qwen3-0.6B fp16 end-to-end on V100 — correct generation, both
torch_native (SDPA) and triton attention, pytorch sampling, eager graphs, and
memory-saver release/resume. No site-packages files are edited.

## Upstream floors the patch bypasses

Three independent sm75+ floors, all handled in the patch:

1. **SGLang gate** — `load_model_utils.maybe_downgrade_dtype_for_legacy_gpu:79`
   raises "SGLang only supports sm75 and above." Replaced with the dtype-only
   downgrade (fp16) it does for sm70..sm79, no raise.
2. **sglang-kernel has no sm70 binary** — all wheels (0.4.x sm90/sm100; legacy
   sgl-kernel 0.3.21 cuobjdump-verified sm80+; no sdist). A lazy
   `sys.modules["common_ops"]` stub lets `import sgl_kernel` and its Python
   submodules load; every hot-path CustomOp is forced to pure-torch
   `forward_native` via SGLang's own `set_fused_op_backend(KernelBackend.TORCH)`
   (`SGLANG_FORCE_FUSED_OP_BACKEND=native`), so the stub is never executed.
3. **FlashInfer floor ("requires sm75")** — sidestepped: attention uses
   `--attention-backend torch_native` (pure SDPA) or `triton`; sampling uses
   `--sampling-backend pytorch`. Neither imports flashinfer on the hot path.

Plus an environment hazard: `sgl_kernel._preload_cuda_library` dlopens the
system CUDA-home runtime (12.4) with RTLD_GLOBAL; it predates
`cudaGetDriverEntryPointByVersion`, so torch's libc10_cuda fails to bind
(undefined symbol) when memory saver runs. The patch preloads the torch
wheel's own CUDA 12.6 runtime first (RTLD_GLOBAL), fixing the binding.

## Acceptance results (Qwen3-0.6B fp16, single V100, GPU exclusive)

Correctness: both attention backends answer GSM8K correctly (16-3-4=9, 9*2=
**$18**); greedy "capital of France" -> "Paris".

| metric | torch_native (SDPA) | triton |
|---|---|---|
| single-stream decode (128 tok, incl prefill) | 16.38 tok/s | **20.10 tok/s** |
| batch 64 aggregate (8192 tok) | 60.50 tok/s | **601.40 tok/s** |
| memory release (weights+kv) | 28418 -> 510 MiB | 29612 -> 1708 MiB |
| generation after resume | Paris (correct) | Paris (correct) |

triton is the recommended sm70 backend: single-stream 1.2x and batch-64 ~10x
faster than torch_native SDPA.

Memory saver requires `--enable-weights-cpu-backup`: without it released
weights have no host copy and resume runs on garbage (token 0 / "!!!!"). Resume
is asynchronous; a caller must poll until output is correct.

## Native-op profile at Qwen3-0.6B shapes (fp16, N=4096, 200 iters)

Microbenchmark `scripts/sm70_op_bench.py`: per-call ms and share of the listed
ops. Attention (SDPA) dominates; the three elementwise ops kern replaced are
~22% combined at this shape.

| op | ms/call | share |
|---|---|---|
| sdpa_prefill N=4096 (causal) | 2.067 | 54.9% |
| rope(q+k) | 0.563 | 15.0% |
| fused_add_rmsnorm | 0.383 | 10.2% |
| sdpa_decode KV=4096 | 0.292 | 7.7% |
| rmsnorm | 0.245 | 6.5% |
| silu_and_mul | 0.212 | 5.6% |

kern's TileLang rmsnorm/fused_add_rmsnorm/silu_and_mul
(`meshy/kernels/__init__.py`, v100/kern ee8b168) are 3.9-14.9x faster than
these torch-native kernels; the patch will select them via a
`MESHY_SM70_TILELANG` switch (TODO: wire the switch into the sm70 backend).
Attention remains SDPA/triton (no TileLang replacement scoped there yet).

## What runs (training side, same venv)

- torchtitan Qwen3-0.6B one fwd+bwd step fp32+fp16: PASS.
- CPU pytest subset (17 files): 87 passed.

## Reproduce

```bash
# serve + full acceptance (correctness, tok/s, release/resume):
bash /data00/meshy/env/sm70_accept.sh                    # torch_native
ATTENTION_BACKEND=triton PORT=30121 bash sm70_accept.sh  # triton
# minimal Paris smoke:
bash /data00/meshy/env/sm70_sglang_patched.sh
# operator profile:
/data00/meshy/venv/bin/python scripts/sm70_op_bench.py 4096
```

