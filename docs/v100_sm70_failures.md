# sm70 (V100) operator/import failure list — for TileLang replacement

Tesla V100-SXM2-32GB, compute capability **7.0 (sm_70)**, host driver 535/CUDA 12.2,
nvcc 12.4. Env: torch 2.13.0+cu126 (arch_list sm_50..sm_90, **sm_70 present**),
sglang **0.5.18**, sglang-kernel **0.4.6.post1** (its pin) / also tested 0.4.7,
flashinfer-python 0.6.x, Python 3.12.14. Run 2026-09-28.

## Headline: the whole SGLang kernel stack floors at sm75

SGLang 0.5.18 cannot serve on V100. This is not one missing operator; three
independent layers each hard-require sm75+:

1. **SGLang itself** — explicit gate before any model op:
   `sglang/srt/model_executor/model_runner_components/load_model_utils.py:79`
   in `maybe_downgrade_dtype_for_legacy_gpu`:
   ```python
   if torch.cuda.get_device_capability()[1] < 5:
       raise RuntimeError("SGLang only supports sm75 and above.")
   ```
   (sm70 minor version is 0 < 5; sm75 is 7,5.)

2. **sglang-kernel ships no sm70 binaries.** Verified by listing the wheels and
   with `cuobjdump --list-elf` (nvcc 12.4):
   - 0.4.6.post1 / 0.4.7 (pypi): only `sgl_kernel/sm90/common_ops.so` and
     `sgl_kernel/sm100/common_ops.so` — no sm70.
   - legacy `sgl-kernel` 0.3.21: `common_ops` is sm90/sm100; `flash_ops.so`
     fat binary is sm80/sm86/sm90; `spatial_ops.so` sm80/sm89/sm90/sm100/sm120.
     Still no sm70.
   - No sdist is published (wheels only), so there is no `pip` source build.
   On import, `sgl_kernel/__init__.py` calls
   `_load_architecture_specific_ops()`, which raises on sm70:
   ```
   ImportError:
   Attempted locations:
   1. .../sgl_kernel/sm100/common_ops.* (only sm100 present)
   2. Fallback .../sgl_kernel/common_ops.* - found files: []
   GPU Info:
   - Compute capability: 70
   - Expected variant: SM70 (precise math for compatibility)
   - CUDA version: 12.6
   Error details: libnvrtc.so.13: cannot open shared object file
   ```
   The loader *names* "SM70 (precise math)" but no wheel ships the variant.

3. **FlashInfer floors at sm75.** Even bypassing (1) and stubbing (2), the
   forward pass dies at the first RMSNorm — sgl_kernel's Python wrapper falls
   back to flashinfer, whose JIT rejects the arch:
   `flashinfer/jit/core.py:109` `check_cuda_arch()`:
   `RuntimeError: FlashInfer requires GPUs with sm75 or higher`.

Downgrading SGLang does not help: 0.5.10+ hard-pins sglang-kernel; 0.5.1
pins legacy sgl-kernel (sm80+ per cuobjdump); flashinfer is the same sm75 wall
regardless of SGLang version. The prebuilt-kernel assumption "sm75 and above"
holds across the stack.

## Hot-path operators that need an sm70 implementation

Diagnostic method: with the sm75 gate bypassed and the sgl_kernel native loader
replaced by a lazy dummy (so imports succeed), the fp16 Qwen3-0.6B model loads
and the forward begins. The first call into a missing kernel is the order below.
Each native op is invoked as `torch.ops.sgl_kernel.<name>`; replacing the
RMSNorm alone just surfaces the next one, so this is a set, not a single fix.

1. **RMSNorm** — every transformer block (28x in Qwen3-0.6B).
   `sglang/srt/layers/layernorm.py:567` -> `sgl_kernel.rmsnorm`
   -> `sgl_kernel/elementwise.py:114 rmsnorm` -> flashinfer fallback ->
   `FlashInfer requires GPUs with sm75 or higher`.
   Native symbol: `torch.ops.sgl_kernel.rmsnorm`.

Operators after RMSNorm were not reached (forward aborted at block 1). A full
replacement pass should expect at minimum, in forward/serve order:
rmsnorm / fused_add_rmsnorm, RoPE, the triton attention path itself (verify it
JITs for sm70), the sampling/top-k path (`fast_topk`), KV-cache fill/copy
(`assign_*_cache_locs`), and the all-reduce only if TP>1 (not needed for the
single-card target). These are the candidates to enumerate by continuing the
diagnostic run with an sm70 RMSNorm in place.

## What works on sm70 today (same venv)

- **torchtitan Qwen3-0.6B fwd+bwd: PASS** (native `model_registry`, random init,
  meta -> to_empty(cuda) -> init_weights, batch 2x32):
  ```
  [fp32] logits=(2, 32, 151936) loss=10.6389 gradnorm=39.3386
  [fp16] logits=(2, 32, 151936) loss=11.0263 gradnorm=38.2500
  SM70_TITAN_TRAIN_STEP_PASS
  ```
  sdpa attention and torch ops run on sm70 in both fp32 and fp16.
- **CPU pytest subset: 87 passed** (CUDA_VISIBLE_DEVICES="").
- torch/torchaudio/torchvision/torchtitan/triton/meshy import clean; triton
  attention backend is selectable in SGLang but is blocked upstream by the
  gates above before it is exercised.

## Environment quirks (not sm70 kernels, but required to reproduce)

- sglang-kernel pins a cu13 torch; force-reinstall the cu126 torch trio AFTER
  installing it or torch silently becomes 2.13.0+cu130 (arch_list sm_75.., no
  sm_70) with CUDA-version import errors.
- httpx 0.28.1 cannot parse IPv6 CIDRs in the default `NO_PROXY`
  (`fe80::/10`,`fd00::/8`): `httpx.InvalidURL: Invalid port ':'`. Set
  `NO_PROXY=localhost,127.0.0.1,::1,<corp domains>`.
- Direct egress is ~3x the corp proxy for wheel downloads; see v100_install.md.

## Reproduce

```bash
# clean (gated) failure:
bash /data00/meshy/env/sm70_sglang_smoke.sh          # -> "SGLang only supports sm75 and above"
# patched diagnostics: backup then neutralise the gate in load_model_utils.py
# and replace _load_architecture_specific_ops() with a lazy dummy in
# sgl_kernel/__init__.py; rerun -> model loads, forward dies at flashinfer RMSNorm.
# Backups taken during diagnosis: *.orig-sm70diag (already restored).
```
