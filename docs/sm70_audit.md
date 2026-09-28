# sm70 (Tesla V100) gap audit — Meshy GRPO on Qwen3

Date: 2026-09-28. Branch: `v100/kern`. Target: single V100-SXM2-32GB (compute
7.0, no bf16 tensor cores), Qwen3-0.6B/1.7B dense, TP=1,
SGLang rollout + torchtitan GRPO colocated.

Code audited: Meshy (this repo), torchtitan
`0.1.0.dev20260501+cu126` (`third_party/`, unpacked for the audit), SGLang
main `c048ebd10` (local clone `~/code/sglang`).

Design basis: *Defeating the Training-Inference Mismatch via FP16*
(arXiv:2510.26788) — fp16 for rollout logprobs, train forward and softmax,
uniformly on both engines, with dynamic loss scaling; no fp32 master-weight
requirement beyond standard AMP safety. Meshy's rollout logprobs/trainer
logprob/softmax path already upcasts logits to fp32 for cross-entropy
(`meshy/backend/titan/trainer.py:398-411`), so the PPO ratio is fp32 end to
end; only the network matmuls run fp16.

## What `training.dtype` actually controls in torchtitan

`TrainingConfig.dtype` sets the **storage** dtype, not compute:
`ForgeEngine.__init__` meta-inits the model under
`set_default_dtype(TORCH_DTYPE_MAP[dtype])`
(`torchtitan/experiments/forge/engine.py:146`); after `to_empty`, params,
gradients and Adam moments live in that dtype. Compute dtype is a separate
field, `TrainingConfig.mixed_precision_param` (default `bfloat16`), handed to
FSDP2 `MixedPrecisionPolicy(param_dtype=…)`
(`torchtitan/models/qwen3/parallelize.py:102-110`). FSDP all-gathers the
fp32 shard into the param dtype for every forward/backward and casts grads
back. This applies **even at world size 1**: torchtitan deliberately keeps
the `fsdp` mesh at degree 1 "so `fully_shard()` can apply
MixedPrecisionPolicy even at degree 1"
(`torchtitan/distributed/parallel_dims.py:73-76`). So `dtype="float32"`,
`mixed_precision_param="float16"` gives fp32 master weights/grads/Adam +
fp16 matmuls on a single card. There is no DDP/autocast path that would
bypass it.

Uniform-fp16 storage (`dtype="float16"`) is unsound at this recipe's
lr. At |w| ≈ 0.02 the fp16 ULP is ≈ 1.5e-5; an AdamW step at lr=1e-6
(weight_decay=0.1 adds ≈ 2e-9) is below half a ULP and rounds away. Adam
moments in fp16 additionally lose small gradients. The bf16 default is
worse on paper (ULP ≈ 1.2e-4 at the same |w|) and cannot run on sm70 at
all. On-device confirmation: `scripts/sm70_optimizer_update.py` (see
"Pending verification" below).

## Findings — Meshy + torchtitan

### P0 — training could not run correctly on sm70 before this change

1. `meshy/config.py:36` — `TrainerConfig.dtype` defaulted to `"bfloat16"`
   with no fp16/fp32 validation; every recipe hard-coded `"bfloat16"`
   (`recipe/grpo_gsm8k_qwen3_8b.py:62` and 13 others). bf16 on sm70 has no
   tensor-core path and is slow/partially unsupported by PyTorch.
   **Fix**: dtype is now `Literal["bfloat16","float16","float32"]`; new
   `mixed_precision_param` knob; sm70 recipe uses fp32 storage + fp16
   compute.
2. torchtitan `mixed_precision_param` was not plumbed: Meshy's
   `build_forge_config` constructed `TrainingConfig` without it
   (`meshy/backend/titan/config.py:100-107`), so it defaulted to
   `bfloat16` — fp16 storage + bf16 all-gather, broken on V100. **Fix**:
   plumbed through, auto-derived (`dtype=float16` → fp16 compute; else
   bf16). Same for the critic (`meshy/engine/critic.py:129-142`).
3. No loss scaling anywhere: `loss.backward()` / `optimizer.step()` in
   `meshy/backend/titan/trainer.py:929-952` (and the OPD override
   `meshy/backend/titan/opd.py`, critic
   `meshy/backend/titan/critic/engine.py:511-522`). fp16 backward without
   scaling overflows at the first attention/CE grads. **Fix**: dynamic
   `torch.cuda.amp.GradScaler`, enabled exactly when
   `mixed_precision_param=="float16"`, wired in all three step loops with
   `unscale_` before `clip_grad_norm_`; `train/loss_scale` metric emitted
   per step.
4. `meshy/backend/titan/critic/engine.py:114-116` — meta-init recognised
   only bfloat16/float32 (`torch.bfloat16 if … else torch.float32`);
   `dtype=float16` silently built an fp32 model. **Fix**: full dtype map.

### P1 — works only with the right config / layout

5. `torchtitan/models/common/attention.py:128-131` — `VarlenAttention`
   hard-casts q/k/v to `torch.bfloat16` ("varlen attention currently only
   supports bf16/fp16 inputs" comment notwithstanding). The packed batch
   layout (`TrainerParamsConfig.batch_layout="packed"`,
   `attn_backend="varlen"`) therefore cannot run on sm70. **Fix**: none in
   Meshy; use the default padded layout + SDPA (the v100 recipe does).
6. SDPA backend list
   (`torchtitan/models/common/attention.py:271-275`) is
   `[CUDNN_ATTENTION, FLASH_ATTENTION, MATH]`. On V100 with fp16, PyTorch
   dispatches to the fused FA2 SDPA kernel (fp16 sm70 supported) or MATH;
   cudnn attention is simply skipped if unavailable. No change needed.
7. `compile_model` / inductor: recipes already set `compile_model=False`;
   keep it off on sm70 (no cudagraphs, sm-specific codegen risk; see
   `meshy/backend/titan/trainer.py:290-298`).
8. Optimizer: torchtitan defaults to fused AdamW
   (`torchtitan/components/optimizer.py:112`). Fused AdamW keeps fp32
   moments only when params are fp32 — another reason fp32 master storage
   is the right default. No scaler support needed in the container itself;
   Meshy drives `scaler.step` per inner optimizer.
9. RMSNorm/RoPE precision: RoPE cos/sin (Qwen3) computes in fp32 and casts
   back (`torchtitan/models/common/rope.py:391-397`); critic value head
   forces fp32 (`meshy/backend/titan/critic/model.py:172-181`). Both are
   dtype-agnostic and safe under fp16 compute.
10. HF weight sync: trainer saves the master (fp32) state dict
    (`meshy/backend/titan/trainer.py:1017-1051`); SGLang
    `update_weights_from_disk` loads under its own model dtype (fp16 with
    `--dtype half`), so the cast happens at load. Rollout policy = fp16
    compute policy, matching the FP16 paper. No dtype field needed in the
    checkpoint.

## Findings — SGLang (main c048ebd10)

No sm70 capability gating exists anywhere; arch gates start at sm80/sm90.

### P0

1. Default attention backend selects flashinfer whenever the package
   imports — `python/sglang/srt/server_args.py:2725-2729` +
   `python/sglang/srt/utils/common.py:326-333` (importlib probe, no
   capability check). FlashInfer ships sm75+ cubins only
   (`docs/get_started/install.md:229`). Fix: `--attention-backend triton`.
2. sgl-kernel prebuilt wheel compiles only sm80/sm89/sm90/sm100 SASS with
   no PTX fallback (`sgl-kernel/CMakeLists.txt:130,197-213`); on older
   cards `sgl_kernel/load_utils.py:54-62` loads the sm100 variant, so
   every launch fails with "no kernel image". Dense Qwen3's hard
   dependency is RMSNorm: `python/sglang/srt/layers/layernorm.py:83-88,
   305-307` unconditionally call `sgl_kernel.rmsnorm`/
   `fused_add_rmsnorm`. `sgl_kernel/elementwise.py:105-122` first tries
   `flashinfer.norm` (a triton kernel JIT-able for sm70) when flashinfer
   imports — likely to work, unverified on device. Fallbacks if it
   doesn't: source-build sgl-kernel with `-gencode=arch=compute_70,
   code=sm_70`, or patch `forward_cuda` to `sglang.jit_kernel.norm` /
   `forward_native` for major < 8.
3. fp16 dtype is opt-in: `dtype="auto"` adopts Qwen3's bf16 HF config
   (`python/sglang/srt/configs/model_config.py:1437-1455`,
   KV cache auto-follows at `model_runner.py:2189-2207`). Fix:
   `--dtype half` (wired as `"dtype": "half"` in the recipe's
   server_args).

### P1

4. Sampler defaults to flashinfer + sgl-kernel renorm
   (`python/sglang/srt/layers/sampler.py:27-35`,
   `server_args.py:2673-2677`); greedy already routes to torch
   (`sampler.py:213-218`). Fix: `--sampling-backend pytorch`.
5. fa3 asserts sm80+/sm90
   (`layers/attention/attention_registry.py:181-204`); sglang srt has no
   FA2 backend, so triton is the only sm70 option (torch_native is the
   slow fallback and forces graph disable).
6. CUDA graph is on by default (`server_args.py:717`) with raise, no
   fallback on capture failure
   (`model_executor/cuda_graph_runner.py:700-707`). sm70 capture failure
   also poisons the caching allocator (tileRL experience,
   PR sglang#545). Fix: `--disable-cuda-graph` (already in the recipe).
7. fp8 hard-gated at capability 80
   (`layers/quantization/fp8.py:207-210`); don't use fp8 KV cache or
   checkpoints; triton backend has no fp8 path anyway.

### P2 — verified safe / no action

- triton attention has old-arch branches
  (`triton_ops/extend_attention.py:106-108`,
  `prefill_attention.py:180-183` block 64 on sm≤80); no TMA/cp.async
  assumptions; PDL auto-off below sm90 (`triton_backend.py:163-166`).
  Remaining risk is bf16 `tl.dot` — moot with `--dtype half`.
- RoPE/SiLU go through local nvcc JIT
  (`sglang/jit_kernel/{rope,activation}.py`, arch from
  `torch.cuda.get_device_capability()`), or have torch native fallbacks.
- `enable_memory_saver`, pause/resume, `update_weights_from_disk` are
  dtype/arch-neutral (`model_runner.py:1598-1657`).
- Do **not** set `SGLANG_IS_FLASHINFER_AVAILABLE=false`: that would also
  kill the `flashinfer.norm` triton route that currently rescues RMSNorm
  (finding P0-2).

## Changes on this branch

- `meshy/config.py`: `dtype` literal + validation; new
  `mixed_precision_param` field.
- `meshy/backend/titan/config.py`, `meshy/engine/critic.py`: plumb
  mixed-precision dtype (auto: fp16 storage → fp16 compute).
- `meshy/backend/titan/trainer.py`, `…/opd.py`, `…/critic/engine.py`:
  GradScaler in actor/OPD/critic step loops (scale backward, unscale
  before clip, `scaler.step/update`); `train/loss_scale` metric; critic
  meta-init dtype map.
- `recipe/grpo_gsm8k_v100.py` (cherry-picked from `v100/rl` @ c62dcd4):
  default changed to fp32 master + fp16 compute + scaler; server args add
  `attention_backend=triton`, `sampling_backend=pytorch`; docstring memory
  accounting corrected (trainer residency ~11 GiB peak GPU, ~6.7 GiB
  offloaded RAM).
- `scripts/sm70_optimizer_update.py`: one-step AdamW movement test,
  fp16-uniform vs fp32-master/fp16-compute arms.

## Pending on-device verification (blocked on env's T1 venv)

Run after `/data00/meshy/venv` has torch (sm70 arch list) + torchtitan:

```
awb hold v100gpu kern "optimizer ULP measurement"
ssh v100 'cd /data00/meshy/kern/Meshy && PATH=/usr/local/cuda-12.4/bin:$PATH \
  /data00/meshy/venv/bin/python scripts/sm70_optimizer_update.py \
  --flavor 0.6B --out /data00/meshy/kern/ulp.json'
awb release v100gpu kern
```

Expected: fp16 arm shows a small `changed_fraction` (updates rounding
away, quantile of |Δ|/ULP near 0), fp32 arm ~1.0 changed. Then the SGLang
smoke (triton/pytorch/fp16, no graph) — if RMSNorm raises "no kernel
image", that is the one candidate for a TileLang kernel or the sgl-kernel
sm70 rebuild; decide from env's actual failure list rather than
speculatively.
