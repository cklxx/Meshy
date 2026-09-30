"""sm70 flash-decoding paged decode attention (hand-written CUDA).

Kernel: ``csrc/flash_decode.cu`` (half-warp-per-key, 16 B vector loads,
shuffle QK, single block-end merge, LSE split combine). The row length
is read on-device from ``kv_indptr`` differences, so the launch path has
**no CPU sync**: host parameters are only the grid shape (batch bucket,
split count, chunk tokens), which a captured CUDA graph fixes once.

Two entry points:

* :func:`flash_decode_attention` — eager convenience wrapper (picks the
  plan from the actual max length; used by the microbench).
* :class:`FlashDecodePlan` + :func:`run_with_plan` — fixed grid shape
  and preallocated workspaces for the CUDA-graph integration. The plan
  is chosen at capture time from the batch bucket and a max-context
  budget; replay never changes a launch dimension.

ABI matches SGLang triton ``decode_attention_fwd`` inputs.
"""

from __future__ import annotations

import math
import os

import torch

V100_SM = 80
_WARPS_PER_BLOCK = 4
_TARGET_WARPS_PER_SM = 32
_BLOCKS_PER_SM_TARGET = _TARGET_WARPS_PER_SM // _WARPS_PER_BLOCK  # 8
_HALF_WARPS = 8
_HEAD_DIM = 128

_loaded = None


def _lib():
    global _loaded
    if _loaded is not None:
        return _loaded
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "csrc", "flash_decode.cu")
    # Caller puts /usr/local/cuda-12.4/bin on PATH (sm70 + c++17).
    _loaded = load(
        name="meshy_flash_decode_sm70",
        sources=[src],
        extra_cuda_cflags=[
            "-O3", "-gencode=arch=compute_70,code=sm_70",
            "-std=c++17", "--use_fast_math",
        ],
        verbose=False,
    )
    return _loaded


def choose_splits(batch, num_kv_heads, max_ctx,
                 num_sm=V100_SM, blocks_target=_BLOCKS_PER_SM_TARGET,
                 chunk_min=64):
    """Split count for ~32 resident warps/SM, capped at ctx/chunk_min."""
    base = max(1, batch) * num_kv_heads
    want = num_sm * blocks_target
    s = max(1, math.ceil(want / base))
    s = min(s, max(1, max_ctx // chunk_min))
    return max(1, s)


def choose_chunk(max_ctx, num_splits):
    """Tokens per split, rounded up to a whole number of half-warps."""
    c = max(_HALF_WARPS, math.ceil(max_ctx / num_splits))
    return ((c + _HALF_WARPS - 1) // _HALF_WARPS) * _HALF_WARPS


class FlashDecodePlan:
    """Fixed launch geometry + static workspaces for one bucket."""

    def __init__(self, num_heads, num_kv_heads, head_dim, grid_batch,
                 num_splits, chunk_tokens, device, dtype=torch.float16):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.grid_batch = grid_batch
        self.num_splits = num_splits
        self.chunk_tokens = chunk_tokens
        kv_blocks = grid_batch * num_kv_heads
        # [splits, seq*kv_head, group=2, D] partial; matching LSE.
        self.partial_out = torch.empty(
            (num_splits, kv_blocks, num_heads // num_kv_heads, head_dim),
            dtype=dtype, device=device)
        self.partial_lse = torch.empty(
            (num_splits, kv_blocks, num_heads // num_kv_heads),
            dtype=torch.float32, device=device)

    @property
    def max_slots(self):
        return self.grid_batch * self.num_splits * self.chunk_tokens


def make_plan(num_heads, num_kv_heads, head_dim, batch_bucket,
              max_ctx_budget, device):
    """Capture-time plan: split/chunk cover max_ctx_budget."""
    splits = choose_splits(batch_bucket, num_kv_heads, max_ctx_budget)
    chunk = choose_chunk(max_ctx_budget, splits)
    return FlashDecodePlan(
        num_heads, num_kv_heads, head_dim, batch_bucket, splits, chunk,
        device)


def run_with_plan(q, k_flat, v_flat, kv_indptr, kv_indices, plan,
                  sm_scale, out=None):
    """Zero-CPU-sync launch under a fixed plan.

    q must already be padded to ``plan.grid_batch`` rows; kv_indices
    must be padded (or a static buffer of length >= plan.max_slots).
    Padding rows have indptr[n]==indptr[n+1] (zero length), so their
    blocks compute nothing and the output slice is ignored by the
    caller.
    """
    m = q.shape[0]
    if out is None:
        out = torch.empty(
            (m, plan.num_heads, plan.head_dim), dtype=q.dtype,
            device=q.device)
    # No padding of kv_indices: padding rows have indptr[n+1]==indptr[n]
    # (seq_len 0), so every one of their tokens is invalid and the
    # kernel's predicated slot load never touches the index buffer.
    _lib().launch_flash_decode(
        q.contiguous(), k_flat.contiguous(), v_flat.contiguous(),
        kv_indptr.to(torch.int32).contiguous(),
        kv_indices.to(torch.int32).contiguous(),
        plan.grid_batch, plan.num_splits, plan.chunk_tokens,
        float(sm_scale), plan.partial_out, plan.partial_lse, out)
    return out


def flash_decode_attention(q, k_cache, v_cache, kv_indptr, kv_indices,
                           seq_lens=None, *, sm_scale=None,
                           num_splits=None):
    """Eager wrapper: derives lengths from kv_indptr when seq_lens is
    None (graph-safe), else from the provided tensor."""
    m, h, d = q.shape
    kvh = k_cache.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    if seq_lens is not None:
        max_ctx = int(seq_lens.max().item())
    else:
        lens = kv_indptr[1:] - kv_indptr[:-1]
        max_ctx = int(lens.max().item())
    if num_splits is None:
        num_splits = choose_splits(m, kvh, max_ctx)
    chunk = choose_chunk(max_ctx, num_splits)
    plan = FlashDecodePlan(h, kvh, d, m, num_splits, chunk, q.device)
    return run_with_plan(
        q, k_cache, v_cache, kv_indptr, kv_indices, plan, sm_scale)
