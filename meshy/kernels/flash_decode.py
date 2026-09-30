"""sm70 flash-decoding paged decode attention (hand-written CUDA).

Thin loader/launcher around ``csrc/flash_decode.cu``. Grid is
(sequence * kv_head, split); each 128-thread block runs eight
half-warps, each streaming its own key range with 16 B vector loads and
in-half-warp QK shuffles (no per-key shared-memory reduction). A second
kernel merges the KV splits by log-sum-exp.

ABI matches SGLang's triton ``decode_attention_fwd`` inputs: q
[M,H,D], flat k/v cache [num_slots, KVH, D], ``kv_indptr`` [M+1],
flat ``kv_indices`` and ``seq_lens`` [M].
"""

from __future__ import annotations

import math
import os

import torch

V100_SM = 80
_WARPS_PER_BLOCK = 4
_TARGET_WARPS_PER_SM = 32
_BLOCKS_PER_SM_TARGET = _TARGET_WARPS_PER_SM // _WARPS_PER_BLOCK  # 8

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


def choose_splits(batch, num_kv_heads, ctx,
                 num_sm=V100_SM, blocks_target=_BLOCKS_PER_SM_TARGET,
                 chunk_min=64):
    """Pick splits for >= blocks_target*num_sM resident blocks.

    bs64 x 8 kv heads already yields 512 blocks (~6.4/SM, 25 warps);
    ctx >=1k adds splits 2-4 to push in-flight loads toward 32 warps/SM.
    """
    base = batch * num_kv_heads
    want = num_sm * blocks_target
    s = max(1, math.ceil(want / base))
    s = min(s, max(1, ctx // chunk_min))
    return max(1, s)


def flash_decode_attention(q, k_cache, v_cache, kv_indptr, kv_indices,
                           seq_lens, *, sm_scale=None, num_splits=None):
    """Run the flash-decoding kernel; returns [M,H,D] fp16."""
    m, h, d = q.shape
    kvh = k_cache.shape[1]
    assert d == 128 and h // kvh == 2
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    max_ctx = int(seq_lens.max().item())
    if num_splits is None:
        num_splits = choose_splits(m, kvh, max_ctx)
    # chunk must cover ceil(max_ctx/splits) tokens and be a multiple of
    # 8 half-warps.
    chunk = max(8, math.ceil(max_ctx / num_splits / 8) * 8)
    total_slots = m * num_splits * chunk
    if kv_indices.shape[0] < total_slots:
        idx = torch.cat([kv_indices,
                         kv_indices.new_zeros(total_slots - kv_indices.shape[0])])
    else:
        idx = kv_indices
    partial_out = torch.empty(
        (num_splits, m * kvh, 2, d), dtype=torch.float16, device=q.device)
    partial_lse = torch.empty(
        (num_splits, m * kvh, 2), dtype=torch.float32, device=q.device)
    out = torch.empty((m, h, d), dtype=torch.float16, device=q.device)
    _lib().launch_flash_decode(
        q.contiguous(), k_cache.contiguous(), v_cache.contiguous(),
        kv_indptr.to(torch.int32).contiguous(),
        idx.to(torch.int32).contiguous(),
        seq_lens.to(torch.int32).contiguous(),
        int(num_splits), int(chunk), float(sm_scale),
        partial_out, partial_lse, out)
    return out
