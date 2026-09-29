"""sm70 flash-decoding paged decode attention (no tensor cores).

Parallelism is (sequence, kv-head, KV chunk). A block owns one kv
head's GQA group and streams a chunk of K/V once, sharing that read
across the ``group_size`` q heads; 128 threads each hold one head_dim
lane and do fp32 dot/FMA work. A second kernel merges chunk partials by
log-sum-exp. KV addressing uses SGLang's flat decode metadata directly
(``kv_indptr`` + contiguous ``kv_indices``), so this is a drop-in
replacement shape for ``decode_attention_fwd``.

Target: raise effective decode KV bandwidth toward 400 GB/s on V100 by
maximizing occupancy and vectorized loads rather than HMMA (the T3e
HMMA kernel lost the long-context batch-64 cells — see
``sm70_tl_attention.md``).
"""

from __future__ import annotations

import math

import torch

from . import _tl
from ._flash_decode_kernel import (
    make_flash_decode_combine,
    make_flash_decode_partial,
)

V100_SM = 80
# Two resident chunks per (seq, kv-head) is the occupancy target; the
# block is light on shared memory (512 B), so occupancy is register/
# warp-scheduler bound — keep this modest.
_BLOCKS_PER_SM = 2

_cache: dict[tuple, object] = {}


def choose_splits(batch, num_kv_heads, ctx, num_sm=V100_SM,
                  blocks_per_sm=_BLOCKS_PER_SM, chunk_min=64):
    """Smallest split count filling num_sm*blocks_per_sm chunks."""
    base = batch * num_kv_heads
    want = num_sm * blocks_per_sm
    if base <= 0:
        return 1
    s = max(1, math.ceil(want / base))
    # Round chunk down toward chunk_min by capping splits at ctx/chunk.
    s = min(s, max(1, ctx // chunk_min))
    return max(1, s)


def flash_decode_attention(q, k_cache, v_cache, kv_indptr, kv_indices,
                           seq_lens, *, sm_scale=None, num_splits=None):
    """Flash-decoding paged decode.

    q [M,H,D] fp16; k/v cache [num_slots, KVH, D] fp16 (the 4-D c128
    pool reshaped to (-1, KVH, D)); kv_indptr [M+1] int32; kv_indices
    flat int32 token slots (length sum(seq_lens), padded internally);
    seq_lens [M] int32. Returns [M,H,D] fp16.
    """
    m, h, d = q.shape
    kvh = k_cache.shape[1]
    group = h // kvh
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    max_ctx = int(seq_lens.max().item())
    if num_splits is None:
        num_splits = choose_splits(m, kvh, max_ctx)
    # chunk_tokens is a compile-time constant per kernel; round the max
    # context up so each split owns a whole number of tokens.
    chunk = math.ceil(max_ctx / num_splits)
    key = (h, kvh, d, group, chunk)
    pair = _cache.get(key)
    if pair is None:
        pair = (
            make_flash_decode_partial(_tl, h, kvh, d, group, chunk),
            make_flash_decode_combine(_tl, h, d),
        )
        _cache[key] = pair
    partial_factory, combine_factory = pair

    device = q.device
    # Pad flat indices to m * num_splits * chunk slots so a chunk's
    # out-of-length gathers never read past the buffer (they are masked
    # to slot 0 in the kernel).
    total_pad = m * num_splits * chunk
    if kv_indices.shape[0] < total_pad:
        pad = torch.zeros(total_pad - kv_indices.shape[0],
                          dtype=kv_indices.dtype, device=device)
        idx = torch.cat([kv_indices, pad])
    else:
        idx = kv_indices
    scale_t = torch.tensor([sm_scale], dtype=torch.float32, device=device)
    partial_out = torch.empty(
        (num_splits, m, h, d), dtype=torch.float16, device=device)
    partial_lse = torch.empty(
        (num_splits, m, h), dtype=torch.float32, device=device)
    partial_factory(q.contiguous(),
                    k_cache.contiguous(), v_cache.contiguous(),
                    kv_indptr.to(torch.int32).contiguous(),
                    idx.to(torch.int32).contiguous(),
                    seq_lens.to(torch.int32).contiguous(), scale_t,
                    partial_out, partial_lse)
    out = torch.empty((m, h, d), dtype=torch.float16, device=device)
    combine_factory(partial_out, partial_lse, out)
    return out
