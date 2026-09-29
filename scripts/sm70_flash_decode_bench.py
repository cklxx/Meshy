"""sm70 flash-decoding attention: GPU correctness + latency.

Bench for T3f. Compares
``meshy.kernels.flash_decode.flash_decode_attention`` against SGLang
triton ``decode_attention_fwd`` and fp32 SDPA, batch 32/64/128 x ctx
512/1k/2k/4k, fp16, Qwen3-0.6B shapes (16 q / 8 kv heads, dim 128),
c128 page size 16. Reports us/step, effective single-layer HBM GB/s
(batch*ctx*8*128*2(K/V)*2B / us), and max abs diff vs the fp32 ref.

Run on the V100 with the GPU held, using the CUDA 12.4 nvcc:

    PATH=/usr/local/cuda-12.4/bin:$PATH \
    python scripts/sm70_flash_decode_bench.py --triton
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from meshy.kernels.flash_decode import flash_decode_attention, choose_splits

H, KVH, D = 16, 8, 128
PAGE = 16


def make_inputs(batch, ctx, device="cuda"):
    torch.manual_seed(0)
    pages_per_seq = math.ceil(ctx / PAGE)
    num_pages = 2 + batch * pages_per_seq
    k_pool = torch.randn(num_pages, PAGE, KVH, D, device=device,
                         dtype=torch.float16) * 0.1
    v_pool = torch.randn_like(k_pool) * 0.1
    q = torch.randn(batch, H, D, device=device, dtype=torch.float16) * 0.1
    block_table = (
        torch.arange(2, 2 + batch * pages_per_seq, device=device,
                     dtype=torch.int32).reshape(batch, pages_per_seq))
    seq_lens = torch.full((batch,), ctx, device=device, dtype=torch.int32)
    lengths = seq_lens.long()
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(lengths, 0)
    tok = torch.arange(ctx, device=device)
    per_row = block_table[:, tok // PAGE] * PAGE + tok % PAGE
    kv_indices = per_row.reshape(-1).to(torch.int32)
    return q, k_pool, v_pool, block_table, seq_lens, kv_indptr, kv_indices


def ref_full(q, k_pool, v_pool, kv_indices, seq_lens):
    m = q.shape[0]
    device = q.device
    n = int(seq_lens.max().item())
    kf = k_pool.reshape(-1, KVH, D)
    vf = v_pool.reshape(-1, KVH, D)
    slots = kv_indices.reshape(m, n)
    g = H // KVH
    h_map = torch.arange(H, device=device) // g
    k = kf[slots].index_select(2, h_map)
    v = vf[slots].index_select(2, h_map)
    qh = q.unsqueeze(2).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).float()
    o = torch.nn.functional.scaled_dot_product_attention(
        qh, kh, vh, scale=1.0 / math.sqrt(D))
    return o[:, :, 0, :].to(torch.float16)


def tl_call(q, k_pool, v_pool, kv_indptr, kv_indices, seq_lens):
    return flash_decode_attention(
        q, k_pool.reshape(-1, KVH, D), v_pool.reshape(-1, KVH, D),
        kv_indptr, kv_indices, seq_lens)


def triton_decode(q, k_pool, v_pool, kv_indptr, kv_indices, seq_lens):
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
    from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton

    batch = q.shape[0]
    device = q.device
    max_split = 16
    num_sm = torch.cuda.get_device_properties(device).multi_processor_count
    num_kv_splits = torch.empty(batch, dtype=torch.int32, device=device)
    get_num_kv_splits_triton[(1,)](
        num_kv_splits, seq_lens, batch, 1, H, KVH, max_split, num_sm,
        MAX_NUM_SEQ=max(256, 1 << (batch - 1).bit_length()))
    o = torch.empty_like(q)
    attn_logits = torch.empty(
        (batch, H, max_split, D), dtype=torch.float16, device=device)
    attn_lse = torch.empty(
        (batch, H, max_split), dtype=torch.float32, device=device)
    decode_attention_fwd(
        q, k_pool, v_pool, o, kv_indptr, kv_indices,
        attn_logits, attn_lse, num_kv_splits, max_split,
        1.0 / math.sqrt(D), 1.0, 1.0, 0.0, page_size=PAGE,
    )
    return o


def bench(fn, *args, warmup=5, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data00/meshy/kern/flash_decode_bench.json")
    ap.add_argument("--batches", default="32,64,128")
    ap.add_argument("--ctxs", default="512,1024,2048,4096")
    ap.add_argument("--triton", action="store_true")
    args = ap.parse_args()

    combos = [(int(b), int(c))
              for b in args.batches.split(",")
              for c in args.ctxs.split(",")]
    results = []
    for batch, ctx in combos:
        (q, k_pool, v_pool, block_table, seq_lens,
         kv_indptr, kv_indices) = make_inputs(batch, ctx)
        # KV traffic only (K and V, fp16). Q/O/index traffic is tiny by
        # comparison and NOT counted; tl_gbs is therefore the effective
        # KV bandwidth = theoretical bytes / measured time.
        kv_bytes = batch * ctx * KVH * D * 2 * 2
        qo_bytes = batch * H * D * 2 * 2  # Q read + O write, for reference

        def bw(ms):
            return round(kv_bytes / (ms / 1e3) / 1e9, 0)

        o_tl = tl_call(q, k_pool, v_pool, kv_indptr, kv_indices, seq_lens)
        o_ref = ref_full(q, k_pool, v_pool, kv_indices, seq_lens)
        err = (o_tl.float() - o_ref.float()).abs().max().item()
        tl_ms = bench(tl_call, q, k_pool, v_pool, kv_indptr, kv_indices,
                      seq_lens)
        splits = choose_splits(batch, KVH, ctx)
        row = {"batch": batch, "ctx": ctx, "splits": splits,
               "kv_mb": round(kv_bytes / 1e6, 1),
               "qo_mb": round(qo_bytes / 1e6, 2),
               "max_abs_err": err,
               "tl_us": round(tl_ms * 1000, 1), "tl_gbs": bw(tl_ms)}
        if args.triton:
            try:
                o_tr = triton_decode(
                    q, k_pool, v_pool, kv_indptr, kv_indices, seq_lens)
                tr_ms = bench(triton_decode, q, k_pool, v_pool, kv_indptr,
                              kv_indices, seq_lens)
                row["triton_us"] = round(tr_ms * 1000, 1)
                row["triton_gbs"] = bw(tr_ms)
                row["triton_err"] = (
                    o_tr.float() - o_ref.float()).abs().max().item()
                row["speedup"] = round(tr_ms / tl_ms, 2)
            except Exception as e:  # noqa: BLE001
                row["triton_error"] = repr(e)
        results.append(row)
        print(json.dumps(row), flush=True)
        del q, k_pool, v_pool, block_table, seq_lens, kv_indptr, kv_indices
        del o_tl, o_ref
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("FLASH_DECODE_BENCH_DONE ->", args.out)


if __name__ == "__main__":
    main()
