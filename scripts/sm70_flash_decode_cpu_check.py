"""CPU logic mirror for the sm70 flash-decoding kernel.

No GPU/tilelang: mirrors chunk split (uniform ceil(seqlen/splits)),
flat kv_indices addressing, per-(kv-head) two-q-head online softmax,
and LSE combine; checks against dense fp32 SDPA.

    python scripts/sm70_flash_decode_cpu_check.py
"""

from __future__ import annotations

import math

import torch

H, KVH, D = 16, 8, 128
G = H // KVH


def choose_splits(batch, ctx, num_sm=80, blocks_per_sm=2, chunk_min=64):
    want = num_sm * blocks_per_sm
    s = max(1, math.ceil(want / max(1, batch * KVH)))
    s = min(s, max(1, ctx // chunk_min))
    return max(1, s)


def mirror(q, k_flat, v_flat, indptr, indices, seq_lens, num_splits,
           chunk):
    m = q.shape[0]
    scale = 1.0 / math.sqrt(D)
    po = torch.zeros(num_splits, m, H, D)
    lse = torch.full((num_splits, m, H), -3.0e38)
    for b in range(m):
        n = int(seq_lens[b])
        base = int(indptr[b])
        for h in range(H):
            kvh = h // G
            for sp in range(num_splits):
                acc = torch.zeros(D)
                mm, ll = -3.0e38, 0.0
                lo, hi = sp * chunk, min((sp + 1) * chunk, n)
                for t in range(lo, hi):
                    slot = int(indices[base + t])
                    s = float((q[b, h].float()
                               * k_flat[slot, kvh].float()).sum() * scale)
                    nm = max(mm, s)
                    corr = math.exp(mm - nm) if mm > -3.0e38 else 0.0
                    p = math.exp(s - nm)
                    acc = acc * corr + p * v_flat[slot, kvh].float()
                    ll = ll * corr + p
                    mm = nm
                if ll > 0.0:
                    po[sp, b, h] = acc / ll
                    lse[sp, b, h] = mm + math.log(ll)
    out = torch.zeros(m, H, D)
    for b in range(m):
        for h in range(H):
            mstar = max(lse[:, b, h].max().item(), -3.0e38)
            ws = [math.exp(lse[sp, b, h] - mstar)
                  if lse[sp, b, h] > -3.0e38 else 0.0
                  for sp in range(num_splits)]
            denom = sum(ws)
            out[b, h] = sum(po[sp, b, h] * ws[sp]
                            for sp in range(num_splits)) / denom
    return out


def dense_ref(q, k_flat, v_flat, indptr, indices, seq_lens):
    m = q.shape[0]
    scale = 1.0 / math.sqrt(D)
    outs = []
    for b in range(m):
        n = int(seq_lens[b])
        base = int(indptr[b])
        slots = indices[base:base + n].long()
        k = k_flat[slots].unsqueeze(2).expand(n, KVH, G, D).reshape(n, H, D)
        v = v_flat[slots].unsqueeze(2).expand(n, KVH, G, D).reshape(n, H, D)
        qh = q[b].unsqueeze(1).unsqueeze(0).float()
        kh = k.permute(1, 0, 2).unsqueeze(0).float()
        vh = v.permute(1, 0, 2).unsqueeze(0).float()
        o = torch.nn.functional.scaled_dot_product_attention(
            qh, kh, vh, scale=scale)[0, :, 0, :]
        outs.append(o)
    return torch.stack(outs)


def build(seqlens, num_slots_extra=8, seed=0):
    torch.manual_seed(seed)
    m = len(seqlens)
    total = sum(seqlens)
    k_flat = torch.randn(total + num_slots_extra, KVH, D) * 0.1
    v_flat = torch.randn_like(k_flat) * 0.1
    q = torch.randn(m, H, D) * 0.1
    indptr = torch.zeros(m + 1, dtype=torch.int32)
    indptr[1:] = torch.tensor(seqlens).cumsum(0)
    # Distinct pseudo-random physical slots per row (valid range).
    slots = (torch.arange(total) * 7 + 3) % k_flat.shape[0]
    indices = slots.to(torch.int32)
    return q, k_flat, v_flat, indptr, indices, torch.tensor(
        seqlens, dtype=torch.int32)


def case(seqlens, chunk, tag, splits=None):
    q, kf, vf, ip, ix, sl = build(seqlens, seed=hash(tag) % 1000)
    nsp = splits or max(1, math.ceil(max(seqlens) / chunk))
    o = mirror(q, kf, vf, ip, ix, sl, nsp, chunk)
    ref = dense_ref(q, kf, vf, ip, ix, sl)
    err = (o - ref).abs().max().item()
    print(f"{tag}: lens={seqlens} splits={nsp} chunk={chunk} err={err:.2e}")
    assert err < 2e-3, (tag, err)


def main():
    # Chunk-unaligned, empty trailing chunks, rows shorter than chunk,
    # multi-row variable lengths.
    case([512], 128, "bs1-512-c128")
    case([500], 128, "bs1-misaligned")
    case([1024], 256, "bs1-1k-c256")
    case([33], 64, "bs1-short-empty-chunks", splits=8)
    case([512, 2048, 33], 256, "bs3-varlen")
    case([4096, 1], 256, "bs2-extreme")
    case([129, 130, 131, 128], 64, "bs4-odd")
    case([1700] * 64, 256, "bs64-ctx1700")
    print("ALL_FLASH_DECODE_CPU_CHECKS_PASS")


if __name__ == "__main__":
    main()
