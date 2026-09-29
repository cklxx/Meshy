# TileLang prim_func builder for the sm70 flash-decoding kernel.
#
# NO ``from __future__ import annotations`` (tilelang evaluates the
# prim_func annotations with this module's globals).
#
# Parallelism is (sequence, kv-head, KV chunk). One 128-thread block
# owns one kv-head's GQA group (group_size q heads share a single read
# of each K/V vector); thread d holds head_dim lane d and keeps, per q
# head, a Q register and an fp32 output accumulator. No tensor cores:
# QK is a 128-lane fp32 dot reduced across the block, PV a per-lane
# FMA. Every thread loads its own lane of K/V directly — lanes in a
# warp are contiguous in d, so the reads coalesce into 128 B
# transactions. Online softmax in fp32; a second kernel combines chunk
# partials by log-sum-exp.

import torch  # noqa: F401


def make_flash_decode_partial(_tl, H, KVH, D, group_size, chunk_tokens):
    tilelang, T = _tl()
    assert H == KVH * group_size

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def factory():
        num_split = T.dynamic("s")
        m = T.dynamic("m")
        num_slots = T.dynamic("p")

        @T.prim_func
        def partial(
            Q: T.Tensor([m, H, D], "float16"),
            KFlat: T.Tensor([num_slots, KVH, D], "float16"),
            VFlat: T.Tensor([num_slots, KVH, D], "float16"),
            KvIndptr: T.Tensor([m + 1], T.int32),
            KvIndices: T.Tensor([T.dynamic("ni")], T.int32),
            SeqLens: T.Tensor([m], T.int32),
            LogitScale: T.Tensor([1], "float32"),
            PartialOut: T.Tensor([num_split, m, H, D], "float16"),
            PartialLse: T.Tensor([num_split, m, H], "float32"),
        ):
            with T.Kernel(m, KVH, num_split, threads=D) as (bx, by, bz):
                q0 = T.alloc_fragment([D], "float32")
                q1 = T.alloc_fragment([D], "float32")
                prod0 = T.alloc_fragment([D], "float32")
                prod1 = T.alloc_fragment([D], "float32")
                dot0 = T.alloc_fragment([1], "float32")
                dot1 = T.alloc_fragment([1], "float32")
                acc0 = T.alloc_fragment([D], "float32")
                acc1 = T.alloc_fragment([D], "float32")
                slot = T.alloc_fragment([1], T.int32)
                seqlen = T.alloc_fragment([1], "float32")
                st = T.alloc_fragment([2], "float32")
                mm = T.alloc_fragment([2], "float32")
                ll = T.alloc_fragment([2], "float32")
                corr = T.alloc_fragment([2], "float32")
                pv = T.alloc_fragment([2], "float32")

                for d in T.Parallel(D):
                    q0[d] = T.cast(Q[bx, by * group_size, d], "float32")
                    q1[d] = T.cast(
                        Q[bx, by * group_size + 1, d], "float32")
                    acc0[d] = 0.0
                    acc1[d] = 0.0
                mm[0] = -3.0e38
                mm[1] = -3.0e38
                ll[0] = 0.0
                ll[1] = 0.0
                seqlen[0] = T.cast(SeqLens[bx], "float32")
                base = KvIndptr[bx]

                for t0 in T.serial(chunk_tokens):
                    t = bz * chunk_tokens + t0
                    valid = T.cast(t, "float32") < seqlen[0]
                    slot[0] = T.if_then_else(valid,
                                             KvIndices[base + t], 0)
                    for d in T.Parallel(D):
                        kv = T.cast(KFlat[slot[0], by, d], "float32")
                        prod0[d] = q0[d] * kv
                        prod1[d] = q1[d] * kv
                    T.reduce_sum(prod0, dot0)
                    T.reduce_sum(prod1, dot1)
                    st[0] = T.if_then_else(
                        valid, dot0[0] * LogitScale[0], -3.0e38)
                    st[1] = T.if_then_else(
                        valid, dot1[0] * LogitScale[0], -3.0e38)
                    for g in T.Parallel(2):
                        corr[g] = T.exp(mm[g] - T.max(mm[g], st[g]))
                        pv[g] = T.if_then_else(
                            valid,
                            T.exp(st[g] - T.max(mm[g], st[g])), 0.0)
                        mm[g] = T.max(mm[g], st[g])
                        ll[g] = ll[g] * corr[g] + pv[g]
                    for d in T.Parallel(D):
                        vv = T.cast(VFlat[slot[0], by, d], "float32")
                        acc0[d] = acc0[d] * corr[0] + pv[0] * vv
                        acc1[d] = acc1[d] * corr[1] + pv[1] * vv

                for d in T.Parallel(D):
                    PartialOut[bz, bx, by * group_size, d] = T.cast(
                        acc0[d] / T.max(ll[0], 1e-30), "float16")
                    PartialOut[bz, bx, by * group_size + 1, d] = T.cast(
                        acc1[d] / T.max(ll[1], 1e-30), "float16")
                PartialLse[bz, bx, by * group_size] = T.if_then_else(
                    ll[0] > 0.0, mm[0] + T.log(ll[0]), -3.0e38)
                PartialLse[bz, bx, by * group_size + 1] = T.if_then_else(
                    ll[1] > 0.0, mm[1] + T.log(ll[1]), -3.0e38)

        return partial

    return factory


def make_flash_decode_combine(_tl, H, D):
    tilelang, T = _tl()

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def factory():
        num_split = T.dynamic("s")
        m = T.dynamic("m")

        @T.prim_func
        def combine(
            PartialOut: T.Tensor([num_split, m, H, D], "float16"),
            PartialLse: T.Tensor([num_split, m, H], "float32"),
            Out: T.Tensor([m, H, D], "float16"),
        ):
            with T.Kernel(m, H, threads=D) as (b, h):
                mstar = T.alloc_fragment([1], "float32")
                lse = T.alloc_fragment([1], "float32")
                w = T.alloc_fragment([1], "float32")
                denom = T.alloc_fragment([1], "float32")
                acc = T.alloc_fragment([D], "float32")
                mstar[0] = -3.0e38
                denom[0] = 0.0
                for d in T.Parallel(D):
                    acc[d] = 0.0
                for sp in T.serial(num_split):
                    lse[0] = PartialLse[sp, b, h]
                    mstar[0] = T.max(mstar[0], lse[0])
                for sp in T.serial(num_split):
                    lse[0] = PartialLse[sp, b, h]
                    w[0] = T.if_then_else(
                        lse[0] > -3.0e38, T.exp(lse[0] - mstar[0]), 0.0)
                    denom[0] += w[0]
                    for d in T.Parallel(D):
                        acc[d] += w[0] * T.cast(
                            PartialOut[sp, b, h, d], "float32")
                for d in T.Parallel(D):
                    Out[b, h, d] = T.cast(acc[d] / denom[0], "float16")

        return combine

    return factory
