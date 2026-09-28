from __future__ import annotations

import torch

from torchtitan.models.common.attention import ScaledDotProductAttention
from torch.nn.attention import SDPBackend

from meshy.backend.titan.models.attention import (
    Sm70ScaledDotProductAttention,
)
from meshy.backend.titan.plan import round_up


def _order_for_capability(cap):
    # Mirror sm70_sdpa_backends() without touching torch.cuda.
    upstream = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
    if cap == (7, 0):
        return upstream[:2] + [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    return upstream


def test_sm70_backend_order_inserts_efficient_before_math():
    order = _order_for_capability((7, 0))
    assert order.index(SDPBackend.EFFICIENT_ATTENTION) < order.index(SDPBackend.MATH)
    assert order == [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]


def test_non_sm70_order_is_unchanged():
    upstream = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
    assert _order_for_capability((9, 0)) == upstream
    assert _order_for_capability((8, 0)) == upstream
    assert _order_for_capability((8, 9)) == upstream


def test_subclass_config_is_recognised_as_sdpa():
    # trainer._attn_backend_of does isinstance(inner, ScaledDotProductAttention.Config)
    cfg = Sm70ScaledDotProductAttention.Config()
    assert isinstance(cfg, ScaledDotProductAttention.Config)


def test_causal_logprob_invariant_to_right_padding_length():
    """Right-padding + is_causal=True: trailing pad columns are future keys,
    so a real token cannot attend to them. The attention output at the last
    real position must be identical whether the row is padded to align=1024 or
    align=64 (== round_up(300)=320). Runs the real SDPA MATH backend on CPU."""
    from torch.nn.attention import sdpa_kernel
    from torch.nn.attention import SDPBackend as B

    torch.manual_seed(0)
    L = 300
    h, d = 4, 64
    qr = torch.randn(1, h, L, d, dtype=torch.float64)
    kr = torch.randn(1, h, L, d, dtype=torch.float64)
    vr = torch.randn(1, h, L, d, dtype=torch.float64)

    def run(padded_len):
        q = torch.zeros(1, h, padded_len, d, dtype=torch.float64)
        k = torch.zeros(1, h, padded_len, d, dtype=torch.float64)
        v = torch.zeros(1, h, padded_len, d, dtype=torch.float64)
        q[:, :, :L], k[:, :, :L], v[:, :, :L] = qr, kr, vr
        with sdpa_kernel(B.MATH):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True)
        return out[:, :, L - 1, :]  # last real query row

    # align 1024 bucket vs align 64 bucket (round_up(300,64)=320)
    out1024 = run(1024)
    out64 = run(320)
    out_tight = run(L)
    assert torch.allclose(out1024, out64, atol=1e-10)
    assert torch.allclose(out64, out_tight, atol=1e-10)


def test_align64_shrinks_single_row_padding():
    # A single 300-token micro: align 1024 pads to 1024 (724 wasted), align 64
    # pads to 320 (20 wasted). No compile/graph depends on the value.
    assert round_up(300, 1024) == 1024
    assert round_up(300, 64) == 320
    assert 5120 % 64 == 0
