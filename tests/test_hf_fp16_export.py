from __future__ import annotations

import os
import tempfile

import torch
from safetensors.torch import load_file

from meshy.backend.titan.trainer import hf_export_to_fp16


def test_fp16_export_preserves_keys_and_values():
    torch.manual_seed(0)
    src = {
        "layer.weight": torch.randn(64, 128, dtype=torch.float32),
        "layer.bias": torch.randn(64, dtype=torch.float32),
        # already fp16 must be untouched (no error, stays fp16)
        "embed.weight": torch.randn(100, 64, dtype=torch.float16),
        # integer tensor (e.g. a buffer) is left as-is
        "counter": torch.arange(4, dtype=torch.int64),
    }
    fp32_snapshot = {
        "layer.weight": src["layer.weight"].clone(),
        "layer.bias": src["layer.bias"].clone(),
    }
    out = hf_export_to_fp16(src)
    # same object, same keys
    assert out is src
    assert set(out.keys()) == {
        "layer.weight", "layer.bias", "embed.weight", "counter"
    }
    # dtypes
    assert out["layer.weight"].dtype == torch.float16
    assert out["layer.bias"].dtype == torch.float16
    assert out["embed.weight"].dtype == torch.float16
    assert out["counter"].dtype == torch.int64
    # fp16 cast equals the fp32 source cast tensor-by-tensor
    assert torch.equal(out["layer.weight"], fp32_snapshot["layer.weight"].half())
    assert torch.equal(out["layer.bias"], fp32_snapshot["layer.bias"].half())


def test_fp16_safetensors_roundtrip_matches_fp32_cast():
    torch.manual_seed(1)
    state = {f"w{i}": torch.randn(128, 256) for i in range(4)}
    ref = {k: v.half().clone() for k, v in state.items()}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.safetensors")
        from safetensors.torch import save_file
        save_file(hf_export_to_fp16(state), path)
        loaded = load_file(path)
    assert set(loaded) == set(ref)
    for k in ref:
        assert loaded[k].dtype == torch.float16
        assert torch.equal(loaded[k], ref[k])


def test_fp16_export_cuts_bytes_roughly_half():
    sd = {"w": torch.randn(1000, 1000, dtype=torch.float32)}
    n_fp32 = sd["w"].numel() * 4
    hf_export_to_fp16(sd)
    n_fp16 = sd["w"].numel() * 2
    assert n_fp16 == n_fp32 // 2
