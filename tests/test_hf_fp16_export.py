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


def _rss_mib() -> int:
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)


def test_release_idle_host_memory_returns_to_baseline():
    """After gather(fp16) -> drop -> gc -> malloc_trim, RSS must return to
    within 100 MiB of the pre-gather baseline. Linux/proc only; skipped on mac.
    """
    import pytest

    if not os.path.exists("/proc/self/statm"):
        pytest.skip("proc/statm RSS check is Linux-only")
    from meshy.backend.titan.trainer import release_idle_host_memory

    n_tensors, per = 200, 4_000_000  # ~800M params fp32 = 3.2 GB transient
    baseline = _rss_mib()
    sd = {f"t{i}": torch.randn(per, dtype=torch.float32) for i in range(n_tensors)}
    at_gather = _rss_mib()
    assert at_gather - baseline > 1500  # sanity: gather actually used >1.5 GiB
    hf_export_to_fp16(sd)
    del sd
    release_idle_host_memory()
    after = _rss_mib()
    assert after - baseline <= 100, (
        f"RSS did not return: baseline {baseline} gather {at_gather} after {after}"
    )
