from __future__ import annotations

import pytest


def _build(enable_checkpoint: bool, hf_path: str | None):
    torchtitan = pytest.importorskip("torchtitan")  # noqa: F841 (heavy deps live on GPU box)
    from meshy.backend.titan.config import build_forge_config
    from meshy.config import TrainerConfig

    trainer = TrainerConfig(
        model_name="qwen3",
        model_flavor="0.6B",
        attn_backend="sdpa",
        enable_checkpoint=enable_checkpoint,
    )
    return build_forge_config(trainer, hf_model_path=hf_path).checkpoint


def test_warmstart_dcp_off_is_load_only_but_still_enabled_for_load():
    # HF warm start + XRL_ENABLE_DCP_CKPT=0: load must work, save must not.
    ckpt = _build(False, "/tmp/fake-hf-model")
    assert ckpt.enable is True  # load() early-returns unless enabled
    assert ckpt.load_only is True  # _should_save() -> False


def test_warmstart_dcp_on_enables_both():
    ckpt = _build(True, "/tmp/fake-hf-model")
    assert ckpt.enable is True
    assert ckpt.load_only is False


def test_coldstart_dcp_off_not_load_only():
    # No HF path: enable follows the user flag; nothing forces load_only.
    ckpt = _build(False, None)
    assert ckpt.enable is False
    assert ckpt.load_only is False


def test_coldstart_dcp_on():
    ckpt = _build(True, None)
    assert ckpt.enable is True
    assert ckpt.load_only is False
