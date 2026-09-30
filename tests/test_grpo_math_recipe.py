"""T11 MATH arm config: the single grpo_math_v100 recipe selects GRPO vs DAPO
purely via env, and both arms share advantage/clip/loss and clean40b geometry.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def recipe(monkeypatch):
    # Ensure a clean module with neither arm switch set, then return the
    # callable that reads env at call time (so no reload is needed per arm).
    monkeypatch.delenv("XRL_DYNAMIC_SAMPLING", raising=False)
    monkeypatch.delenv("XRL_OVERLONG_SHAPING", raising=False)
    monkeypatch.delenv("XRL_DYNAMIC_MAX_PROMPTS", raising=False)
    return importlib.import_module("recipe.grpo_math_v100")


def _rollout_cfg(recipe):
    g = next(x for x in recipe.SERVICE_GROUPS if x.id == "rollout")
    return g.config


def test_grpo_arm_has_no_dynamic_or_overlong(recipe, monkeypatch):
    grp = recipe._rollout_group().config
    assert grp.dynamic_sampling is False
    assert grp.dynamic_max_prompts is None
    assert grp.reward_shaping is None


def test_dapo_arm_enables_dynamic_overlong_and_default_cap(recipe, monkeypatch):
    monkeypatch.setenv("XRL_DYNAMIC_SAMPLING", "1")
    monkeypatch.setenv("XRL_OVERLONG_SHAPING", "1")
    cfg = recipe._rollout_group().config
    assert cfg.dynamic_sampling is True
    # 192 = 3x the 64-prompt target; this, not oversample_factor, caps draws.
    assert cfg.dynamic_max_prompts == 192
    assert cfg.reward_shaping == "meshy.reward:dapo_overlong_penalty"
    assert cfg.reward_shaping_kwargs == {
        "max_response_len": 4096, "cache_len": 1024}


def test_dynamic_max_prompts_env_override_to_128(recipe, monkeypatch):
    monkeypatch.setenv("XRL_DYNAMIC_SAMPLING", "1")
    monkeypatch.setenv("XRL_DYNAMIC_MAX_PROMPTS", "128")
    cfg = recipe._rollout_group().config
    assert cfg.dynamic_max_prompts == 128


def test_both_arms_pin_clean40b_geometry_and_clip(recipe):
    # Shared by both arms from the one recipe; must match clean40b's init
    # (max_tokens_per_micro=4096, mini=64 -> 8 updates/512, seq_align=64).
    tp = recipe._trainer_params()
    assert tp.max_tokens_per_micro == 4096
    assert tp.mini_batch_size == 64
    assert tp.seq_bucket == 64
    assert (tp.ppo_clip_eps_low, tp.ppo_clip_eps_high) == (0.2, 0.28)
