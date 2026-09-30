"""DAPO overlong reward shaping piecewise boundaries (T10).

L_max=4096, L_cache=1024 -> soft region [3072, 4096); truncated always -1.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from meshy.reward import dapo_overlong_penalty

L_MAX, L_CACHE, SOFT = 4096, 1024, 3072


def _sample(length: int, *, truncated: bool = False):
    return SimpleNamespace(masks=[1] * length, truncated=truncated)


@pytest.mark.parametrize("length", [0, 100, SOFT - 1, SOFT])
def test_no_penalty_below_soft_start(length):
    assert dapo_overlong_penalty(1.0, _sample(length), max_response_len=L_MAX, cache_len=L_CACHE) == 1.0
    assert dapo_overlong_penalty(0.0, _sample(length), max_response_len=L_MAX, cache_len=L_CACHE) == 0.0


def test_penalty_ramps_linearly_inside_buffer():
    first = dapo_overlong_penalty(1.0, _sample(SOFT + 1), max_response_len=L_MAX, cache_len=L_CACHE)
    assert first == pytest.approx(1.0 - 1.0 / L_CACHE)
    mid_len = SOFT + L_CACHE // 2
    assert dapo_overlong_penalty(1.0, _sample(mid_len), max_response_len=L_MAX, cache_len=L_CACHE) == pytest.approx(0.5)
    penultimate = dapo_overlong_penalty(1.0, _sample(L_MAX - 1), max_response_len=L_MAX, cache_len=L_CACHE)
    assert penultimate == pytest.approx(1.0 / L_CACHE)


def test_full_penalty_at_and_above_cap():
    assert dapo_overlong_penalty(1.0, _sample(L_MAX), max_response_len=L_MAX, cache_len=L_CACHE) == pytest.approx(0.0)
    assert dapo_overlong_penalty(1.0, _sample(L_MAX + 500), max_response_len=L_MAX, cache_len=L_CACHE) == pytest.approx(0.0)
    # Penalty stacks on the 0/1 correctness reward: a wrong answer goes negative.
    assert dapo_overlong_penalty(0.0, _sample(L_MAX), max_response_len=L_MAX, cache_len=L_CACHE) == pytest.approx(-1.0)


def test_truncated_pays_full_unit_regardless_of_length():
    # Short response that was nonetheless cut: full penalty, length ignored.
    assert dapo_overlong_penalty(1.0, _sample(100, truncated=True), max_response_len=L_MAX, cache_len=L_CACHE) == 0.0


def test_nonpositive_cache_rejected():
    with pytest.raises(ValueError):
        dapo_overlong_penalty(1.0, _sample(10), max_response_len=L_MAX, cache_len=0)
