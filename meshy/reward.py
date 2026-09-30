"""Per-sample reward shaping, applied by the rollout before anything else sees it.

A task reward function answers "is this response correct". Shaping answers "and
what should that be worth", which is a different question and a different place
in the pipeline: it belongs to the reward, not to the advantage estimator.

That distinction is load-bearing here. The DAPO soft-overlong penalty used to
live inside :mod:`meshy.advantage`, fused with the group-mean subtraction --
which meant a run with an external critic (``external_advantage=True``) skipped
the advantage pipeline entirely and silently lost the penalty with it. These
functions take one sample and return one number, so they apply identically
whether the advantage comes from a group baseline or from a value net.

Every function here has the signature::

    fn(reward: float, sample: Any, **kwargs) -> float

``sample`` is a :class:`meshy.utils.sample.Sample`, so a shaping rule can read
the mask, the token count or the ``truncated`` stamp.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

__all__ = [
    "response_length",
    "soft_overlong_penalty",
    "dapo_overlong_penalty",
]


def response_length(mask: Iterable[float]) -> int:
    """Length of the first contiguous run of response tokens in ``mask``.

    The first run, not the total, so a multi-turn sample is measured by the
    turn the length budget actually applies to.
    """
    length = 0
    in_response = False
    for value in mask:
        if value:
            in_response = True
            length += 1
        elif in_response:
            break
    return length


def soft_overlong_penalty(
    reward: float,
    sample: Any,
    *,
    max_response_len: int = 126976,
    buffer_len: int | None = None,
    penalty_factor: float | None = None,
) -> float:
    """DAPO's soft length penalty: taper the reward over the last ``buffer_len``.

    A response shorter than ``max_response_len - buffer_len`` is untouched.
    Past that, the penalty ramps linearly from 0 to ``-penalty_factor`` at the
    cap, so the model is pushed away from the length limit before it is cut off
    rather than only being punished once truncation has already destroyed the
    answer. ``justrl_ii_recipe.md`` §5 names runaway repetition burning to the
    full budget as the primary degradation mechanism; this is what deters it.

    The penalty is added to the raw reward, so a correct-but-overlong response
    can land below the solve threshold. That is intended -- and it is why
    ``raw_reward`` is published alongside, so solve-rate metrics keep measuring
    correctness rather than length (see ``meshy.backend.titan.metrics``).

    Defaults follow the environment variables the original pipeline documented.
    """
    buffer_len = (
        int(os.environ.get("OVERLONG_BUFFER_LEN", "25395"))
        if buffer_len is None
        else int(buffer_len)
    )
    penalty_factor = (
        float(os.environ.get("OVERLONG_PENALTY_FACTOR", "1.0"))
        if penalty_factor is None
        else float(penalty_factor)
    )
    if buffer_len <= 0:
        return float(reward)

    length = response_length(sample.masks)
    expected = int(max_response_len) - buffer_len
    if length <= expected:
        return float(reward)
    penalty = (expected - length) / buffer_len * penalty_factor
    return float(reward) + max(penalty, -penalty_factor)


def dapo_overlong_penalty(
    reward: float,
    sample: Any,
    *,
    max_response_len: int = 4096,
    cache_len: int = 1024,
) -> float:
    """DAPO overlong reward shaping with a hard truncated penalty.

    Piecewise penalty subtracted from the 0/1 correctness reward:

    * ``length <= L_max - L_cache``: no penalty;
    * inside the soft buffer ``(L_max - L_cache, L_max)``: linear penalty
      ramping 0 -> 1, ``(length - (L_max - L_cache)) / L_cache``;
    * ``length >= L_max`` or the response was truncated
      (``sample.truncated``): penalty 1.

    A response cut at the cap is wrong-shaped by definition, so it loses the
    full unit whether or not its final answer happened to parse; the explicit
    ``truncated`` stamp is what catches a response that stopped exactly on the
    boundary. Defaults L_max=4096 / L_cache=1024 per the MATH DAPO config.

    Unlike :func:`soft_overlong_penalty` this is an additive *subtraction in
    [0, 1]* matched to a 0/1 task reward, not the factor-scaled tapering used by
    the ±1 long-context recipes. Gated on/off by the recipe
    (``XRL_OVERLONG_SHAPING``); when unset the rollout leaves the reward alone.
    """
    max_response_len = int(max_response_len)
    cache_len = int(cache_len)
    if cache_len <= 0:
        raise ValueError("dapo_overlong_penalty requires cache_len > 0")
    soft_start = max_response_len - cache_len
    length = response_length(sample.masks)

    # A cut-off response pays the full unit regardless of its measured length.
    if bool(getattr(sample, "truncated", False)):
        penalty = 1.0
    elif length <= soft_start:
        return float(reward)
    elif length >= max_response_len:
        penalty = 1.0
    else:
        penalty = (length - soft_start) / cache_len
    return float(reward) - max(0.0, min(1.0, penalty))
