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
