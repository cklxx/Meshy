"""Unbiased pass@k estimator (Chen et al., 2021, arXiv:2107.03374).

For each problem with ``n`` samples of which ``c`` are correct, the unbiased
estimator is::

    pass@k = 1 - C(n - c, k) / C(n, k)

with the conventions C(a, k) = 0 for a < k (so pass@k = 1 when a solution is
known to exist but cannot be excluded by n - k samples) and 0 for c = 0.
Computed with the product form to avoid large factorials.
"""

from __future__ import annotations

import math


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem (n samples, c correct)."""
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def aggregate_pass_at_k(counts: list[tuple[int, int]],
                       ks: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
                       ) -> dict[int, dict[str, float]]:
    """Mean pass@k over problems.

    ``counts`` is ``(n, c)`` per problem. Reports each k that every problem has
    enough samples for (``k <= min(n)``) plus the number of contributing
    problems, so a partial-k report (e.g. k=32 with n=8) is dropped loudly
    rather than silently biased.
    """
    if not counts:
        return {}
    out: dict[int, dict[str, float]] = {}
    for k in ks:
        # Each k uses the problems that have at least k samples; a k no problem
        # supports is skipped, never silently computed on fewer samples.
        eligible = [(n, c) for n, c in counts if n >= k]
        if not eligible:
            continue
        vals = [pass_at_k(n, c, k) for n, c in eligible]
        out[k] = {
            "pass_at_k": sum(vals) / len(vals),
            "problems": len(eligible),
            "samples_per_problem": min(n for n, _ in eligible),
        }
    return out
