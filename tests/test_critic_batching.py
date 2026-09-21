"""The critic's window planning and its streaming gauges (no GPU, no model).

Two things the critic used to get wrong and now depends on the planner for:

* the window is **split** across the DP mesh -- it used to be replicated, so
  ``dp_shard_degree=2`` bought memory and no throughput;
* micro-batches are sized by a **token budget** -- they used to be a fixed row
  count, so a 480-row window was 480 forwards whatever the rows contained.

Plus the gauges, which had to become streaming sums for either of those to be
computable: rank 0 no longer holds the whole window.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from meshy.backend.titan.critic.metrics import (
    Accumulator,
    critic_diagnostics,
    diagnostics_from_states,
)
from meshy.backend.titan.plan import PlannerConfig, build_plan

ALIGN = 8
SEQ_LEN = 4096


def _planner_config(**kw) -> PlannerConfig:
    base = dict(
        layout="padded",
        mini_batch_size=240,
        seq_len=SEQ_LEN,
        align=ALIGN,
        max_tokens_per_micro=None,
        micro_batch_size=None,
    )
    base.update(kw)
    return PlannerConfig(**base)


def _lengths(n: int, seed: int = 0) -> list[int]:
    """Lengths with the live run's shape: a long tail over a short median."""
    rng = random.Random(seed)
    return [min(SEQ_LEN, max(16, int(rng.lognormvariate(5.6, 1.1)))) for _ in range(n)]


# ── DP split ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dp_size", [1, 2, 4])
def test_every_row_is_owned_by_exactly_one_dp_rank(dp_size):
    lengths = _lengths(480)
    plan = build_plan(
        lengths, lengths, dp_size,
        _planner_config(mini_batch_size=480 // dp_size, max_tokens_per_micro=2048),
    )

    owned = [g for r in range(dp_size) for g in plan.local_indices[r]]
    assert sorted(owned) == list(range(480)), "rows lost or duplicated by the DP split"
    assert all(len(plan.local_indices[r]) == 480 // dp_size for r in range(dp_size))


def test_dp_split_balances_tokens_not_row_counts():
    """A row-count split would hand one rank the whole tail."""
    lengths = _lengths(480)
    plan = build_plan(
        lengths, lengths, 2,
        _planner_config(mini_batch_size=240, max_tokens_per_micro=2048),
    )
    per_rank = [sum(lengths[g] for g in plan.local_indices[r]) for r in range(2)]
    assert max(per_rank) / min(per_rank) < 1.02


def test_the_plan_is_the_same_on_every_rank():
    """Ranks derive the plan from the broadcast rows; none of it is communicated."""
    lengths = _lengths(480)
    cfg = _planner_config(max_tokens_per_micro=2048)
    a = build_plan(lengths, lengths, 2, cfg)
    b = build_plan(lengths, lengths, 2, cfg)
    assert a.local_indices == b.local_indices
    assert a.per_rank == b.per_rank


# ── micro-batch packing ─────────────────────────────────────────────────
def test_a_token_budget_collapses_the_forward_count():
    """The point of the change: far fewer forwards for the same rows."""
    lengths = _lengths(480)
    by_rows = build_plan(
        lengths, lengths, 2, _planner_config(micro_batch_size=1)
    )
    by_tokens = build_plan(
        lengths, lengths, 2, _planner_config(max_tokens_per_micro=2048)
    )

    n_rows = sum(len(m.micros) for m in by_rows.per_rank[0])
    n_tokens = sum(len(m.micros) for m in by_tokens.per_rank[0])
    assert n_rows == 240, "micro_batch_size=1 must still mean one row per forward"
    assert n_tokens < n_rows / 2


def test_no_micro_batch_exceeds_the_budget_and_long_rows_still_fit():
    budget = 2048
    lengths = _lengths(480)
    plan = build_plan(lengths, lengths, 2, _planner_config(max_tokens_per_micro=budget))

    for rank in range(2):
        for mini in plan.per_rank[rank]:
            for micro in mini.micros:
                if micro.is_filler:
                    continue
                cost = len(micro.sample_idx) * micro.seq_len
                # A row longer than the whole budget gets a forward to itself
                # rather than being dropped or truncated.
                assert cost <= budget or len(micro.sample_idx) == 1
                assert micro.seq_len % ALIGN == 0
                assert micro.seq_len >= max(micro.doc_lens)


def test_every_rank_runs_the_same_number_of_forwards():
    """FSDP issues collectives per forward; an unequal count deadlocks."""
    lengths = _lengths(480)
    plan = build_plan(lengths, lengths, 2, _planner_config(max_tokens_per_micro=2048))
    counts = {
        rank: [len(mini.micros) for mini in plan.per_rank[rank]] for rank in range(2)
    }
    assert counts[0] == counts[1]


def test_padding_overhead_stays_small():
    """Length-sorted packing is what keeps the padded layout affordable."""
    lengths = _lengths(480)
    plan = build_plan(lengths, lengths, 2, _planner_config(max_tokens_per_micro=2048))

    real = forward = 0
    for mini in plan.per_rank[0]:
        for micro in mini.micros:
            real += micro.n_tokens
            forward += len(micro.sample_idx) * micro.seq_len
    assert 1.0 - real / forward < 0.15


# ── streaming gauges ────────────────────────────────────────────────────
def _window(rows: int = 24, span: int = 40, seed: int = 0):
    torch.manual_seed(seed)
    mask = torch.zeros(rows, span)
    for j in range(rows):
        mask[j, 3 + j % 5 :] = 1.0
    returns = torch.randn(rows, span) * mask
    values = (returns + 0.3 * torch.randn(rows, span)) * mask
    rewards = (torch.rand(rows) > 0.5).float()
    groups = torch.arange(rows) // 4
    return values, returns, mask, rewards, groups


def test_streaming_gauges_match_the_whole_window_form():
    values, returns, mask, rewards, groups = _window()
    want = critic_diagnostics(values, returns, mask, rewards, groups)

    acc = Accumulator()
    for lo in range(0, values.shape[0], 5):
        hi = lo + 5
        acc.update(
            values[lo:hi], returns[lo:hi], mask[lo:hi], rewards[lo:hi], groups[lo:hi]
        )
    got = diagnostics_from_states([acc.state])

    for key, expected in want.as_dict().items():
        actual = got.as_dict()[key]
        if isinstance(expected, float) and math.isnan(expected):
            assert math.isnan(actual), key
        else:
            assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6), key


def test_gauges_merge_across_ranks_regardless_of_row_order():
    """Rank 0 merges partials from a DP split; the gauges must not see the seam."""
    values, returns, mask, rewards, groups = _window(rows=24, seed=1)
    want = critic_diagnostics(values, returns, mask, rewards, groups)

    # An interleaved split, like the planner's length-sorted snake deal.
    states = []
    for rank in range(2):
        sel = torch.arange(rank, 24, 2)
        acc = Accumulator()
        acc.update(values[sel], returns[sel], mask[sel], rewards[sel], groups[sel])
        states.append(acc.state)
    got = diagnostics_from_states(states)

    assert got.n_rows == want.n_rows
    assert got.n_tokens == want.n_tokens
    for key, expected in want.as_dict().items():
        actual = got.as_dict()[key]
        if isinstance(expected, float) and math.isnan(expected):
            assert math.isnan(actual), key
        else:
            assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6), key


def test_filler_rows_do_not_move_the_gauges():
    """Rank-equalising filler micro-batches carry an all-zero mask."""
    values, returns, mask, rewards, groups = _window(seed=2)
    want = critic_diagnostics(values, returns, mask, rewards, groups)

    acc = Accumulator()
    acc.update(values, returns, mask, rewards, groups)
    plain = diagnostics_from_states([acc.state])

    assert plain.value_loss == pytest.approx(want.value_loss, rel=1e-5)
    assert plain.n_tokens == want.n_tokens
