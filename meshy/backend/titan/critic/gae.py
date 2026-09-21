"""GAE advantage estimation with the recipe's VAPO length-adaptive λ.

The recipe (`justrl_ii_recipe.md` §1) uses the VAPO trick: the GAE trace
parameter λ is not a constant but varies per sequence with its length,

    λᵢ = 1 − 1 / (α · Lᵢ)

so a longer reasoning chain gets a longer credit horizon instead of the
bootstrap decay the constant λ would impose. The recipe's α=1.5 is tuned for
128k-long generations (the upstream VAPO default α=0.05 is far too short for
that regime — a token budget of L=15k caps the credit half-life at ~500
tokens).

This module is pure tensor work, independent of the rest of the critic package:

* :func:`compute_vapo_gae` — per-token advantages from token rewards and
  values, segment-aware so the ``packed`` batch layout stays correct.
* :func:`compute_returns` — discounted return-to-go, the target the value head
  regresses onto.
* :func:`advantages_to_per_sequence` — the masked token-mean per-sample
  reduction that ``TitanTrainer``'s ``Batch.advantages`` ([n_docs]) consumes
  today, so an existing PPO loop can switch to critic advantages without a
  batch-schema change.

Both recursions operate on tensors in **true temporal order**: they walk
adjacent timesteps, so a CP-sharded (head-tail permuted) sequence must be
gathered — ``CpSharder.gather_seq`` — before it reaches this module.

Why they are not written as loops
---------------------------------
Both are first-order backward recursions over the sequence. Written as a
``for t in range(S-1, -1, -1)`` they cost ``S`` kernel launches — at
``S = 126976`` and a 480-row window that is tens of millions of launches per
window, which dominated the critic's wall clock. :func:`_reverse_scan` replaces
the loop with a closed form:

    A_t = (1/w_t) · Σ_{k=t..end(t)} w_k · x_k,     w_t = d^(t - start(t))

where ``d`` is the per-run decay (``γ·λ``, constant inside a run) and
``start``/``end`` delimit the maximal run of linked timesteps containing ``t``.
That is two cumulative sums and three gathers, regardless of ``S``.

The form is only usable while ``w`` stays representable. It does here by
construction: with γ=1 and λ=1−1/(α·L), ``w`` bottoms out at
``λ^L ≈ e^(-1/α)`` — 0.51 for the recipe's α=1.5 — so the dynamic range over a
full 128k sequence is less than 2×. :func:`_reverse_scan` checks that bound at
runtime and falls back to the exact sequential recursion when a configuration
(a strongly discounted γ, a tiny α) would underflow it, so the fast path is an
optimisation and never a semantic change.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "compute_vapo_gae",
    "compute_returns",
    "advantages_to_per_sequence",
]

#: Smallest ``w`` the closed form tolerates before it hands over to the
#: sequential recursion. Well above float32's denormal range, so the division
#: by ``w`` cannot amplify rounding into anything meaningful.
_W_FLOOR = 1e-12


def _lambda_per_doc(lengths: torch.Tensor, *, alpha: float, epsilon: float = 1e-6) -> torch.Tensor:
    """VAPO λᵢ = 1 − 1/(α·Lᵢ), per document, clamped to [0, 1)."""
    L = lengths.clamp(min=1).float()
    lam = 1.0 - 1.0 / (alpha * L)
    return lam.clamp(0.0, 1.0 - epsilon)


def _continuation_mask(mask: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
    """``1`` where position ``t+1`` continues the episode that ``t`` is in.

    A position continues its predecessor only when it is a real assistant token
    *in the same document*, which needs both inputs. ``doc_ids`` alone misses
    the ``padded`` layout's padding tail — ``batch.py`` gives the whole row the
    row's document id — and ``mask`` alone would let a prompt prefix collect the
    response's credit.
    """
    rows, S = mask.shape
    device = mask.device
    valid = mask > 0
    same_doc = torch.zeros(rows, S, dtype=torch.bool, device=device)
    same_doc[:, :-1] = doc_ids[:, 1:] == doc_ids[:, :-1]
    next_valid = torch.zeros(rows, S, dtype=torch.bool, device=device)
    next_valid[:, :-1] = valid[:, 1:]
    return (valid & same_doc & next_valid).to(torch.float32)


def _run_geometry(
    valid: torch.Tensor,  # [rows, S] bool
    cont: torch.Tensor,   # [rows, S] float/bool, links t -> t+1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-position ``(index, run start index, run end index)``.

    A *run* is a maximal set of consecutive valid positions linked by ``cont``.
    Both recursions reset at run boundaries, so a run is exactly the span a
    position's accumulator may reach.
    """
    rows, S = valid.shape
    device = valid.device
    idx = torch.arange(S, device=device).unsqueeze(0).expand(rows, S)
    linked = cont > 0

    from_prev = torch.zeros_like(valid)
    from_prev[:, 1:] = linked[:, :-1]
    starts = valid & ~from_prev
    # cummax over indices: monotone, so it is exact regardless of the payload.
    start = torch.cummax(torch.where(starts, idx, torch.full_like(idx, -1)), dim=1).values

    ends = valid & ~linked
    far = torch.full_like(idx, S)
    end = torch.flip(
        torch.cummin(torch.flip(torch.where(ends, idx, far), [1]), dim=1).values, [1]
    )
    return idx, start, end


def _reverse_scan(
    x: torch.Tensor,      # [rows, S] per-step source term, zero outside ``valid``
    decay: torch.Tensor,  # [rows, S] per-step decay, constant within a run
    valid: torch.Tensor,  # [rows, S] bool
    cont: torch.Tensor,   # [rows, S] links t -> t+1
) -> torch.Tensor:
    """``A_t = x_t + decay_t · cont_t · A_{t+1}``, without the sequential loop.

    ``decay`` carries γ·λ *without* ``cont`` folded in: run truncation comes
    from the run geometry, not from a zero factor, which keeps ``w`` away from
    zero. See the module docstring for the closed form and its validity bound.
    """
    rows, S = x.shape
    idx, start, end = _run_geometry(valid, cont)

    # Decay of the run each position belongs to, read at the run's first
    # position (constant within the run by contract).
    d_run = decay.gather(1, start.clamp(min=0))
    j = (idx - start).clamp(min=0)
    # ``w = d_run ** j`` via logs: j reaches 1e5, and repeated squaring in pow
    # is no more accurate here while being far slower on long rows.
    w = torch.exp(j.to(torch.float64) * torch.log(d_run.to(torch.float64).clamp(min=_W_FLOOR)))
    w = torch.where(valid, w, torch.ones_like(w))

    if bool(valid.any()) and bool((w[valid] < _W_FLOOR).any()):
        return _reverse_scan_reference(x, decay, valid, cont)

    wx = w * torch.where(valid, x, torch.zeros_like(x)).to(torch.float64)
    # Suffix sums, with a zero column appended so ``end + 1`` past the row end
    # reads 0 instead of needing a branch.
    suffix = torch.flip(torch.cumsum(torch.flip(wx, [1]), dim=1), [1])
    suffix = torch.cat([suffix, suffix.new_zeros(rows, 1)], dim=1)
    tail = suffix.gather(1, (end + 1).clamp(max=S))
    out = (suffix[:, :S] - tail) / w
    return torch.where(valid, out, torch.zeros_like(out)).to(x.dtype)


def _reverse_scan_reference(
    x: torch.Tensor,
    decay: torch.Tensor,
    valid: torch.Tensor,
    cont: torch.Tensor,
) -> torch.Tensor:
    """Sequential form of :func:`_reverse_scan` — the definition, and the oracle.

    Kept both as the numerical fallback and as what the unit tests compare the
    closed form against.
    """
    rows, S = x.shape
    x = torch.where(valid, x, torch.zeros_like(x))
    out = torch.zeros_like(x)
    acc = torch.zeros(rows, device=x.device, dtype=x.dtype)
    for t in range(S - 1, -1, -1):
        acc = x[:, t] + decay[:, t] * cont[:, t] * acc
        out[:, t] = acc
    return torch.where(valid, out, torch.zeros_like(out))


def compute_returns(
    rewards: torch.Tensor,   # [rows, S] per-token reward
    mask: torch.Tensor,      # [rows, S] float assistant-token mask
    doc_ids: torch.Tensor,   # [rows, S] segment index
    *,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Discounted return-to-go ``G_t = r_t + γ·G_{t+1}`` — the value target.

    This is what :meth:`CriticEngine.train_value` regresses onto. Segment-aware
    on the same terms as :func:`compute_vapo_gae`, so the two agree about where
    an episode ends. Zero outside ``mask``.

    With the recipe's setup (γ=1, a single outcome reward on the last response
    token) this is simply that reward broadcast across the response — the
    ``[R, R, ..., R]`` target — but it is written as a real recursion so
    per-token or shaped rewards work unchanged.
    """
    rewards = rewards.float()
    cont = _continuation_mask(mask, doc_ids)
    valid = mask > 0
    decay = torch.full_like(rewards, float(gamma))
    return _reverse_scan(rewards, decay, valid, cont)


def compute_vapo_gae(
    rewards: torch.Tensor,        # [rows, S] per-token reward (last assistant token nonzero)
    values: torch.Tensor,         # [rows, S] per-token V(s_t) from the critic
    mask: torch.Tensor,           # [rows, S] float assistant-token mask
    doc_ids: torch.Tensor,        # [rows, S] segment index; n_docs = padding slot
    n_docs: int,
    *,
    lengths: torch.Tensor | None = None,   # [n_docs] optional; derived from mask
    gamma: float = 1.0,
    alpha: float = 1.5,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VAPO GAE with per-sequence adaptive λ.

    Args:
        rewards: per-token reward signal. Usually zero except on the last
            assistant token (the outcome reward for that sequence).
        values: per-token critic value V(s_t).
        mask: float assistant-token mask, used both to size each document and
            to ignore padding in the ``lengths`` derivation.
        doc_ids: segment index per token, in ``[0, n_docs]``. The recursion is
            document-aware: together with ``mask`` it resets the bootstrap and
            the accumulator at the last real token of each document, so neither
            the ``packed`` layout (several samples concatenated into one row)
            nor the ``padded`` layout's padding tail leaks credit.
        n_docs: number of real samples; ``doc_ids == n_docs`` is the padding
            slot.
        lengths: ``[n_docs]`` document lengths for λ. Defaults to each
            document's masked token count, i.e. the number of *response* tokens
            — the horizon the recipe counts. Pass explicitly to measure
            something else.

    Returns:
        ``(adv, lambda_per_doc)`` where ``adv`` is the per-token generalised
        advantage ``[rows, S]`` and ``lambda_per_doc`` the adapted λ per
        document ``[n_docs]``. ``adv`` is zero outside ``mask``, but is
        otherwise **not** masked; apply ``mask`` at the loss site.

    Note:
        GAE is a sequential recursion over time, so ``reward``/``value`` must be
        in true temporal order. CP's head-tail sharding permutes the sequence;
        gather before calling.

        The index convention is the caller's to choose, and it must match the
        one the values were produced and trained under. The trainer passes
        *shifted* ``rewards``/``mask`` (position ``t`` = the state that emits
        token ``t+1``) so the advantage lands on the same grid as the PPO
        ratio; see ``meshy/backend/titan/batch.py``.
    """
    rows, S = rewards.shape
    # The recursion accumulates over up to 128k steps; fp32 keeps it from
    # drifting at bf16 precision.
    values = values.float()
    rewards = rewards.float()
    device = rewards.device

    flat_ids = doc_ids.reshape(-1)
    if lengths is None:
        cnt = mask.new_zeros(n_docs + 1, dtype=torch.float32).index_add_(
            0, flat_ids, mask.reshape(-1).float()
        )
        lengths = cnt[:n_docs]
    # Slot ``n_docs`` (padding) gets λ=0, so nothing bootstraps into or out of it.
    lam_doc = torch.cat(
        [_lambda_per_doc(lengths.to(device), alpha=alpha, epsilon=epsilon),
         rewards.new_zeros(1)]
    )
    lam = lam_doc[doc_ids]  # [rows, S]

    # TD residual: δ_t = r_t + γ·V_{t+1} − V_t. See :func:`_continuation_mask`
    # for why the episode boundary needs both ``mask`` and ``doc_ids``.
    valid = mask > 0
    cont = _continuation_mask(mask, doc_ids).to(values.dtype)  # [rows, S]

    # At a terminal token the episode ends, so V_{t+1} = 0 and δ_t = r_t − V_t
    # (the standard episodic-GAE convention). Keeping the baseline subtraction
    # here is the whole point of a critic: the outcome reward lands on the last
    # assistant token, and that is precisely where the value baseline must
    # reduce variance.
    shifted_v = torch.cat([values[:, 1:], values.new_zeros(rows, 1)], dim=1)
    delta = (rewards + gamma * shifted_v * cont - values) * valid.to(values.dtype)

    # A_t = δ_t + γ·λ_t·A_{t+1}, dropped wherever ``cont`` is 0 so no credit
    # crosses a document boundary or the padding gap.
    adv = _reverse_scan(delta, gamma * lam, valid, cont)
    return adv, lam_doc[:n_docs]


def advantages_to_per_sequence(
    adv: torch.Tensor,       # [rows, S_local] per-token advantages
    mask: torch.Tensor,      # [rows, S_local] float assistant-token mask
    doc_ids: torch.Tensor,   # [rows, S_local] -> segment index
    n_docs: int,
) -> torch.Tensor:
    """Masked token-mean of per-token advantages, one scalar per sample.

    This is the reduction :class:`meshy.backend.titan.Batch` expects
    (``[n_docs]``). The padding slot is ignored.
    """
    flat_ids = doc_ids.reshape(-1)
    seg_sum = adv.new_zeros(n_docs + 1).index_add_(
        0, flat_ids, (adv * mask).reshape(-1)
    )
    seg_cnt = mask.new_zeros(n_docs + 1).index_add_(0, flat_ids, mask.reshape(-1))
    return seg_sum[:n_docs] / seg_cnt[:n_docs].clamp(min=1)
