"""Group-aware critic health metrics (recipe §2).

A low value loss does **not** mean the critic is useful, and a global AUC of
~0.5 does not mean it is useless. The critic is a deterministic function of the
prefix, so every response in a group shares the same prompt-boundary value; with
dynamic sampling forcing each group to be mixed, a *global* AUC over individual
responses is structurally pinned near 0.5 (the recipe records 0.496 global
against 0.640 within-prompt on the same dump). Judging the critic therefore
needs metrics that respect the grouping:

* :func:`prompt_pearson` — does it read question difficulty at all?
* :func:`within_prompt_auc` — inside one question, does it rank a correct
  response above an incorrect one?
* :func:`variance_reduction` — does subtracting V beat the whitening baseline?

plus :func:`calibration_gap`, the mean signed error, which catches a value head
that has drifted off the reward scale while still ranking correctly.

All four take per-token tensors in the padded ``[rows, S]`` layout plus a
``group_ids`` array saying which prompt each row came from. They are pure
tensor/numpy work with no distributed calls, so they run identically offline
over a trajectory dump and online inside a trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch

__all__ = [
    "CriticDiagnostics",
    "critic_diagnostics",
    "prompt_pearson",
    "within_prompt_auc",
    "variance_reduction",
    "calibration_gap",
]


def _response_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean of ``values`` along the sequence → one scalar per row."""
    m = mask.float()
    return (values.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def prompt_pearson(
    row_value: torch.Tensor,   # [rows] response-mean V
    row_reward: torch.Tensor,  # [rows] sequence reward
    group_ids: torch.Tensor,   # [rows] prompt index
) -> float:
    """Pearson r between each prompt's mean V and its pass rate.

    Measures whether the critic predicts *question difficulty* — the thing a
    per-group baseline like GRPO gets for free and a critic has to learn. The
    recipe measures ≈0 here, i.e. the critic never learned difficulty; it is
    reported so that stays visible rather than being assumed.

    Returns ``nan`` when there are fewer than two prompts or either side is
    constant (correlation is undefined, not zero).
    """
    groups = torch.unique(group_ids)
    if groups.numel() < 2:
        return float("nan")
    v = torch.stack([row_value[group_ids == g].mean() for g in groups]).float()
    r = torch.stack([row_reward[group_ids == g].mean() for g in groups]).float()
    v_c, r_c = v - v.mean(), r - r.mean()
    denom = v_c.norm() * r_c.norm()
    if denom.item() == 0.0:
        return float("nan")
    return float((v_c @ r_c / denom).item())


def within_prompt_auc(
    row_value: torch.Tensor,    # [rows] response-mean V
    row_correct: torch.Tensor,  # [rows] bool
    group_ids: torch.Tensor,    # [rows] prompt index
) -> float:
    """Pooled-pairs AUC of V ranking correct above incorrect *within* a prompt.

    Every (correct, incorrect) pair inside a prompt contributes one comparison,
    and the pairs are pooled across prompts (so a prompt with more usable pairs
    counts for more). Ties score 0.5. Prompts that are all-correct or
    all-incorrect contribute nothing.

    Returns ``nan`` when no prompt has both classes — which is exactly what a
    non-mixed batch looks like, and is worth surfacing rather than reporting a
    fake 0.5.
    """
    wins = 0.0
    pairs = 0
    for g in torch.unique(group_ids):
        sel = group_ids == g
        v, c = row_value[sel].float(), row_correct[sel].bool()
        pos, neg = v[c], v[~c]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        wins += float((diff > 0).sum().item()) + 0.5 * float((diff == 0).sum().item())
        pairs += pos.numel() * neg.numel()
    if pairs == 0:
        return float("nan")
    return wins / pairs


def variance_reduction(
    values: torch.Tensor,   # [rows, S]
    returns: torch.Tensor,  # [rows, S]
    mask: torch.Tensor,     # [rows, S]
) -> float:
    """``1 − Var(return − V) / Var(return)`` over masked tokens.

    The direct question "is subtracting V better than subtracting nothing?".
    Positive means the critic beats the whitening baseline; **negative means it
    is actively worse than not subtracting at all**, which the recipe notes
    happens briefly right after a fresh warmup and should clear within ~10
    steps. The recipe's best observed value is 14%.
    """
    sel = mask.reshape(-1) > 0
    if int(sel.sum()) < 2:
        return float("nan")
    r = returns.reshape(-1)[sel].float()
    v = values.reshape(-1)[sel].float()
    var_r = r.var(unbiased=False)
    if var_r.item() == 0.0:
        return float("nan")
    return float((1.0 - (r - v).var(unbiased=False) / var_r).item())


def calibration_gap(
    values: torch.Tensor,   # [rows, S]
    returns: torch.Tensor,  # [rows, S]
    mask: torch.Tensor,     # [rows, S]
) -> float:
    """Mean signed error ``E[V − return]`` over masked tokens.

    Catches a value head sitting at the wrong level while still ranking
    correctly — invisible to AUC, and it biases every advantage by a constant.
    """
    m = mask.float()
    n = m.sum().clamp(min=1.0)
    return float((((values.float() - returns.float()) * m).sum() / n).item())


@dataclass(frozen=True)
class CriticDiagnostics:
    """The recipe's critic verdict, plus the raw loss for context."""

    value_loss: float
    critic_auc: float           # within-prompt, pooled pairs
    critic_var_ratio: float     # variance reduction; >0 beats whitening
    critic_calibration_gap: float
    prompt_pearson: float
    n_rows: int
    n_prompts: int
    n_tokens: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"value_loss={self.value_loss:.4f}  "
            f"within_prompt_auc={self.critic_auc:.4f}  "
            f"var_reduction={self.critic_var_ratio:+.4f}  "
            f"calib_gap={self.critic_calibration_gap:+.4f}  "
            f"prompt_pearson={self.prompt_pearson:+.4f}  "
            f"[{self.n_rows} rows / {self.n_prompts} prompts / {self.n_tokens} tokens]"
        )


def critic_diagnostics(
    values: torch.Tensor,     # [rows, S] per-token V
    returns: torch.Tensor,    # [rows, S] per-token return-to-go
    mask: torch.Tensor,       # [rows, S] float assistant mask
    rewards: torch.Tensor,    # [rows] sequence reward
    group_ids: torch.Tensor,  # [rows] prompt index
    *,
    correct_threshold: float = 0.5,
) -> CriticDiagnostics:
    """Compute all four gauges plus the masked value loss in one pass."""
    m = mask.float()
    n_tok = m.sum().clamp(min=1.0)
    loss = float(((((values.float() - returns.float()) ** 2) * m).sum() / n_tok).item())

    row_value = _response_mean(values, m)
    row_correct = rewards.float() >= correct_threshold

    return CriticDiagnostics(
        value_loss=loss,
        critic_auc=within_prompt_auc(row_value, row_correct, group_ids),
        critic_var_ratio=variance_reduction(values, returns, m),
        critic_calibration_gap=calibration_gap(values, returns, m),
        prompt_pearson=prompt_pearson(row_value, rewards.float(), group_ids),
        n_rows=int(values.shape[0]),
        n_prompts=int(torch.unique(group_ids).numel()),
        n_tokens=int(m.sum().item()),
    )
