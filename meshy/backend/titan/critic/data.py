"""Read a rollout ``trajectory.jsonl`` into critic training batches.

The producer is :class:`meshy.worker.rollout.TrajectoryLogger`. It writes one
JSON object per rollout sample, and **only its verbose mode carries the token
arrays** the critic needs:

    {"round": 1, "reward": 1.0, "advantage": 0.0, "ground_truth": "...",
     "trajectory": [...messages...], "response_tokens": 412,
     "tokens": [...], "logprobs": [...], "masks": [...]}     <- verbose only

So the log must have been produced with ``verbose_trajectory_log: true``
(``meshy/config.py``); :func:`load_samples` says so explicitly rather than
failing on a missing key.

Grouping
--------
Two of the recipe's three critic gauges are *per prompt*, so rows have to be
attributable to the prompt they came from. The prompt is recovered from the
tokens themselves — the leading run of ``masks == 0`` — rather than from the
rendered ``trajectory`` messages, so it does not depend on chat-template
formatting. Rows whose prompt token ids match exactly share a group.

Masks and the index convention
------------------------------
``masks[i] == 1`` means token ``i`` is an assistant token. :func:`make_batch`
takes a ``shift`` flag that selects which of the two index conventions the
batch is expressed in:

``shift=False`` (raw)
    Position ``t`` *is* token ``t``. Useful for inspection and for the offline
    tools, which only ever look at one sequence at a time.

``shift=True`` (**the pipeline's convention**)
    Position ``t`` is the state that has consumed tokens ``<= t`` and is about
    to emit token ``t+1``. ``mask``/``rewards`` are shifted left by one;
    ``input_ids``/``positions`` are not, because the value head's output at
    position ``t`` already *is* the value of that state.

The shifted convention is the one the whole PPO pipeline uses: the actor's
``batch.py`` shifts its mask and rollout log-probs the same way so that
``new_lp[t] = log p(token t+1)``. Training the value head on the raw grid and
then consuming ``V`` on the shifted grid would put the baseline one token out
of step with the action it is supposed to baseline — and the last response
token would be regressed onto ``R`` when nothing remains to be earned. Keep the
critic and the trainer on the same flag.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch

__all__ = [
    "TrajectorySample",
    "load_samples",
    "make_batch",
    "sample_from_row",
    "samples_from_rows",
    "shift_left",
    "split_by_prompt",
]


@dataclass(frozen=True)
class TrajectorySample:
    tokens: list[int]
    masks: list[float]
    reward: float
    prompt_key: tuple[int, ...]
    group_id: int = -1

    @property
    def n_response(self) -> int:
        return int(sum(1 for m in self.masks if m))


def _prompt_key(tokens: list[int], masks: list[float]) -> tuple[int, ...]:
    """Leading run of non-assistant tokens — the prompt, as token ids."""
    cut = 0
    for i, m in enumerate(masks):
        if m:
            cut = i
            break
    else:
        cut = len(tokens)
    return tuple(tokens[:cut])


def load_samples(
    path: str | Path,
    *,
    max_seq_len: int,
    limit: int | None = None,
    drop_truncated: bool = False,
) -> list[TrajectorySample]:
    """Parse the JSONL log into samples, assigning a group id per prompt.

    Sequences longer than ``max_seq_len`` are dropped rather than truncated: a
    truncated tail would move the outcome reward onto a token that is not
    actually the end of the response, silently corrupting every value target in
    that row.
    """
    path = Path(path)
    raw: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
            if limit is not None and len(raw) >= limit:
                break

    if not raw:
        raise ValueError(f"{path}: no records")
    missing = [k for k in ("tokens", "masks") if k not in raw[0]]
    if missing:
        raise ValueError(
            f"{path}: records lack {missing}. The critic needs token-level data, "
            f"which TrajectoryLogger only writes in verbose mode — re-run the "
            f"rollout with verbose_trajectory_log: true."
        )

    samples: list[TrajectorySample] = []
    n_long = n_empty = 0
    for rec in raw:
        tokens = [int(t) for t in rec["tokens"]]
        masks = [float(m) for m in rec["masks"]]
        if len(tokens) != len(masks):
            raise ValueError(
                f"{path}: tokens ({len(tokens)}) and masks ({len(masks)}) differ in length"
            )
        if len(tokens) > max_seq_len:
            n_long += 1
            continue
        if drop_truncated and bool(rec.get("truncated", False)):
            continue
        if not any(masks):
            n_empty += 1
            continue
        samples.append(
            TrajectorySample(
                tokens=tokens,
                masks=masks,
                reward=float(rec.get("reward", 0.0)),
                prompt_key=_prompt_key(tokens, masks),
            )
        )

    if not samples:
        raise ValueError(
            f"{path}: every record was dropped "
            f"({n_long} over max_seq_len={max_seq_len}, {n_empty} with no response tokens)"
        )

    # Stable group ids, in first-seen order.
    order: dict[tuple[int, ...], int] = {}
    for s in samples:
        order.setdefault(s.prompt_key, len(order))
    samples = [
        TrajectorySample(s.tokens, s.masks, s.reward, s.prompt_key, order[s.prompt_key])
        for s in samples
    ]
    return samples


def sample_from_row(row: Any) -> TrajectorySample:
    """One live TQ rollout row -> :class:`TrajectorySample`.

    The offline path reads ``trajectory.jsonl``; the critic Service instead
    receives ``TensorDict`` rows off TransferQueue carrying the columns in
    ``meshy.config.CRITIC_INPUT_FIELDS``. Both feed the same
    :func:`make_batch`, so the GAE and gauge code has one input shape.

    ``group_id`` is left at ``-1``: rows arrive without a prompt identity, and
    grouping is recovered from the prompt tokens by
    :func:`samples_from_rows`.
    """
    tokens = [int(t) for t in row["tokens"].reshape(-1).tolist()]
    masks = [float(m) for m in row["mask_assistant"].reshape(-1).tolist()]
    if len(tokens) != len(masks):
        raise ValueError(
            f"row has {len(tokens)} tokens but {len(masks)} mask entries"
        )
    reward = float(row["reward"].reshape(-1)[0].item())
    return TrajectorySample(
        tokens=tokens,
        masks=masks,
        reward=reward,
        prompt_key=_prompt_key(tokens, masks),
        group_id=-1,
    )


def samples_from_rows(rows: Sequence[Any]) -> list[TrajectorySample]:
    """Convert TQ rows and assign a group id per distinct prompt.

    Two of the critic's three gauges are per-prompt, so rows must be
    attributable to the question they came from. Rollout rows carry no group
    column, so the prompt is recovered from the tokens themselves -- the
    leading run of ``mask == 0`` -- exactly as the offline loader does. That
    keeps the wire format unchanged: no ``group_id`` column has to be added to
    the rollout schema for the critic's sake.
    """
    samples = [sample_from_row(row) for row in rows]
    order: dict[tuple[int, ...], int] = {}
    for s in samples:
        if s.prompt_key not in order:
            order[s.prompt_key] = len(order)
    return [
        replace(s, group_id=order[s.prompt_key]) for s in samples
    ]


def split_by_prompt(
    samples: list[TrajectorySample],
    *,
    val_frac: float | None = None,
    val_size: int | None = None,
    seed: int = 0,
) -> tuple[list[TrajectorySample], list[TrajectorySample]]:
    """Train/val split **by prompt**, never by row.

    Splitting by row would put siblings of a validation response in the training
    set. The critic conditions on the prompt, so it would have already seen that
    prompt's difficulty — the held-out score would be measuring memorisation.
    """
    if (val_frac is None) == (val_size is None):
        raise ValueError("specify exactly one of val_frac or val_size")

    groups = sorted({s.group_id for s in samples})
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(groups), generator=g).tolist()
    shuffled = [groups[i] for i in perm]

    if val_size is None:
        assert val_frac is not None
        if not 0.0 <= val_frac < 1.0:
            raise ValueError("val_frac must be in [0, 1)")
        n_val = int(round(len(groups) * val_frac))
        val_groups = set(shuffled[:n_val])
    else:
        if val_size < 0 or val_size >= len(samples):
            raise ValueError(
                f"val_size must be in [0, {len(samples) - 1}], got {val_size}"
            )
        group_sizes = {
            group_id: sum(s.group_id == group_id for s in samples)
            for group_id in groups
        }
        # Subset-sum over whole prompt groups. This keeps the split leak-free
        # while still honoring an exact requested number of validation rows.
        parents: dict[int, tuple[int, int] | None] = {0: None}
        for group_id in shuffled:
            size = group_sizes[group_id]
            for total in sorted(list(parents), reverse=True):
                new_total = total + size
                if new_total <= val_size and new_total not in parents:
                    parents[new_total] = (total, group_id)
            if val_size in parents:
                break
        if val_size not in parents:
            sizes = sorted(set(group_sizes.values()))
            raise ValueError(
                f"cannot form val_size={val_size} without splitting a prompt group; "
                f"prompt group sizes present: {sizes}"
            )
        val_groups: set[int] = set()
        total = val_size
        while total:
            parent = parents[total]
            assert parent is not None
            total, group_id = parent
            val_groups.add(group_id)

    train = [s for s in samples if s.group_id not in val_groups]
    val = [s for s in samples if s.group_id in val_groups]
    return train, val


def shift_left(x: torch.Tensor) -> torch.Tensor:
    """``x[:, 1:] + [0]`` — move a per-token tensor onto the next-token grid.

    Equivalent to the actor's per-sample ``batch.py::_shift`` even on a padded
    row: everything past a row's own length is already zero, so shifting the
    whole row drops exactly the last real position, which is the one with no
    successor to account for.
    """
    return torch.cat([x[:, 1:], x.new_zeros(x.shape[0], 1)], dim=1)


def make_batch(
    batch: list[TrajectorySample],
    *,
    device: torch.device | str = "cpu",
    pad_to: int | None = None,
    shift: bool = False,
) -> dict[str, torch.Tensor]:
    """Padded ``[rows, S]`` batch — one sample per row.

    Returns ``input_ids``, ``positions``, ``mask``, ``doc_ids``, ``rewards``
    (per-token, on each row's last response position), ``row_rewards``
    ``[rows]`` and ``group_ids`` ``[rows]``.

    ``doc_ids`` follows the actor's ``padded`` convention (``batch.py``): the
    whole row carries the row's index, padding included. The padding is
    distinguished by ``mask``, which is what the GAE/returns code keys on.

    ``shift`` moves ``mask`` and ``rewards`` onto the next-token grid; see the
    module docstring for why the live pipeline always sets it. The masked token
    count per row is unchanged by the shift (a response never starts at index
    0), so the VAPO λ derived from it is the same either way.
    """
    rows = len(batch)
    S = pad_to or max(len(s.tokens) for s in batch)

    input_ids = torch.zeros(rows, S, dtype=torch.long)
    mask = torch.zeros(rows, S, dtype=torch.float32)
    rewards = torch.zeros(rows, S, dtype=torch.float32)
    for j, s in enumerate(batch):
        L = len(s.tokens)
        input_ids[j, :L] = torch.tensor(s.tokens, dtype=torch.long)
        mask[j, :L] = torch.tensor(s.masks, dtype=torch.float32)
        # Outcome reward on the sample's final response token.
        last = max(i for i, m in enumerate(s.masks) if m)
        rewards[j, last] = s.reward

    if shift:
        mask = shift_left(mask)
        rewards = shift_left(rewards)

    positions = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(rows, S).contiguous()
    doc_ids = torch.arange(rows, dtype=torch.long).unsqueeze(1).expand(rows, S).contiguous()

    out = {
        "input_ids": input_ids,
        "positions": positions,
        "mask": mask,
        "doc_ids": doc_ids,
        "rewards": rewards,
        "row_rewards": torch.tensor([s.reward for s in batch], dtype=torch.float32),
        "group_ids": torch.tensor([s.group_id for s in batch], dtype=torch.long),
    }
    return {k: v.to(device) for k, v in out.items()}
