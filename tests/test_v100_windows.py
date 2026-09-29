from __future__ import annotations

import os

import pytest

from meshy.dataset.base import Dataset
from recipe import v100_windows


# ---------- fake HF dataset ------------------------------------------------

class _FakeHF:
    """List-backed stand-in for a HF dataset: shuffle(seed) is a seeded
    permutation, select(range) returns a list of row dicts."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def shuffle(self, seed):
        import random

        idx = list(range(len(self.rows)))
        random.Random(seed).shuffle(idx)
        return _FakeHF([self.rows[i] for i in idx])

    def select(self, rng):
        return [self.rows[i] for i in rng]


def _rows(n):
    return [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(n)]


class _ProbeDataset(Dataset):
    """Dataset with a trivial template so no tokenizer is needed."""

    def _load(self, n, seed, **kw):
        self._base_dataset = _FakeHF(_rows(n))
        self.dataset = self._base_dataset
        self.batch_size = kw.get("batch_size", 64)
        self.base_seed = seed
        self.epoch = 0
        self.index = 0
        self.wrap_epochs = kw.get("wrap_epochs", False)
        self._apply_epoch(0)
        si = kw.get("start_index", 0)
        if si:
            self.seek(si)

    def apply_chat_template(self, data, builder):
        return data["question"]


def _make(n, seed=42, batch_size=64, **kw):
    ds = _ProbeDataset.__new__(_ProbeDataset)
    ds._load(n, seed, batch_size=batch_size, **kw)
    return ds


def _drain(ds, batches):
    out = []
    for _ in range(batches):
        b = ds.next_batch(None)
        if not b:
            break
        out.extend(b)
    return out


# ---------- offset / continuity --------------------------------------------

def test_start0_matches_baseline_order():
    n = 7473
    baseline = _make(n, seed=42, batch_size=64)
    ds0 = _make(n, seed=42, batch_size=64, start_index=0)
    assert _drain(ds0, 10) == _drain(baseline, 10)


def test_start_window10_first_prompt_is_641st():
    # start_window=10 with 64 prompts/window -> global prompt offset 640.
    n = 7473
    base = _make(n, seed=42, batch_size=64)
    first_896 = _drain(base, 14)  # 14 windows * 64 = 896 prompts
    warm = _make(n, seed=42, batch_size=64, start_index=10 * 64)
    first_warm = _drain(warm, 1)[0]
    assert first_warm == first_896[640]  # 641st prompt (0-based 640)


def test_adjacent_runs_concatenate_without_gap_or_duplicate():
    n = 7473
    reference = _make(n, seed=42, batch_size=64)
    ref = _drain(reference, 14)  # windows 0..14 -> global prompts 0..895

    run0 = _make(n, seed=42, batch_size=64, start_index=0 * 64)
    run10 = _make(n, seed=42, batch_size=64, start_index=10 * 64)
    run14 = _make(n, seed=42, batch_size=64, start_index=14 * 64)
    joined = (
        _drain(run0, 10)          # 0..639
        + _drain(run10, 4)        # 640..895
        + _drain(run14, 0)        # boundary sanity
    )
    assert joined == ref
    assert len(joined) == len(set(joined))  # no repeats


def test_wrap_into_new_epoch_no_internal_repeat():
    # Small split: offset 96 of 100 with batch 10 -> 4 rows epoch0 + 6 epoch1.
    n = 100
    ds = _make(n, seed=42, batch_size=10, start_index=96, wrap_epochs=True)
    batch = ds.next_batch(None)
    assert len(batch) == 10
    # Positions 96..99 of the seeded epoch-0 ordering, then 0..5 of epoch 1.
    epoch0 = _FakeHF(_rows(n)).shuffle(42)
    epoch1 = _FakeHF(_rows(n)).shuffle(43)
    expected_tail = [epoch0.select(range(96, 100))[i]["question"] for i in range(4)]
    expected_new = [epoch1.select(range(6))[i]["question"] for i in range(6)]
    assert batch[:4] == expected_tail
    assert batch[4:] == expected_new
    assert len(set(batch[4:])) == 6  # no repeat within the new epoch slice
    assert ds.epoch == 1 and ds.index == 6


def test_no_wrap_returns_empty_at_epoch_end():
    n = 100
    ds = _make(n, seed=42, batch_size=10, start_index=96, wrap_epochs=False)
    assert ds.next_batch(None) == []  # original stop-at-exhaustion behavior


# ---------- window plan -----------------------------------------------------

def _tagged_dcp(root, steps):
    for s in steps:
        os.makedirs(os.path.join(root, f"step-{s}"), exist_ok=True)


def test_plan_hf_warmstart_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv("XRL_CKPT_DIR", str(tmp_path / "ck"))
    monkeypatch.setenv("XRL_RUN_TAG", "r1")
    monkeypatch.setenv("XRL_START_WINDOW", "10")
    plan = v100_windows.resolve_start_window(40)
    assert not plan.resumed
    assert plan.start_window == 10
    assert plan.batches_this_run == 40
    assert plan.eval_offset == 10
    assert v100_windows.resolve_eval_offset() == 10


def test_plan_dcp_resume_wins_and_derives_remainder(monkeypatch, tmp_path):
    root = tmp_path / "ck"
    monkeypatch.setenv("XRL_CKPT_DIR", str(root))
    monkeypatch.setenv("XRL_RUN_TAG", "r2")
    monkeypatch.setenv("XRL_START_WINDOW", "10")  # must be ignored
    _tagged_dcp(root / "grpo_gsm8k_v100" / "r2" / "checkpoint", [4, 10])
    plan = v100_windows.resolve_start_window(14)
    assert plan.resumed
    assert plan.start_window == 10  # latest DCP step
    assert plan.batches_this_run == 4  # only the remainder
    assert plan.eval_offset == 0  # restored versions are absolute
    assert v100_windows.resolve_eval_offset() == 0


def test_plan_default_start_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("XRL_CKPT_DIR", str(tmp_path / "ck"))
    monkeypatch.setenv("XRL_RUN_TAG", "r3")
    monkeypatch.delenv("XRL_START_WINDOW", raising=False)
    plan = v100_windows.resolve_start_window(14)
    assert (plan.start_window, plan.eval_offset, plan.resumed) == (0, 0, False)
    assert plan.batches_this_run == 14


# ---------- BoundedGSM8K end to end (only where recipe imports) -------------

def test_bounded_gsm8k_seek_and_bound(monkeypatch):
    recipe = pytest.importorskip("recipe.grpo_gsm8k_v100")
    import meshy.dataset.base as base
    from meshy.dataset.gsm8k import GSM8K

    monkeypatch.setattr(
        base.datasets, "load_dataset", lambda **kw: _FakeHF(_rows(7473))
    )
    # Skip the chat template/tokenizer; the prompt identity is the row index
    # embedded in the answer (``_extract_gsm8k_answer`` parses "#### i").
    monkeypatch.setattr(
        GSM8K, "apply_chat_template", lambda self, d, b: int(d["answer"][5:])
    )

    def run(start_window, n_batches):
        ds = recipe.BoundedGSM8K(
            batch_size=64,
            split="train",
            seed=42,
            start_window=start_window,
            batches_this_run=n_batches,
        )
        out = []
        while True:
            b = ds.next_batch(None)
            if not b:
                return out
            out.extend(b)

    r0 = run(0, 10)
    r10 = run(10, 4)
    joined = r0 + r10  # windows 0..14 contiguous
    assert len(joined) == 14 * 64
    assert len(joined) == len(set(joined))  # no repeats across the warm start

    # A single stream over windows 0..14 must equal the two runs concatenated.
    full = recipe.BoundedGSM8K(
        batch_size=64, split="train", seed=42,
        start_window=0, batches_this_run=14,
    )
    ref = []
    while True:
        b = full.next_batch(None)
        if not b:
            break
        ref.extend(b)
    assert joined == ref

