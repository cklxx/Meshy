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


def test_resume_inference_boots_from_restored_step_hf(monkeypatch, tmp_path):
    # DCP step-10 + exported weights/actor_train-0/v10 present -> inference
    # must boot from v10, not the base model.
    ck = tmp_path / "ck" / "grpo_gsm8k_v100" / "r4" / "checkpoint" / "step-10"
    ck.mkdir(parents=True)
    (ck / ".metadata").write_text("x")
    rt = tmp_path / "run"
    v10 = rt / "weights" / "actor_train-0" / "v10"
    v10.mkdir(parents=True)
    (v10 / "model.safetensors").write_bytes(b"w")
    monkeypatch.setenv("XRL_CKPT_DIR", str(tmp_path / "ck"))
    monkeypatch.setenv("XRL_RUN_TAG", "r4")
    monkeypatch.setenv("XRL_RUNTIME_DIR", str(rt))
    base = "/models/Qwen3-0.6B"
    assert v100_windows.resume_inference_model_path(base) == str(v10)


def test_resume_inference_falls_back_without_hf_export(monkeypatch, tmp_path):
    ck = tmp_path / "ck" / "grpo_gsm8k_v100" / "r5" / "checkpoint" / "step-10"
    ck.mkdir(parents=True)
    (ck / ".metadata").write_text("x")
    monkeypatch.setenv("XRL_CKPT_DIR", str(tmp_path / "ck"))
    monkeypatch.setenv("XRL_RUN_TAG", "r5")
    monkeypatch.setenv("XRL_RUNTIME_DIR", str(tmp_path / "run"))  # no weights dir
    base = "/models/Qwen3-0.6B"
    assert v100_windows.resume_inference_model_path(base) == base


def test_resume_inference_base_on_cold_start(monkeypatch, tmp_path):
    monkeypatch.setenv("XRL_CKPT_DIR", str(tmp_path / "ck"))
    monkeypatch.setenv("XRL_RUN_TAG", "r6")
    monkeypatch.setenv("XRL_RUNTIME_DIR", str(tmp_path / "run"))
    base = "/models/Qwen3-0.6B"
    assert v100_windows.resume_inference_model_path(base) == base


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



# ---------- T7: persisted resume cursor (no-replay across hot starts) ------

def _set_runtime(monkeypatch, tmp_path, tag):
    monkeypatch.setenv("XRL_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("XRL_RUN_TAG", tag)
    monkeypatch.setenv("XRL_CKPT_DIR", str(tmp_path / "ck"))
    monkeypatch.delenv("XRL_START_WINDOW", raising=False)


def test_cursor_roundtrip_and_path_namespacing(monkeypatch, tmp_path):
    _set_runtime(monkeypatch, tmp_path, "runA")
    path = v100_windows.rollout_cursor_path()
    assert path.endswith(os.path.join("rollout", "runA", "rollout_cursor.json"))
    assert v100_windows.read_rollout_cursor() == 0
    v100_windows.write_rollout_cursor(5, consumed_prompts=40)
    assert v100_windows.read_rollout_cursor() == 5
    assert v100_windows.read_rollout_cursor_prompts() == 40
    monkeypatch.setenv("XRL_RUN_TAG", "runB")  # fresh run sees no cursor
    assert v100_windows.read_rollout_cursor() == 0


def test_resolve_start_window_uses_persisted_cursor_without_env(monkeypatch, tmp_path):
    _set_runtime(monkeypatch, tmp_path, "runA")
    v100_windows.write_rollout_cursor(10, consumed_prompts=80)
    plan = v100_windows.resolve_start_window(40)
    assert plan.resumed is True
    assert plan.start_window == 10
    assert plan.start_prompt == 80
    assert plan.eval_offset == 10


def test_explicit_start_window_overrides_cursor(monkeypatch, tmp_path):
    _set_runtime(monkeypatch, tmp_path, "runA")
    v100_windows.write_rollout_cursor(10, consumed_prompts=80)
    monkeypatch.setenv("XRL_START_WINDOW", "3")
    plan = v100_windows.resolve_start_window(40)
    assert plan.start_window == 3 and plan.start_prompt is None


def test_two_segment_resume_draws_disjoint_prompts(monkeypatch, tmp_path):
    """Reproduces the replay bug; asserts the persisted cursor fixes it."""
    import recipe.grpo_gsm8k_v100 as recipe
    import meshy.dataset.base as base
    from meshy.dataset.gsm8k import GSM8K

    _set_runtime(monkeypatch, tmp_path, "cleanRun")
    monkeypatch.setattr(base.datasets, "load_dataset", lambda **kw: _FakeHF(_rows(7473)))
    monkeypatch.setattr(
        GSM8K, "apply_chat_template", lambda self, d, b: int(d["answer"][5:])
    )
    bs, W = 64, 5

    def drain(start_window, batches, seek_prompt=None):
        ds = recipe.BoundedGSM8K(
            batch_size=bs, split="train", seed=42,
            start_window=start_window, batches_this_run=batches,
        )
        if seek_prompt is not None:
            ds.seek(seek_prompt)
        out = []
        while True:
            b = ds.next_batch(None)
            if not b:
                break
            out.extend(b)
        return ds, out

    ds_a, seg_a = drain(0, W)
    v100_windows.write_rollout_cursor(W, consumed_prompts=ds_a.global_position)
    assert ds_a.global_position == W * bs

    plan_b = v100_windows.resolve_start_window(40)
    # Segment B seeks to the cursor but, for this comparison, runs another W
    # windows; its run-length budget (batches_this_run) is a separate concern.
    _, seg_b = drain(plan_b.start_window, W, plan_b.start_prompt)

    assert set(seg_a).isdisjoint(set(seg_b))
    # One continuous BoundedGSM8K stream over 2W windows is the reference.
    _, ref = drain(0, 2 * W)
    assert seg_a + seg_b == ref


def test_cross_epoch_reshuffle_stays_reproducible(monkeypatch, tmp_path):
    """Past one epoch the order uses seed+epoch and a resume reproduces it."""
    import recipe.grpo_gsm8k_v100 as recipe
    import meshy.dataset.base as base
    from meshy.dataset.gsm8k import GSM8K

    _set_runtime(monkeypatch, tmp_path, "smallRun")
    n, bs = 100, 64
    monkeypatch.setattr(base.datasets, "load_dataset", lambda **kw: _FakeHF(_rows(n)))
    monkeypatch.setattr(
        GSM8K, "apply_chat_template", lambda self, d, b: int(d["answer"][5:])
    )

    def build_seek(seek_to):
        ds = recipe.BoundedGSM8K(
            batch_size=bs, split="train", seed=42, start_window=2, batches_this_run=2,
        )
        ds.seek(seek_to)
        return ds

    b1 = build_seek(140).next_batch(None)
    assert build_seek(140).next_batch(None) == b1
    assert len(b1) == 64  # wraps epoch1[40:100] + epoch2[0:4]
    epoch1 = _FakeHF(_rows(n)).shuffle(43)
    epoch2 = _FakeHF(_rows(n)).shuffle(44)
    e1 = [int(epoch1.select(range(40, 100))[i]["answer"][5:]) for i in range(60)]
    e2 = [int(epoch2.select(range(0, 4))[i]["answer"][5:]) for i in range(4)]
    assert b1 == e1 + e2


def test_dcp_checkpoint_dir_matches_recipe_segment(monkeypatch, tmp_path):
    """dcp_checkpoint_dir leaf recipe segment must equal each recipe's
    dump_folder, or a crashed MATH arm cannot find its DCP to resume."""
    ck = tmp_path / "ck"
    monkeypatch.setenv("XRL_CKPT_DIR", str(ck))
    monkeypatch.setenv("XRL_RUN_TAG", "math-dapo-40w")
    monkeypatch.setenv("XRL_RECIPE", "grpo_math_v100")
    assert v100_windows.dcp_checkpoint_dir() == str(
        ck / "grpo_math_v100" / "math-dapo-40w" / "checkpoint"
    )
    monkeypatch.setenv("XRL_RECIPE", "grpo_gsm8k_v100")
    monkeypatch.setenv("XRL_RUN_TAG", "gsm-run")
    assert v100_windows.dcp_checkpoint_dir() == str(
        ck / "grpo_gsm8k_v100" / "gsm-run" / "checkpoint"
    )


def test_dcp_checkpoint_dir_accepts_fully_qualified_recipe(monkeypatch, tmp_path):
    """launch.py receives recipe.grpo_math_v100; the leaf name must win even
    if the dotted form reaches the resolver (it writes the leaf back)."""
    monkeypatch.setenv("XRL_CKPT_DIR", str(tmp_path / "ck"))
    monkeypatch.setenv("XRL_RUN_TAG", "r")
    monkeypatch.setenv("XRL_RECIPE", "recipe.grpo_math_v100")
    assert v100_windows.dcp_checkpoint_dir().endswith(
        "grpo_math_v100/r/checkpoint"
    )
