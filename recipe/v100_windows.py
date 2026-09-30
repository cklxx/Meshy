"""Single source of truth for where a resumed/continued RL run starts.

A training *window* consumes one prompt batch (``ROLLOUT_BATCH`` prompts,
each expanded to ``GROUP_SIZE`` completions) and produces one optimizer
schedule step. Three things must agree on the same starting window, and this
module resolves all three so they can never be double-applied:

* the dataset prompt offset (hot starts previously replayed prompts from 0);
* the number of prompt batches this run should emit;
* the in-loop eval alignment offset.

Two mutually exclusive ways to start partway through the curriculum:

* DCP resume — a ``step-N`` directory exists under this run's checkpoint
  folder. TorchTitan restores model + optimizer + scheduler + absolute step
  ``N``, so data resumes at window ``N`` and weight versions stay absolute
  (eval offset 0).
* HF/weight-only warm start (``XRL_START_WINDOW=W``) — there is no DCP
  directory; the trainer's step resets to 0, so data starts at prompt window
  ``W`` and the eval offset is ``W`` to keep cumulative alignment.

DCP wins when present; the env knob is then ignored (never stacked on top).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

_DCP_STEP_RE = re.compile(r"step-(\d+)$")

#: Authoritative resume cursor the rollout process advances once per emitted
#: window. Unlike DCP (which an Adam-reset / HF warm start deliberately discards
#: by using a fresh run tag) this lives under a fixed per-curriculum path in the
#: runtime root and survives optimizer resets, so a continued run finds the next
#: unseen prompt window even when no step-N checkpoint matches its run tag.
ROLLOUT_CURSOR_FILE = "rollout_cursor.json"


def _runtime_root() -> str:
    return os.environ.get("XRL_RUNTIME_DIR", "").rstrip("/")


def rollout_cursor_path() -> str | None:
    """Filesystem path of the resume cursor, or None without a runtime root.

    Namespaced by run tag (the same key DCP checkpoints use) so a brand-new run
    under a fresh ``XRL_RUN_TAG`` never inherits a previous experiment's cursor,
    while an Adam-reset continuation that keeps the same tag -- by design --
    resumes where it left off.
    """
    root = _runtime_root()
    if not root:
        return None
    return os.path.join(root, "rollout", run_tag(), ROLLOUT_CURSOR_FILE)


def read_rollout_cursor() -> int:
    """Number of prompt windows already consumed (0 if absent/unreadable)."""
    path = rollout_cursor_path()
    if not path or not os.path.isfile(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return max(0, int(data.get("consumed_windows", 0)))
    except (OSError, ValueError, TypeError):
        return 0


def read_rollout_cursor_prompts() -> int:
    """Number of prompts already consumed (authoritative no-replay offset).

    Prefer this over ``consumed_windows * batch_size``: with DAPO dynamic
    sampling a window draws replacement prompts, so prompt and window counts do
    not stay proportional.
    """
    path = rollout_cursor_path()
    if not path or not os.path.isfile(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return max(0, int(data.get("consumed_prompts", 0)))
    except (OSError, ValueError, TypeError):
        return 0


def write_rollout_cursor(consumed_windows: int, consumed_prompts: int | None = None) -> None:
    """Atomically persist the resume cursor (best-effort).

    ``consumed_windows`` drives eval/version alignment; ``consumed_prompts`` (the
    dataset's global prompt position) is the no-replay seek target.
    """
    path = rollout_cursor_path()
    if not path:
        return
    consumed_windows = max(0, int(consumed_windows))
    if consumed_prompts is None:
        consumed_prompts = consumed_windows
    payload = {
        "consumed_windows": consumed_windows,
        "consumed_prompts": max(0, int(consumed_prompts)),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)



def run_tag() -> str:
    """Resolve the per-run tag exactly as the trainer's dump_folder does."""
    tag = os.environ.get("XRL_RUN_TAG")
    if tag:
        return tag
    runtime = os.environ.get("XRL_RUNTIME_DIR", "").rstrip("/")
    return os.path.basename(runtime) if runtime else "default"


def dcp_checkpoint_dir() -> str:
    """Directory holding this run's ``step-N`` DCP checkpoints."""
    storage_root = os.environ.get("XRL_STORAGE_ROOT", "/3fs/stage/meshy")
    ckpt_dir = os.environ.get(
        "XRL_CKPT_DIR", os.path.join(storage_root, "ckpt")
    )
    return os.path.join(ckpt_dir, "grpo_gsm8k_v100", run_tag(), "checkpoint")


def latest_dcp_step(root: str | None = None) -> int | None:
    """Highest ``step-N`` in this run's checkpoint dir, or None if absent."""
    root = root if root is not None else dcp_checkpoint_dir()
    if not os.path.isdir(root):
        return None
    steps = [
        int(m.group(1))
        for name in os.listdir(root)
        if (m := _DCP_STEP_RE.match(name))
        and os.path.isdir(os.path.join(root, name))
    ]
    return max(steps) if steps else None


@dataclass(frozen=True)
class WindowPlan:
    #: Global prompt window at which this run's first batch starts.
    start_window: int
    #: Prompt batches this run emits (dataset bound; run stops at this many).
    batches_this_run: int
    #: Offset added to the (re)starting weight version for cumulative eval
    #: alignment. 0 for a DCP resume (versions already absolute), else
    #: ``start_window`` (HF warm start numbers versions from 0).
    eval_offset: int
    resumed: bool
    #: Global prompt offset to seek the dataset to. Equals
    #: ``start_window * prompts_per_window`` except after dynamic sampling
    #: (replacement draws advance the prompt cursor past the window*batch grid);
    #: None when it is exactly ``start_window * batch_size`` (the common case).
    start_prompt: int | None = None


def resolve_start_window(rl_steps: int) -> WindowPlan:
    # Explicit XRL_START_WINDOW always wins (operator knows the curriculum
    # position, e.g. a deliberately replayed run).
    explicit = os.environ.get("XRL_START_WINDOW")
    dcp_step = latest_dcp_step()
    if dcp_step is not None:
        return WindowPlan(
            start_window=dcp_step,
            batches_this_run=max(0, rl_steps - dcp_step),
            eval_offset=0,
            resumed=True,
        )
    if explicit is not None:
        start = int(explicit)
        return WindowPlan(
            start_window=start,
            batches_this_run=rl_steps,
            eval_offset=start,
            resumed=False,
        )
    # No DCP and no explicit offset (the Adam-reset / HF warm-start case that
    # used to replay from window 0): recover the data position from the
    # rollout's persisted cursor so a continued run draws unseen prompts.
    cursor = read_rollout_cursor()
    cursor_prompts = read_rollout_cursor_prompts()
    if cursor > 0:
        return WindowPlan(
            start_window=cursor,
            batches_this_run=rl_steps,
            eval_offset=cursor,
            resumed=True,
            start_prompt=cursor_prompts if cursor_prompts > 0 else None,
        )
    return WindowPlan(
        start_window=0,
        batches_this_run=rl_steps,
        eval_offset=0,
        resumed=False,
    )


def resolve_eval_offset() -> int:
    """Cumulative-eval offset (0 on DCP resume, else XRL_START_WINDOW).

    Mirrors :func:`resolve_start_window` without needing ``rl_steps`` so the
    in-loop eval hook (a different module) can stay decoupled from the recipe.
    """
    dcp_step = latest_dcp_step()
    return 0 if dcp_step is not None else int(
        os.environ.get("XRL_START_WINDOW", "0")
    )


def resume_inference_model_path(base_path: str) -> str:
    """HF weights the inference engine must boot from on a DCP resume.

    On a DCP resume the trainer restores model *and* optimizer from
    ``step-N``, but a freshly started SGLang would otherwise boot from the
    configured base model and the genesis GPU grant carries no checkpoint, so
    it generates the first window with base weights labelled version N
    (off-policy, mis-versioned). Booting inference from the matching exported
    HF dir ``weights/actor_train-0/vN`` makes genesis weights equal the
    restored trainer step. Returns ``base_path`` on a cold start or when the
    export is missing.
    """
    step = latest_dcp_step()
    if step is None:
        return base_path
    runtime = os.environ.get("XRL_RUNTIME_DIR", "")
    if not runtime:
        return base_path
    weights = os.path.join(runtime, "weights", "actor_train-0", f"v{step}")
    if os.path.isfile(os.path.join(weights, "model.safetensors")):
        return weights
    return base_path
