"""CPU regression for v100_run_rl.sh geometry precedence (the
"MATH block is a no-op after the generic := lines" eighth-batch trap).

The script cannot be sourced whole (it cd's and launches training), so these
tests extract just the recipe/geometry block and evaluate it under bash.
Precedence required: caller override > MATH default > generic (GSM8K) default.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "v100_run_rl.sh"


def _geometry_block() -> str:
    lines = SCRIPT.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith("# Experiment geometry. Resolve the recipe"))
    end = next(i for i in range(start, len(lines))
               if lines[i].startswith("export XRL_STEPS XRL_EVAL_EVERY"))
    # Stop right after that export line.
    return "\n".join(lines[start:end + 1])


def _run_geom(env: dict[str, str]) -> dict[str, str]:
    block = _geometry_block()
    cmd = block + "\n" + (
        'echo "RB=$XRL_ROLLOUT_BATCH GROUP=$XRL_GROUP_SIZE '
        'MB=$XRL_MINI_BATCH MTPM=$XRL_MAX_TOKENS_PER_MICRO '
        'SEQ=$XRL_SEQ_BUCKET '
        'W=$((XRL_ROLLOUT_BATCH*XRL_GROUP_SIZE)) '
        'UPD=$((XRL_ROLLOUT_BATCH*XRL_GROUP_SIZE/XRL_MINI_BATCH))"'
    )
    out = subprocess.run(["bash", "-c", cmd], env={**env, "PATH": "/usr/bin:/bin"},
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    kv = {}
    for tok in out.stdout.strip().splitlines()[-1].split():
        k, _, v = tok.partition("=")
        kv[k] = v
    return kv


def test_math_default_geometry_is_512x8():
    g = _run_geom({"XRL_RECIPE": "grpo_math_v100"})
    assert g["RB"] == "64" and g["GROUP"] == "8"
    assert g["MB"] == "64"
    assert g["MTPM"] == "4096" and g["SEQ"] == "64"
    assert g["W"] == "512" and g["UPD"] == "8"


def test_math_explicit_mini_batch_is_honoured():
    # Caller override must beat the MATH default (no FATAL).
    g = _run_geom({"XRL_RECIPE": "grpo_math_v100", "XRL_MINI_BATCH": "32"})
    assert g["MB"] == "32" and g["W"] == "512" and g["UPD"] == "16"


def test_gsm8k_generic_defaults_unchanged():
    g = _run_geom({"XRL_RECIPE": "grpo_gsm8k_v100"})
    assert g["RB"] == "8" and g["MB"] == "8"
    assert g["MTPM"] == "" and g["SEQ"] == "1024"
