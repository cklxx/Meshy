#!/bin/bash
# Final same-baseline holdout: 200 questions x4 on the last checkpoint
# (v300, which has no rollout window). Stands up its own SGLang server via
# eval_gsm8k.py. Run only after the RL process group exits cleanly.
set -euo pipefail

export PATH=/usr/local/cuda-12.4/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export MESHY_SGLANG_SM70=1
export MESHY_SM70_TILELANG=1
: "${MESHY_SM70_CUDA_GRAPH:=1}"
export MESHY_SM70_CUDA_GRAPH

cd /data00/meshy/rl/meshy
PY=/data00/meshy/venv/bin/python
export PYTHONPATH=/data00/meshy/rl/meshy

RUN_DIR="${1:?usage: run_rl_final_eval.sh <runtime_dir> [vN]}"
V="${2:-v300}"
CKPT="$RUN_DIR/weights/actor_train-0/$V"
[ -d "$CKPT" ] || { echo "missing $CKPT"; exit 2; }

OUT="/data00/meshy/rl/logs/final_eval_$(basename "$RUN_DIR")_${V}_$(date +%s).jsonl"
echo "FINAL_EVAL checkpoint=$CKPT out=$OUT graph=$MESHY_SM70_CUDA_GRAPH"
START=$(date +%s)
# 8192 to match the 200x4 baseline protocol (eval200_4s lenient 0.7525 was
# measured at 8192, not 4096).
$PY scripts/eval_gsm8k.py --model "$CKPT" --n 200 --samples 4 \
  --max-new-tokens 8192 --out "$OUT"
echo "FINAL_EVAL_DONE elapsed=$(( $(date +%s) - START ))s out=$OUT"
