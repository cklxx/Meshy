#!/bin/bash
# Offline holdout eval over the five cont-run checkpoints, same 4096 protocol,
# one dedicated SGLang server at a time (GPU is idle after RL_DONE).
# Usage: bash v100_offline_eval.sh <4096|8192> <samples> <tag> <ckpt_dir> [more ckpt_dirs...]
set -euo pipefail
PROTO="$1"; SAMPLES="$2"; TAG="$3"; shift 3
export PATH=/usr/local/cuda-12.4/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export MESHY_SGLANG_SM70=1
cd /data00/meshy/rl/meshy
OUT=/data00/meshy/rl/evals_offline
mkdir -p "$OUT"
PY=/data00/meshy/venv/bin/python
PORT=30031
for CK in "$@"; do
  name=$(basename "$CK")
  out="$OUT/${TAG}_${name}.jsonl"
  echo "=== eval $name proto=$PROTO samples=$SAMPLES -> $out ==="
  $PY scripts/eval_gsm8k.py --model "$CK" --n 200 --samples "$SAMPLES" \
      --max-new-tokens "$PROTO" --temperature 0.6 --top-p 0.95 --top-k 20 \
      --port "$PORT" --out "$out" 2>&1 | tail -3
done
echo "ALL_OFFLINE_EVALS_DONE $TAG"
