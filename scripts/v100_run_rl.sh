#!/bin/bash
# GRPO run on the single V100, decode CUDA graph on, runtime + weights on 3FS.
# Geometry is overridable for the large-batch/short-run experiment:
#   XRL_ROLLOUT_BATCH prompts/step * XRL_GROUP_SIZE completions = window samples
#   XRL_MINI_BATCH -> optimizer updates/window = batch / mini_batch
# Launched detached via v100_start_rl.sh.
set -euo pipefail

export PATH=/usr/local/cuda-12.4/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export MESHY_SGLANG_SM70=1
export MESHY_SM70_TILELANG=1
# Decode CUDA graph is the sm70 default; graph off only on OOM/manual fallback.
# Do NOT set MESHY_SM70_SAVER_MANAGES_GRAPH (unverified branch, main order).
: "${MESHY_SM70_CUDA_GRAPH:=1}"
export MESHY_SM70_CUDA_GRAPH

# Experiment geometry (defaults: the original 300x64 run).
: "${XRL_STEPS:=300}"
: "${XRL_ROLLOUT_BATCH:=8}"
: "${XRL_GROUP_SIZE:=8}"
: "${XRL_MINI_BATCH:=8}"
: "${XRL_EVAL_EVERY:=50}"
: "${XRL_RUN_TAG:=formal}"
export XRL_STEPS XRL_ROLLOUT_BATCH XRL_GROUP_SIZE XRL_MINI_BATCH XRL_EVAL_EVERY

cd /data00/meshy/rl/meshy
PY=/data00/meshy/venv/bin/python
export PYTHONPATH=/data00/meshy/rl/meshy

# One TQ storage unit (single card, single partition); the spec default 2
# wasted ~1.6 GiB on a 31 GiB host. TQ storage is dict-backed (measured RSS
# ~23 MiB), so prealloc needs no shrink even at a 512-sample window.
export XRL_TQ_STORAGE_UNITS=1
export XRL_RUNTIME_DIR=/3fs/stage/meshy/rollout/rl-${XRL_RUN_TAG}-$(date +%Y%m%d-%H%M%S)
echo "RUNTIME=$XRL_RUNTIME_DIR"
echo "COMMIT=$(git rev-parse HEAD 2>/dev/null || cat DEPLOY_COMMIT 2>/dev/null || cat /data00/meshy/rl/COMMIT 2>/dev/null)"
echo "TAG=$XRL_RUN_TAG STEPS=$XRL_STEPS ROLLOUT_BATCH=$XRL_ROLLOUT_BATCH GROUP=$XRL_GROUP_SIZE WINDOW=$((XRL_ROLLOUT_BATCH*XRL_GROUP_SIZE)) MINI_BATCH=$XRL_MINI_BATCH UPDATES/WIN=$((XRL_ROLLOUT_BATCH*XRL_GROUP_SIZE/XRL_MINI_BATCH)) EVAL_EVERY=$XRL_EVAL_EVERY"
echo "GRAPH=$MESHY_SM70_CUDA_GRAPH START=$(date +%s)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

if $PY scripts/launch.py --recipe recipe.grpo_gsm8k_v100; then
  echo "RL_DONE $(date +%s)"
else
  rc=$?
  echo "RL_FAIL rc=$rc $(date +%s)"
  exit $rc
fi
