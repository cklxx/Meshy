#!/bin/bash
# Formal GRPO run on the single V100: 300 steps, decode CUDA graph on,
# runtime + weights on 3FS. Launched detached via start_rl.sh.
set -euo pipefail

export PATH=/usr/local/cuda-12.4/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export MESHY_SGLANG_SM70=1
export MESHY_SM70_TILELANG=1
# Decode CUDA graph is the sm70 default; graph off only on OOM/manual fallback.
# Do NOT set MESHY_SM70_SAVER_MANAGES_GRAPH (unverified branch, main order).
: "${MESHY_SM70_CUDA_GRAPH:=1}"
export MESHY_SM70_CUDA_GRAPH

cd /data00/meshy/rl/meshy
PY=/data00/meshy/venv/bin/python
export PYTHONPATH=/data00/meshy/rl/meshy

export XRL_STEPS=300
# One TQ storage unit, not the spec default 2: single card, single data
# partition. Each unit held ~1.6 GiB RSS in smoke; host only had 3/31 GiB
# free, so this is the largest safe reclaim (~1.6 GiB).
export XRL_TQ_STORAGE_UNITS=1
export XRL_RUNTIME_DIR=/3fs/stage/meshy/rollout/rl-formal-$(date +%Y%m%d-%H%M%S)
echo "RUNTIME=$XRL_RUNTIME_DIR"
echo "COMMIT=$(git rev-parse HEAD 2>/dev/null || cat DEPLOY_COMMIT 2>/dev/null || cat /data00/meshy/rl/COMMIT 2>/dev/null)"
echo "GRAPH=$MESHY_SM70_CUDA_GRAPH START=$(date +%s)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

if $PY scripts/launch.py --recipe recipe.grpo_gsm8k_v100; then
  echo "RL_DONE $(date +%s)"
else
  rc=$?
  echo "RL_FAIL rc=$rc $(date +%s)"
  exit $rc
fi
