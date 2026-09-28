#!/bin/bash
# Detached launcher for the formal RL run: sampler + RL process, both in a new
# session so ssh disconnect cannot kill them. Prints RUNTIME from rl.log.
LOGDIR=/data00/meshy/rl/logs
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$LOGDIR"
pkill -f sample_mem60.sh 2>/dev/null || true
sleep 1
setsid "$SCRIPT_DIR/v100_sample_mem60.sh" "$LOGDIR/rl_mem_$(date +%Y%m%d-%H%M%S).csv" \
  >/dev/null 2>&1 </dev/null &
setsid "$SCRIPT_DIR/v100_run_rl.sh" > "$LOGDIR/rl.log" 2>&1 </dev/null &
sleep 3
echo "launched:"; pgrep -af "v100_run_rl.sh|sample_mem60.sh" | grep -v pgrep
echo "--- rl.log ---"; sleep 1; head -8 "$LOGDIR/rl.log" 2>/dev/null
