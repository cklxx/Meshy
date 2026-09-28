#!/bin/bash
# 60s GPU + host-memory sampler for the formal run. Logs full `free -m`
# columns (main order: host had only 3/31 GiB free in smoke).
OUT="${1:?usage: sample_mem60.sh <out.csv>}"
echo "ts,gpu_mib,gpu_util,mem_total,mem_used,mem_free,mem_avail" > "$OUT"
while true; do
  g=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
      | sed 's/ //g;s/,/ /')
  set -- $(free -m | awk '/^Mem:/{print $2,$3,$4,$7}')
  echo "$(date +%s),$g,$1,$2,$3,$4" >> "$OUT"
  sleep 60
done
