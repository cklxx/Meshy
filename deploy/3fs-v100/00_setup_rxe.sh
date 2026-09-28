#!/usr/bin/env bash
# Create a soft-RoCE (rxe) device on eth0, persistent for the current boot.
# Idempotent: re-adding rxe0 after it already exists is reported and skipped.
set -euo pipefail

DEV=${RXE_NETDEV:-eth0}
NAME=${RXE_NAME:-rxe0}

sudo modprobe rdma_rxe
if rdma link show "$NAME" >/dev/null 2>&1; then
  echo "$NAME already exists:"
  rdma link show "$NAME"
else
  sudo rdma link add "$NAME" type rxe netdev "$DEV"
  echo "created $NAME on $DEV"
fi
rdma link show "$NAME"
