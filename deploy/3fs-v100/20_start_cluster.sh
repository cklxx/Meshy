#!/usr/bin/env bash
# Start the single-node 3FS cluster inside the 3fs-node container and bring up
# the FUSE mount. Every service loads a full local config via --cfg (the two
# co-located storage services need distinct ports/target paths, so they cannot
# share one mgmtd STORAGE template); they still register with mgmtd.
#
# Order: FDB -> mgmtd -> init cluster (once) -> meta -> storage x2 ->
# user/targets/chains -> FUSE.
set -euo pipefail

ETC=/opt/3fs/etc
LOG=/opt/3fs/log
BIN=/opt/3fs/bin
HOST=10.37.2.27
M="RDMA://$HOST:8000"
# FDB cluster key must be hex; "stage" is the 3FS cluster_id, unrelated.
FDB_CLUSTER="stage:3f510ca1@127.0.0.1:4500"

mkdir -p "$LOG" /3fs/stage /opt/3fs/data/s1 /opt/3fs/data/s2 \
         /opt/3fs/fdb/data /var/log/3fs

log() { echo "[3fs-start] $*"; }

fix_hosts() {
  # Under --network=host the container keeps the host's hostname. If it
  # resolves only to IPv6 (AAAA via the corporate resolver), the services
  # advertise an empty IPv4 routing address and the client connects to
  # TCP://0.0.0.0:0. Pin the name to the box's IPv4 before anything starts.
  local hn ip
  hn=$(hostname)
  ip=$HOST
  if ! getent ahostsv4 "$hn" >/dev/null 2>&1; then
    echo "$ip $hn" >> /etc/hosts
    log "pinned $hn -> $ip in /etc/hosts"
  fi
}

fix_hosts

start_fdb() {
  if pgrep -x fdbserver >/dev/null; then log "fdb already running"; return; fi
  echo "$FDB_CLUSTER" > /etc/foundationdb/fdb.cluster
  # Run fdbserver directly. Under fdbmonitor inside this container it bound to
  # the container-hostname address and never listened; binding 127.0.0.1
  # explicitly is all a single-node cluster needs.
  nohup /usr/sbin/fdbserver \
    -p 127.0.0.1:4500 -C /etc/foundationdb/fdb.cluster \
    -d /opt/3fs/fdb/data -L "$LOG" \
    --memory 1073741824 > "$LOG/fdbserver.out" 2>&1 &
  for _ in $(seq 1 30); do
    fdbcli --exec "status minimal" >/dev/null 2>&1 && break
    sleep 1
  done
  # one-process cluster; harmless once it already exists
  fdbcli --exec "configure new single ssd" >/dev/null 2>&1 || true
  log "fdb up"
}

start_daemon() {
  local name=$1; shift
  if [ -f "$LOG/$name.pid" ] && kill -0 "$(cat "$LOG/$name.pid")" 2>/dev/null; then
    log "$name already running"; return
  fi
  log "starting $name"
  nohup "$@" > "$LOG/$name.out" 2>&1 &
  echo $! > "$LOG/$name.pid"
}

svc() { # name bin launcher app cfg
  start_daemon "$1" "$BIN/$2" \
    --launcher_cfg "$ETC/$3" --app-cfg "$ETC/$4" --cfg "$ETC/$5"
}

# admin_cli talking to the local mgmtd. The TOML array value must quote its
# string element, so it is ["RDMA://10.37.2.27:8000"] on the command line.
acl() {
  "$BIN/admin_cli" -cfg "$ETC/admin_cli.toml" \
    --config.mgmtd_client.mgmtd_server_addresses "[\"$M\"]" "$@"
}

start_fdb

# init-cluster writes directly to FDB; per the upstream guide it runs before
# mgmtd starts. Idempotent enough to gate on the same provisioned marker.
if [ ! -f "$ETC/.provisioned" ]; then
  "$BIN/admin_cli" -cfg "$ETC/admin_cli.toml" \
    "init-cluster --mgmtd $ETC/mgmtd_main.toml 1 1048576 16"
fi

svc mgmtd mgmtd_main mgmtd_main_launcher.toml mgmtd_main_app.toml mgmtd_main.toml
sleep 3

if [ ! -f "$ETC/.provisioned" ]; then
  acl "user-add --root --admin 0 root" | tee "$ETC/token.raw"
  # token table row: "Token  <token>(Expired at N/A)"
  sed -n 's/^Token[[:space:]]\+\([A-Za-z0-9+/=]\+\)(.*/\1/p' "$ETC/token.raw" \
    | head -1 > "$ETC/token.txt"
  log "admin token: $(cat "$ETC/token.txt")"
fi

svc meta meta_main meta_main_launcher.toml meta_main_app.toml meta_main.toml
sleep 2
svc storage_s1 storage_main storage_main_s1_launcher.toml storage_main_s1_app.toml storage_main_s1.toml
svc storage_s2 storage_main storage_main_s2_launcher.toml storage_main_s2_app.toml storage_main_s2.toml

# wait for both storage nodes to register before allocating their targets
if [ ! -f "$ETC/.provisioned" ]; then
  for _ in $(seq 1 30); do
    out=$(acl "list-nodes" 2>/dev/null)
    echo "$out" | grep -q 10001 && echo "$out" | grep -q 10002 && break
    sleep 2
  done
  bash "$ETC/provision_targets.sh"
  # New targets take a few heartbeats to reach SERVING/UPTODATE; wait so the
  # first file create does not race chain availability.
  for _ in $(seq 1 30); do
    n=$(acl "list-targets" 2>/dev/null | grep -c "SERVING.*UPTODATE")
    [ "$n" -ge $((16*2)) ] && break
    sleep 2
  done
  touch "$ETC/.provisioned"
fi

if mountpoint -q /3fs/stage; then
  log "fuse already mounted"
else
  svc fuse hf3fs_fuse_main hf3fs_fuse_main_launcher.toml hf3fs_fuse_main_app.toml hf3fs_fuse_main.toml
  sleep 3
fi

# Mount root is root-owned 0755 and 3FS refuses chmod on it. Expose the
# Meshy job directories owned by the training user (host uid 1001), so the
# trainer/SGLang can write without root. Set MESHY_UID to override.
MESHY_UID=${MESHY_UID:-1001}
MESHY_GID=${MESHY_GID:-1001}
mkdir -p /3fs/stage/meshy/kvcache /3fs/stage/meshy/ckpt /3fs/stage/meshy/rollout
chown -R "$MESHY_UID:$MESHY_GID" /3fs/stage/meshy
# One world-writable scratch dir for non-Meshy smoke tests/host users.
mkdir -p /3fs/stage/data && chmod 777 /3fs/stage/data

log "nodes:"
acl "list-nodes" || true
log "mount:"
mount | grep hf3fs || true
