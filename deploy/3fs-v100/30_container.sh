#!/usr/bin/env bash
# Host-side lifecycle for the all-in-one 3FS container on v100.
# Run from /data00/meshy/fs3.
#
# privileged + host net -> sees rxe0 and 10.37.2.27, can create a FUSE mount.
# The host-facing stage is /data00/meshy/fs3/mnt, bind-mounted at /3fs with
# rshared propagation so the FUSE mount is visible on the host too.
set -euo pipefail

ROOT=/data00/meshy/fs3
RUN=$ROOT/run
NAME=3fs-node
IMAGE=3fs:dev
# Host-facing mount. The task requires the FUSE mount at host /3fs/stage; we
# bind host /3fs into the container at /3fs with rshared propagation so the
# FUSE mount created inside appears on the host at the same path.
HOST_FS=/3fs
HOST_STAGE=$HOST_FS/stage

init_dirs() {
  sudo mkdir -p "$HOST_STAGE" "$RUN/etc" "$RUN/log" \
           "$RUN/data-s1" "$RUN/data-s2" "$RUN/fdb/data"
  # Make the host dir a shared peer; without this the container's FUSE mount
  # does not propagate back to the host.
  sudo mount --bind "$HOST_FS" "$HOST_FS" 2>/dev/null || true
  sudo mount --make-rshared "$HOST_FS"
}

gen() {
  init_dirs
  cp "$ROOT/gen_configs.py" "$RUN/gen_configs.py"
  # stock templates come from the runtime image (/opt/3fs/configs); generate
  # against the checked-out source when the image is absent.
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker run --rm -v "$RUN:/run" "$IMAGE" \
      bash -c "FS3_RUN=/run FS3_TPL=/opt/3fs/configs python3 /run/gen_configs.py"
  else
    FS3_RUN="$RUN" FS3_TPL="$ROOT/3FS/configs" python3 "$RUN/gen_configs.py"
  fi
  # 20_start_cluster.sh launches fdbserver directly (fdbmonitor proved flaky
  # under this container), so no fdbmonitor conf needs to be staged.
  cp "$ROOT/20_start_cluster.sh" "$ROOT/provision_targets.sh" "$RUN/etc/"
  chmod +x "$RUN/etc/20_start_cluster.sh" "$RUN/etc/provision_targets.sh"
}

case "${1:-run}" in
  build) docker build --network=host -f "$ROOT/Dockerfile.runtime" -t "$IMAGE" "$ROOT" ;;
  init)  gen ;;
  run)
    if [ ! -f "$RUN/etc/admin_cli.toml" ]; then gen; fi
    init_dirs
    docker run -d --name "$NAME" --restart=no \
      --privileged --network=host --ipc=host \
      --device /dev/fuse -v /dev/infiniband:/dev/infiniband \
      -v "$RUN/etc:/opt/3fs/etc" \
      -v "$RUN/log:/opt/3fs/log" \
      -v "$RUN/data-s1:/opt/3fs/data/s1" \
      -v "$RUN/data-s2:/opt/3fs/data/s2" \
      -v "$RUN/fdb:/opt/3fs/fdb" \
      -v "$HOST_FS:/3fs:rshared" \
      "$IMAGE" sleep infinity
    docker exec "$NAME" bash /opt/3fs/etc/20_start_cluster.sh
    ;;
  sh)   docker exec -it "$NAME" bash ;;
  exec) shift; docker exec "$NAME" "$@" ;;
  stop) docker rm -f "$NAME" ;;
  status)
    docker ps --filter name="$NAME" --format '{{.Names}} {{.Status}}'
    mount | grep "$HOST_STAGE" || echo "host stage not mounted"
    ;;
  *) echo "usage: $0 {build|init|run|sh|exec|stop|status}" >&2; exit 2 ;;
esac
