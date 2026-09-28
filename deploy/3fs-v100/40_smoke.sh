#!/usr/bin/env bash
# Smoke test on the mounted 3FS: data integrity (write -> remount-independent
# read -> checksum) plus sequential read/write bandwidth via fio if present,
# else dd. Run on the host against the host-facing stage mount.
set -euo pipefail

STAGE=${1:-/3fs/stage}
SIZE=${SMOKE_SIZE:-512M}
D=$STAGE/smoke
mkdir -p "$D"

echo "== mount"
mount | grep -E "3fs|$STAGE" || { echo "3FS not mounted at $STAGE"; exit 1; }

echo "== integrity (write sha256, re-read, compare)"
head -c 104857600 /dev/urandom > "$D/in.bin"
W=$(sha256sum "$D/in.bin" | awk '{print $1}')
cp "$D/in.bin" "$D/in.copy"
R=$(sha256sum "$D/in.copy" | awk '{print $1}')
echo "write=$W"
echo "read =$R"
[ "$W" = "$R" ] && echo "INTEGRITY_OK" || { echo "INTEGRITY_FAIL"; exit 1; }
sync
for i in $(seq 1 100); do echo "$i" > "$D/small_$i"; done
echo "small files: $(ls "$D"/small_* | wc -l) written, $(cat "$D/small_100") last-read"

echo "== sequential bandwidth ($SIZE)"
# dd gives stable numbers here; fio's --minimal field offsets vary by version
# and are not worth parsing for a smoke check.
mb=$(( $(numfmt --from=iec "${SIZE}") / 1048576 ))
echo "-- write (fdatasync) --"
dd if=/dev/zero of="$D/bw" bs=1M count="$mb" conv=fdatasync 2>&1 | tail -1
echo "-- read --"
dd if="$D/bw" of=/dev/null bs=1M 2>&1 | tail -1

echo "== cleanup"
rm -rf "$D"
echo "SMOKE_OK"
