#!/usr/bin/env bash
# Provision storage targets + chain table for the two co-located services.
#
# A file with strip size 16 needs 16 distinct chains (allocating 16 chains
# from a 1-chain table fails with InvalidFileLayout "found 1"), and every
# chain needs distinct target ids. We therefore create 16 targets per node
# (disk 0, target index 0..15) and 16 RF=2 chains pairing node 10001 and
# 10002. IDs follow gen_chain_table.py's formulas (target prefix 1, chain
# prefix 9, disk_index 0):
#   target(node,i) = ((1_000_000 + node)*1000 + 1)*100 + (i+1)
#   chain(i)       = (9*1000 + 1)*100_000 + (i+1)
set -uo pipefail

ETC=/opt/3fs/etc
BIN=/opt/3fs/bin
N=${STRIPE_CHAINS:-16}
TOKEN=$(cat "$ETC/token.txt")

acl() {
  "$BIN/admin_cli" -cfg "$ETC/admin_cli.toml" \
    --config.mgmtd_client.mgmtd_server_addresses "[\"RDMA://10.37.2.27:8000\"]" \
    --config.user_info.token "$TOKEN" "$@"
}

echo ChainId,TargetId,TargetId > "$ETC/generated_chains.csv"
echo ChainId > "$ETC/generated_chain_table.csv"
for i in $(seq 0 $((N-1))); do
  t1=$(( 1010001001 * 100 + i + 1 ))   # node 10001, disk 0
  t2=$(( 1010002001 * 100 + i + 1 ))   # node 10002, disk 0
  cid=$(( 900100000 + i + 1 ))
  acl "create-target --node-id 10001 --disk-index 0 --target-id $t1 \
       --chain-id $cid --use-new-chunk-engine" 2>&1 | tail -1
  acl "create-target --node-id 10002 --disk-index 0 --target-id $t2 \
       --chain-id $cid --use-new-chunk-engine" 2>&1 | tail -1
  echo "$cid,$t1,$t2" >> "$ETC/generated_chains.csv"
  echo "$cid" >> "$ETC/generated_chain_table.csv"
done

acl "upload-chains $ETC/generated_chains.csv" 2>&1 | tail -1
acl "upload-chain-table --desc stage 1 $ETC/generated_chain_table.csv" 2>&1 | tail -1
echo "chain-tables:"; acl "list-chain-tables" 2>&1 | tail -4
