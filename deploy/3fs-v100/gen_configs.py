#!/usr/bin/env python3
"""Generate single-node 3FS cluster configs from the stock templates.

Two storage services on one host need distinct listener ports and target
paths, and the stock templates size RDMA buffers / block caches for a fat
node (8 GB RDMA buffers + 16 GB KV caches per storage service), which would
exhaust the 31 GB host several times over. This script rewrites only the
keys that must change and leaves everything else stock.

Outputs into etc/:
  fdb.cluster, admin_cli.toml,
  mgmtd_main{,_app,_launcher}.toml, meta_main{,_app,_launcher}.toml,
  storage_main_s1.toml / storage_main_s2.toml (+ app/launcher),
  hf3fs_fuse_main{,_app,_launcher}.toml
"""
import os
import re
import sys

ROOT = os.environ.get("FS3_RUN", "/data00/meshy/fs3/run")
TPL = os.environ.get("FS3_TPL", "/opt/3fs/configs")
ETC = os.path.join(ROOT, "etc")
CLUSTER_ID = "stage"
HOST = os.environ.get("FS3_HOST", "10.37.2.27")
MGMTD_ADDR = f"RDMA://{HOST}:8000"

# (key, new value) replacements applied to every storage service config.
STORAGE_COMMON = {
    # libaio instead of io_uring (ramdisk/loopback friendly, smaller).
    "enable_io_uring": "false",
    # worker threads: stock 32 is sized for 16 NVMe; this box has 8 cores.
    "num_threads": "2",        # aio + write worker (all bare num_threads keys)
    "bg_num_threads": "2",
    "num_channels": "64",
    # Both instances live in directories on the same root filesystem; allow
    # targets that do not carry their own disk UUID.
    "allow_disk_without_uuid": "true",
    # RDMA buffer pool: stock 1024*4MB + 64*64MB = 8 GB -> ~320 MB.
    "rdmabuf_count": "16",
    "big_rdmabuf_count": "4",
    # KV block caches: stock 8 GB each -> 64 MB each.
    "leveldb_block_cache_size": "'64MB'",
    "rocksdb_block_cache_size": "'64MB'",
    "leveldb_write_buffer_size": "'4MB'",
    "rocksdb_write_buffer_size": "'4MB'",
    # chunk meta-store preallocation.
    "allocate_size": "'64MB'",
    # event trace logs to cwd by default; disable to avoid stray writes.
    "enabled": "false",
    # no monitor collector in the single-node cluster.
    "remote_ip": '""',
}

# Per-instance: node id, listener ports, target path.
STORAGE_INSTANCES = [
    dict(suffix="s1", node=10001, rdma_port=8002, tcp_port=9002,
         target="/opt/3fs/data/s1"),
    dict(suffix="s2", node=10002, rdma_port=8003, tcp_port=9003,
         target="/opt/3fs/data/s2"),
]


def read(path):
    with open(path) as f:
        return f.read()


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def set_scalar(text, key, value, occurrence="all"):
    """Replace `key = ...` assignments, optionally only the nth occurrence."""
    pat = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*=\s*).*$")
    matches = list(pat.finditer(text))
    if not matches:
        raise KeyError(f"key {key} not found")
    if occurrence == "all":
        return pat.sub(lambda m: m.group(1) + value, text)
    # single occurrence by index (1-based), rebuild string
    idx = occurrence - 1
    m = matches[idx]
    return text[: m.start()] + m.group(1) + value + text[m.end():]


def set_mgmtd_addr(text):
    """Fill every bare mgmtd_server_addresses = [] in a service cfg.

    With --cfg the service loads this whole file as its template (no remote
    set-config), so all embedded client sections need the address, not just
    the launcher config.
    """
    return re.sub(
        r"(?m)^(\s*mgmtd_server_addresses\s*=\s*)\[\s*\]",
        lambda m: m.group(1) + f'["{MGMTD_ADDR}"]',
        text,
    )


def gen_storage(inst):
    cfg = read(os.path.join(TPL, "storage_main.toml"))
    # listener ports: stock has exactly two listen_port lines (8000 RDMA,
    # 9000 TCP). Replace in document order.
    cfg = set_scalar(cfg, "listen_port", str(inst["rdma_port"]), 1)
    cfg = set_scalar(cfg, "listen_port", str(inst["tcp_port"]), 2)
    # target path list -> this instance's single directory (unique bare key).
    cfg = re.sub(
        r"(?m)^(\s*target_paths\s*=\s*)\[.*$",
        lambda m: m.group(1) + f'["{inst["target"]}"]',
        cfg,
        count=1,
    )
    # scoped worker-thread reductions before the blanket num_threads swap:
    # write_worker.num_threads and aio num_threads are separate keys but share
    # the bare name `num_threads`; all stock values here are 32/8 -> 2 is fine.
    for k, v in STORAGE_COMMON.items():
        cfg = set_scalar(cfg, k, v)
    cfg = set_mgmtd_addr(cfg)
    write(os.path.join(ETC, f"storage_main_{inst['suffix']}.toml"), cfg)

    app = read(os.path.join(TPL, "storage_main_app.toml"))
    app = set_scalar(app, "node_id", str(inst["node"]))
    write(os.path.join(ETC, f"storage_main_{inst['suffix']}_app.toml"), app)

    launcher = read(os.path.join(TPL, "storage_main_launcher.toml"))
    launcher = set_scalar(launcher, "cluster_id", f'"{CLUSTER_ID}"')
    launcher = re.sub(
        r'(?m)^mgmtd_server_addresses\s*=\s*\[.*\]',
        f'mgmtd_server_addresses = ["{MGMTD_ADDR}"]',
        launcher,
    )
    write(os.path.join(ETC, f"storage_main_{inst['suffix']}_launcher.toml"), launcher)


def main():
    os.makedirs(ETC, exist_ok=True)

    # fdb.cluster: the runtime container pins this exact contents. The key
    # (after "stage:") must be hex; an underscore makes fdbcli reject it.
    write(os.path.join(ETC, "fdb.cluster"),
          "stage:3f510ca1@127.0.0.1:4500\n")

    # admin_cli
    admin = read(os.path.join(TPL, "admin_cli.toml"))
    admin = set_scalar(admin, "cluster_id", f'"{CLUSTER_ID}"')
    admin = re.sub(r"(?m)^(\s*clusterFile\s*=\s*).*$",
                   r"\1'/etc/foundationdb/fdb.cluster'", admin)
    write(os.path.join(ETC, "admin_cli.toml"), admin)

    # mgmtd
    mgmtd = read(os.path.join(TPL, "mgmtd_main.toml"))
    mgmtd = set_scalar(mgmtd, "remote_ip", '""')
    write(os.path.join(ETC, "mgmtd_main.toml"), mgmtd)
    mapp = read(os.path.join(TPL, "mgmtd_main_app.toml"))
    mapp = set_scalar(mapp, "node_id", "1")
    write(os.path.join(ETC, "mgmtd_main_app.toml"), mapp)
    ml = read(os.path.join(TPL, "mgmtd_main_launcher.toml"))
    ml = set_scalar(ml, "cluster_id", f'"{CLUSTER_ID}"')
    # Both [fdb] and [kv_engine.fdb] carry a clusterFile key; point both at
    # the container's cluster file. Plain line replace, no DOTALL.
    ml = re.sub(r"(?m)^(\s*clusterFile\s*=\s*).*$",
                r"\1'/etc/foundationdb/fdb.cluster'", ml)
    write(os.path.join(ETC, "mgmtd_main_launcher.toml"), ml)

    # meta
    meta = read(os.path.join(TPL, "meta_main.toml"))
    meta = set_scalar(meta, "remote_ip", '""')
    meta = re.sub(
        r'(?m)^mgmtd_server_addresses\s*=\s*\[.*\]',
        f'mgmtd_server_addresses = ["{MGMTD_ADDR}"]',
        meta,
    )
    write(os.path.join(ETC, "meta_main.toml"), meta)
    eapp = read(os.path.join(TPL, "meta_main_app.toml"))
    eapp = set_scalar(eapp, "node_id", "100")
    write(os.path.join(ETC, "meta_main_app.toml"), eapp)
    el = read(os.path.join(TPL, "meta_main_launcher.toml"))
    el = set_scalar(el, "cluster_id", f'"{CLUSTER_ID}"')
    el = re.sub(
        r'(?m)^mgmtd_server_addresses\s*=\s*\[.*\]',
        f'mgmtd_server_addresses = ["{MGMTD_ADDR}"]',
        el,
    )
    write(os.path.join(ETC, "meta_main_launcher.toml"), el)

    # storage x2
    for inst in STORAGE_INSTANCES:
        gen_storage(inst)

    # fuse: the stock app file is empty; cluster/mountpoint/token live in the
    # launcher config only. The service toml needs the mgmtd address embedded
    # for --cfg mode.
    fuse = read(os.path.join(TPL, "hf3fs_fuse_main.toml"))
    fuse = set_mgmtd_addr(fuse)
    write(os.path.join(ETC, "hf3fs_fuse_main.toml"), fuse)
    write(os.path.join(ETC, "hf3fs_fuse_main_app.toml"), "\n")
    fl = read(os.path.join(TPL, "hf3fs_fuse_main_launcher.toml"))
    fl = set_scalar(fl, "cluster_id", f'"{CLUSTER_ID}"')
    fl = set_scalar(fl, "mountpoint", "'/3fs/stage'")
    fl = set_scalar(fl, "token_file", "'/opt/3fs/etc/token.txt'")
    fl = re.sub(
        r'(?m)^mgmtd_server_addresses\s*=\s*\[.*\]',
        f'mgmtd_server_addresses = ["{MGMTD_ADDR}"]',
        fl,
    )
    write(os.path.join(ETC, "hf3fs_fuse_main_launcher.toml"), fl)

    print(f"generated configs in {ETC}")


if __name__ == "__main__":
    sys.exit(main())
