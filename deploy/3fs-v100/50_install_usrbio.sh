#!/usr/bin/env bash
# Install the cp312 hf3fs_py_usrbio wheel into the shared Meshy venv and make
# its non-bundled runtime libs importable without polluting the host.
#
# The wheel bundles libhf3fs_api_shared.so but dynamically links three Ubuntu
# libs (double-conversion/glog/dwarf) the veLinux host does not ship. We copy
# those from the 3fs runtime container into site-packages; SGLang's own doc
# then just needs site-packages on LD_LIBRARY_PATH.
set -euo pipefail

VENV=${MESHY_VENV:-/data00/meshy/venv}
ROOT=${FS3_ROOT:-/data00/meshy/fs3}
CONTAINER=3fs-node
UV=${UV_BIN:-/data00/home/chenkailun.c/.local/bin/uv}
SP=$("$VENV/bin/python" -c 'import site;print(site.getsitepackages()[0])')
WHL=$(ls "$ROOT"/3FS/dist/hf3fs_py_usrbio-*cp312*.whl | head -1)

"$UV" pip install --python "$VENV/bin/python" --no-deps --force-reinstall "$WHL"

for l in libdouble-conversion.so.3 libglog.so.0 libdwarf.so.1; do
  docker cp -L "$CONTAINER:/usr/lib/x86_64-linux-gnu/$l" "/tmp/$l"
  cp -L "/tmp/$l" "$SP/$l"
  rm -f "/tmp/$l"
done

# The extension's RUNPATH points at the build container (/3FS/build/...), so
# export the package dir at runtime. Put this in an activation fragment.
cat > "$VENV/../hf3fs_usrbio.env" <<EOF
# Source this, or add it to the service environment, to import hf3fs_fuse.io
export LD_LIBRARY_PATH=$SP:\$LD_LIBRARY_PATH
export SGLANG_HICACHE_HF3FS_CONFIG_PATH=$ROOT/hicache_hf3fs_config.json
EOF

echo "installed; run: source $VENV/../hf3fs_usrbio.env"
