#!/usr/bin/env bash
# Build the 3FS builder image (once) and the runtime image (after binaries exist).
# Run on the v100 host from /data00/meshy/fs3.
set -euo pipefail

ROOT=/data00/meshy/fs3
REPO=$ROOT/3FS
P=http://sys-proxy-rd-relay.byted.org:8118

build_builder() {
  docker build --network=host \
    --build-arg HTTP_PROXY=$P --build-arg HTTPS_PROXY=$P \
    --build-arg http_proxy=$P --build-arg https_proxy=$P \
    --build-arg NO_PROXY=localhost,127.0.0.1,.byted.org,byted.org \
    -f "$REPO/dockerfile/dev.dockerfile" -t 3fs-builder:dev "$REPO"
}

compile() {
  # Only the five binaries the single-node cluster needs; analytics/arrow is skipped.
  docker run --rm --network=host \
    -e HTTP_PROXY=$P -e HTTPS_PROXY=$P -e http_proxy=$P -e https_proxy=$P \
    -e NO_PROXY=localhost,127.0.0.1,.byted.org,byted.org \
    -v "$REPO:/3FS" -w /3FS 3fs-builder:dev bash -c '
      export PATH=/root/.cargo/bin:$PATH
      git config --global --add safe.directory /3FS
      cmake -S . -B build \
        -DCMAKE_CXX_COMPILER=clang++-14 -DCMAKE_C_COMPILER=clang-14 \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DSHUFFLE_METHOD=g++11
      cmake --build build -j8 --target mgmtd_main meta_main storage_main hf3fs_fuse_main admin_cli
    '
}

build_runtime() {
  docker build --network=host -f "$ROOT/Dockerfile.runtime" -t 3fs:dev "$ROOT"
}

# Build the hf3fs_py_usrbio cp312 wheel for SGLang HiCache. The builder image
# only has python 3.10, so the uv-managed cpython 3.12 (the one the shared
# venv is built from) is bind-mounted and used to create the build venv.
# Wheel lands in $REPO/dist/hf3fs_py_usrbio-*.whl.
build_usrbio() {
  local uvp=${UVPY_DIR:-/data00/home/chenkailun.c/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu}
  docker run --rm --network=host \
    -e HTTP_PROXY=$P -e HTTPS_PROXY=$P -e http_proxy=$P -e https_proxy=$P \
    -v "$REPO:/3FS" -w /3FS -v "$uvp:$uvp" 3fs-builder:dev bash -c "
      export PATH=/root/.cargo/bin:\$PATH
      git config --global --add safe.directory /3FS
      $uvp/bin/python3.12 -m venv /tmp/bv
      /tmp/bv/bin/pip install -q -U pip setuptools wheel
      CMAKE_ARGS=-DSHUFFLE_METHOD=g++11 CMAKE_BUILD_PARALLEL_LEVEL=8 \
        /tmp/bv/bin/python setup.py bdist_wheel -d /3FS/dist
    "
}

case "${1:-}" in
  builder)   build_builder ;;
  compile)   compile ;;
  runtime)   build_runtime ;;
  usrbio)    build_usrbio ;;
  all)       build_builder && compile && build_runtime ;;
  *) echo "usage: $0 {builder|compile|runtime|usrbio|all}" >&2; exit 2 ;;
esac
