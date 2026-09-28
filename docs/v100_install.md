# V100 (sm70) Installation

Meshy on a single Tesla V100-SXM2-32GB (sm_70, driver 535 / CUDA 12.2).
The generic [manual install](manual_install.md) targets CUDA 12.9 Hopper-class
GPUs; sm70 needs three changes: cu126 wheels (later wheels drop sm_70), fp16/fp32
instead of bf16, and CUDA graph disabled.

## Hardware constraints

- **sm70 has no bf16 tensor cores.** Use fp32 or fp16. Any bf16 matmul path is
  unsupported (wrong results or dtype errors).
- **No CUDA graph capture.** On sm70 capture fails inside SGLang's decode graph
  and the failed capture leaves the torch caching allocator poisoned
  (`captures_underway` assert on the next `empty_cache`). Run eager:
  `disable_cuda_graph=true` (also `disable_cuda_graph_padding=true`).
- **Triton attention backend.** FlashInfer kernels target newer archs; serve
  with `attention_backend=triton`.
- **nvcc 12.4** for any JIT compile (Triton/TileLang): default `/usr/bin/nvcc`
  is 11.8 and rejects `-std=c++20`. `export PATH=/usr/local/cuda-12.4/bin:$PATH`.

## Network: bypass the corp proxy

Measured 2026-09-28 from this host (`curl -4L`, same ~60MB `nvidia-curand-cu12`
wheel, 20s cap):

| source | via proxy | direct (`--noproxy '*'`) |
|---|---|---|
| download.pytorch.org/whl/cu126 | 0.95 MB/s | **3.03 MB/s** |
| mirrors.aliyun.com/pytorch-wheels/cu126 | 0.95 MB/s | 2.68 MB/s |

Direct egress is ~3x faster. All install commands below assume
`unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`.
`pypi.tuna`/`bytedpypi` were unreachable/unpopulated for cu126 wheels at test time.

## Steps

```bash
# Python 3.12 (GitHub releases are slow through the corp proxy; npmmirror hosts
# the python-build-standalone binaries):
export PATH=$HOME/.local/bin:$PATH
export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone
uv python install 3.12
uv venv --python 3.12 /data00/meshy/venv
PY=/data00/meshy/venv/bin/python
export UV_CACHE_DIR=/data00/meshy/uv-cache
PYPI=https://mirrors.aliyun.com/pypi/simple/

# 1. sglang first (pulls its pypi torch, overwritten in step 2)
uv pip install --python $PY --prerelease=allow sglang==0.5.18 --index-url $PYPI

# 2. cu126 torch (includes sm_70 in get_arch_list())
# NOTE: on this box direct egress is ~3x faster than the corp proxy.
# If installs stall, `unset http_proxy https_proxy` first.
uv pip install --python $PY \
    torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

# 3. sglang kernels (cu126 index). sglang-kernel pins a cu13 torch, so
#    step 2 MUST be re-run after this to pin torch back to cu126 — otherwise
#    torch becomes 2.13.0+cu130 (arch_list starts at sm_75, no sm_70).
uv pip install --python $PY sglang-kernel \
    --index-url https://docs.sglang.ai/whl/cu126/ --force-reinstall
uv pip install --python $PY \
    torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

# 4. torchtitan / TransferQueue / meshy
uv pip install --python $PY third_party/torchtitan-0.1.0.dev20260501+cu126-py3-none-any.whl --index-url $PYPI
uv pip install --python $PY TransferQueue==0.1.9 --index-url $PYPI --no-deps
uv pip install --python $PY -e . --index-url $PYPI
```

Verify arch support without initialising the device:

```bash
$PY -c "import torch; assert 'sm_70' in torch.cuda.get_arch_list(); print(torch.__version__, torch.version.cuda)"
```

## Serving on the V100 — BLOCKED upstream (sm75 floor)

SGLang 0.5.18 **does not start on sm70**, independent of flags. Three layers
each require sm75+: SGLang's own gate
(`maybe_downgrade_dtype_for_legacy_gpu`, "SGLang only supports sm75 and above"),
sglang-kernel (no sm70 wheel in any published version, no sdist), and
FlashInfer ("requires GPUs with sm75 or higher"). Downgrading SGLang does not
avoid it. The intended command once the kernel stack has an sm70 path (kern's
TileLang replacement) is:

```bash
$PY -m sglang.launch_server \
    --model-path /data00/meshy/models/Qwen3-0.6B \
    --disable-cuda-graph \
    --attention-backend triton \
    --dtype float16
```

Full tracebacks and the hot-path operator list: [v100_sm70_failures.md](v100_sm70_failures.md).

## What does run

- torchtitan Qwen3-0.6B one fwd+bwd step in fp32 and fp16: PASS.
- CPU pytest subset (17 files): 87 passed.

