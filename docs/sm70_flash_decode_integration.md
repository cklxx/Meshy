# T11i integration checklist: sm70 flash-decode into the RL tree

Source branch: `v100/kern`, based on `origin/v100/t11` (7e249b8).
The RL run uses `PYTHONPATH=/data00/meshy/rl/meshy`
(scripts/v100_run_rl.sh:43), so these files must land in that tree via
the T11i integration branch (merge, not rsync).

## Files and required changes

1. **`meshy/kernels/csrc/flash_decode.cu`** (new) — the hand-written
   warp-key CUDA kernel (half-warp/key, 16 B vector loads, shuffle QK,
   LSE split combine), `PYBIND11_MODULE` exporting
   `launch_flash_decode`. Compiled at runtime by
   `torch.utils.cpp_extension.load` with the CUDA 12.4 nvcc
   (`-std=c++17`, sm_70). First decode JITs it (~1 min) — add to the
   T11 pre-start checklist so window 1 is not mistaken for a hang.

2. **`meshy/kernels/flash_decode.py`** (new) — cpp_extension loader
   (`_lib()`), `choose_splits`/`choose_chunk`, `FlashDecodePlan`,
   `make_plan`, `run_with_plan` (zero host sync; row length read
   on-device from `kv_indptr`), and the eager `flash_decode_attention`
   wrapper. Constants `_HEAD_DIM=128`, group size derived from head
   counts.

3. **`meshy/backend/sglang_sm70.py`** (modified):
   * `bootstrap_pythonpath()` returns
     `repo_root + os.pathsep + boot_dir` (root FIRST). Without the root
     first, server child processes resolve `meshy` from the editable
     `.pth` instead of this tree and the patch silently never loads.
   * `flash_decode_enabled()`: env `MESHY_SM70_FLASH_DECODE == "1"`.
   * `_install_flash_decode_attention()`: wraps
     `sglang.kernels.ops.attention.decode_attention.decode_attention_fwd`;
     strict eligibility — decode only, fp16, head_dim 128,
     contiguous **3-D nhd** KV pool, `q.shape[0] <= MESHY_SM70_CUDA_GRAPH_MAX_BS`;
     non-contiguous c128 4-D strided views and eager overshoot fall back
     to triton.
   * call the installer from `apply_sm70_patch()` after the fused-op
     block, independent of `MESHY_SM70_TILELANG`.

No other modules change. The kernel replaces only the decode attention
call; prefill/extend/MLA paths are untouched.

## Enable in the run

Add one line next to `export MESHY_SM70_TILELANG=1` in
`scripts/v100_run_rl.sh`:

```sh
export MESHY_SM70_FLASH_DECODE=1
```

Optional: `MESHY_SM70_FLASH_DECODE_MAX_CTX` (default 4096) bounds the
capture-time split/chunk plan; graph buckets (1..64) need no change.

## Memory

All 12 per-bucket `FlashDecodePlan` workspaces total ~3.6 MB resident
(largest bucket bs64 = 0.52 MB); they live for the inference process
lifetime, do not touch the graph capture pool or KV layout, and are not
allocated during release/resume. See the colocate handoff test
(`scripts/sm70_flash_decode_colocate_mem.py`,
`docs/sm70_flash_decode_colocate_mem.json`).

## Post-merge verification (kern owns this)

On the integrated tree with `MESHY_SM70_FLASH_DECODE=1`:
1. run `scripts/sm70_flash_decode_e2e.py --only consistency` — eager
   and graph arms must report `sentinel_flash_called=true` AND
   `exact_match=true`; `flash_called=false` with exact_match=true is a
   false pass (the failure mode before the page_size/bootstrap fix).
2. after the first T11 window, confirm the scheduler resume log shows
   the flash plans rebuilt (search `flash-decode plan`), guarding the
   known torch_memory_saver resume OOM path (look at the
   `torch_memory_saver`/`csrc core.cpp` lines, not just the exit code).
