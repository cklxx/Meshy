"""Runtime patch that lets SGLang 0.5.18 serve on NVIDIA sm70 (V100).

Upstream SGLang and its prebuilt kernel stack floor at sm75:

* ``load_model_utils.maybe_downgrade_dtype_for_legacy_gpu`` raises
  "SGLang only supports sm75 and above." on capability minor < 5;
* ``sglang-kernel`` ships only sm90/sm100 binaries (no sm70 wheel, no sdist),
  so ``sgl_kernel`` cannot load ``common_ops``;
* FlashInfer's JIT also requires sm75+.

This patch is applied inside the SGLang server process tree (see
``meshy.backend.sglang_sm70_sitecustomize``) and changes **no files in
site-packages**. It does three things on sm70:

1. Replace the sm75 gate with the dtype-only downgrade (fp16) it performs on
   sm70..sm79, without raising.
2. Satisfy ``sgl_kernel``'s loader by offering a lazy ``common_ops`` stub via
   ``sys.modules``, so ``import sgl_kernel`` and its Python submodules load.
   The stub is never expected to execute: every hot-path CustomOp is forced to
   its pure-torch implementation below.
3. Force every ``BaseFusedOp`` (RMSNorm, fused_add_rmsnorm, SiLUAndMul, RoPE,
   ...) onto ``forward_native`` via SGLang's own global backend switch.

Attention, sampling and CUDA graph are selected with server flags
(``--attention-backend triton --sampling-backend pytorch``, decode graph on by
default, see :func:`sm70_server_defaults`); those are CLI choices, not
monkeypatches.

A native op must never silently run a stubbed kernel. If any code reaches a
``torch.ops.sgl_kernel.*`` call, the stub raises a clear error naming the op so
the gap is recorded for the TileLang replacement.
"""

from __future__ import annotations

import logging
import sys
import types

logger = logging.getLogger("meshy.sglang_sm70")

APPLIED = False


def _is_sm70() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (7, 0)
    except Exception:
        return False


def _preload_bundled_cudart() -> None:
    """Force the torch-bundled CUDA 12.6 runtime globally before SGLang imports.

    ``sgl_kernel.load_utils._preload_cuda_library`` dlopens the CUDA-home
    runtime (system 12.4 on this box) with RTLD_GLOBAL. That library predates
    ``cudaGetDriverEntryPointByVersion``; when torch's libc10_cuda then binds
    against it, import fails with
    ``undefined symbol: cudaGetDriverEntryPointByVersion, version libcudart.so.12``.
    Loading the torch wheel's own 12.6 runtime first makes every later dlopen
    of the same SONAME reuse it instead of the system one.
    """
    import ctypes
    import glob
    import os
    import sys

    candidates = sorted(
        glob.glob(
            os.path.join(
                sys.prefix,
                "lib",
                "python*",
                "site-packages",
                "nvidia",
                "cuda_runtime",
                "lib",
                "libcudart.so.12*",
            )
        )
    )
    if not candidates:
        return
    # prefer the plain SONAME then the newest versioned file
    plain = [c for c in candidates if os.path.basename(c) == "libcudart.so.12"]
    target = (plain or candidates)[0]
    ctypes.CDLL(target, mode=ctypes.RTLD_GLOBAL)


class _StubCommonOps:
    """Stand-in for sgl_kernel's compiled ``common_ops`` shared object.

    The real library is not imported for execution; fused ops run native. Any
    direct attribute access (the loader only logs ``__file__``) yields a
    callable that raises so a missing kernel is loud, never silent.
    """

    __file__ = "<meshy sm70 common_ops stub>"

    def __getattr__(self, name: str):
        def _missing(*args, **kwargs):
            raise RuntimeError(
                f"[sglang-sm70] sgl_kernel native op '{name}' is unavailable on "
                "sm70 and was called directly. This op needs a native/TileLang "
                "implementation; fused CustomOps must be forced to forward_native."
            )

        return _missing


def _install_common_ops_stub() -> None:
    # sgl_kernel.load_utils, on sm70, ends with `import common_ops`. Pre-register
    # a stub module so that import resolves from sys.modules without a real .so.
    if "common_ops" not in sys.modules:
        stub = types.ModuleType("common_ops")
        stub.__file__ = _StubCommonOps.__file__

        def _unexpected_attr(name: str):
            raise RuntimeError(
                f"[sglang-sm70] common_ops.{name} unavailable on sm70 (stub)."
            )

        stub.__getattr__ = _unexpected_attr  # type: ignore[attr-defined]
        sys.modules["common_ops"] = stub


def _patch_gate() -> None:
    from sglang.srt.model_executor.model_runner_components import load_model_utils

    if getattr(load_model_utils, "_meshy_sm70_patched", False):
        return

    def _downgrade_dtype_only(*, server_args, model_config) -> None:
        import torch

        if torch.cuda.get_device_capability()[0] < 8:
            logger.info(
                "[sglang-sm70] capability < sm80: using float16 (gate bypassed for sm70)"
            )
            try:
                from sglang.srt.runtime_context import get_context

                get_context().override("ModelRunner._sm80_dtype_fallback", dtype="float16")
            except Exception:
                pass
            model_config.dtype = torch.float16

    load_model_utils.maybe_downgrade_dtype_for_legacy_gpu = _downgrade_dtype_only
    load_model_utils._meshy_sm70_patched = True  # type: ignore[attr-defined]

    # model_runner imported the name directly, patch it there too.
    try:
        from sglang.srt.model_executor import model_runner

        model_runner.maybe_downgrade_dtype_for_legacy_gpu = _downgrade_dtype_only
    except Exception:
        pass


def _force_native_fused_ops() -> None:
    from sglang.kernels.fused_op import set_fused_op_backend
    from sglang.kernels.spec import KernelBackend

    set_fused_op_backend(KernelBackend.TORCH)
    import os

    os.environ.setdefault("SGLANG_FORCE_FUSED_OP_BACKEND", "native")


def tilelang_enabled() -> bool:
    """Whether the TileLang fused-op path is requested (MESHY_SM70_TILELANG=1)."""
    import os

    return os.environ.get("MESHY_SM70_TILELANG", "0") == "1"


def _install_tilelang_fused_ops(*, prewarm: bool = True) -> None:
    """Redirect RMSNorm / fused_add_rmsnorm / SiLUAndMul to meshy.kernels.

    The sm70 patch forces every BaseFusedOp onto ``KernelBackend.TORCH``, so
    dispatch calls ``forward_native`` directly — that is the single chokepoint
    to override (``forward_cuda`` is never reached under the force). We wrap
    the three classes' ``forward_native``; above the largest compiled row
    bucket the wrapper falls back to the original native implementation, so
    a long prefill chunk can never crash serve.
    """
    import os

    import torch

    import sglang.srt.layers.layernorm as _ln
    import sglang.srt.layers.activation as _act
    from meshy import kernels as _tl

    def _wrap_rmsnorm(cls):
        orig = cls.forward_native

        def forward_native(self, x, residual=None, post_residual_addition=None):
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            eps = float(self.variance_epsilon)
            if self.variance_size_override is not None:
                return orig(self, x, residual, post_residual_addition)
            # Kernels are fp16-only; rl_on_policy_target uses fp32 weights /
            # override_orig_dtype and must stay on the native path.
            if self.weight.data.dtype != torch.float16 or x.dtype != torch.float16:
                return orig(self, x, residual, post_residual_addition)
            if residual is None:
                try:
                    out = _tl.rmsnorm(x, self.weight.data, eps)
                except ValueError:
                    return orig(self, x)
                if x.dim() != 2:
                    out = out.reshape(x.shape)
                return out
            if residual.dtype != torch.float16:
                return orig(self, x, residual, post_residual_addition)
            try:
                _tl.fused_add_rmsnorm(
                    x, residual, self.weight.data, eps
                )
            except ValueError:
                return orig(self, x, residual)
            return x, residual

        cls.forward_native = forward_native  # type: ignore[assignment]
        return cls

    def _wrap_silu(cls):
        orig = cls.forward_native

        def forward_native(self, x):
            try:
                return _tl.silu_and_mul(x)
            except ValueError:
                return orig(self, x)

        cls.forward_native = forward_native  # type: ignore[assignment]
        return cls

    _wrap_rmsnorm(_ln.RMSNorm)
    _wrap_silu(_act.SiluAndMul)

    if prewarm:
        # Qwen3-0.6B: hidden 1024 (RMSNorm) and SwiGLU half-width 3072.
        # Compiling every row bucket once here means /health_generate never
        # lets a first request pay JIT latency.
        norm_widths = [int(v) for v in os.environ.get(
            "MESHY_SM70_TILELANG_NORM_WIDTHS", "1024"
        ).split(",") if v]
        silu_halves = [int(v) for v in os.environ.get(
            "MESHY_SM70_TILELANG_SILU_HALVES", "3072"
        ).split(",") if v]
        compiled = _tl.prewarm(
            norm_widths=norm_widths, silu_halves=silu_halves
        )
        logger.info(
            "[sglang-sm70] TileLang fused ops installed + prewarmed: "
            f"{len(compiled)} kernels (norm {norm_widths}, silu {silu_halves})"
        )
    else:
        logger.info("[sglang-sm70] TileLang fused ops installed (prewarm off)")


def apply_sm70_patch() -> bool:
    """Apply all sm70 compatibility patches. Returns True if applied."""
    global APPLIED
    if APPLIED:
        return True

    # Must precede the first torch import so libc10_cuda binds CUDA 12.6 runtime.
    _preload_bundled_cudart()
    if not _is_sm70():
        return False

    _install_common_ops_stub()
    _patch_gate()
    _force_native_fused_ops()
    if tilelang_enabled():
        # Prewarm happens here, synchronously, before SGLang imports the
        # model — keeping first-request latency free of TileLang JIT.
        _install_tilelang_fused_ops(prewarm=True)
    APPLIED = True
    logger.info(
        "[sglang-sm70] applied sm70 compatibility patch "
        f"(TileLang fused ops: {'on' if tilelang_enabled() else 'off'})"
    )
    return True


def sm70_server_defaults(*, cuda_graph: bool | None = None, max_bs: int | None = None) -> dict:
    """SGLang server flags required on sm70 (caller values win).

    fp16, triton attention (the one backend that does not force-disable CUDA
    graph), pytorch sampling. CUDA graph is enabled for decode by default up to
    ``MESHY_SM70_CUDA_GRAPH_MAX_BS`` (64): sm70 capture has to be verified at
    runtime and prefill graph stays off (variable-shape prefill is where the
    tileRL capture-poisoning failure occurred). Set MESHY_SM70_CUDA_GRAPH=0 to
    force eager for A/B timing. Memory saver keeps a host weight backup so
    release/resume restores real weights.

    Explicit ``cuda_graph`` / ``max_bs`` kwargs win over the environment, so a
    parent building CLI args for a child need not mutate its own os.environ
    (mutating only the child env previously made every A/B server come up
    graph-on, because defaults were read in the parent).
    """
    import os

    if cuda_graph is None:
        cuda_graph = os.environ.get("MESHY_SM70_CUDA_GRAPH", "1") != "0"
    if max_bs is None:
        max_bs = int(os.environ.get("MESHY_SM70_CUDA_GRAPH_MAX_BS", "64"))
    graph_on = cuda_graph
    defaults = {
        "dtype": "float16",
        # triton is both the fastest measured backend and graph-compatible;
        # torch_native forces graph disabled upstream.
        "attention_backend": "triton",
        "sampling_backend": "pytorch",
        "enable_weights_cpu_backup": True,
    }
    if graph_on:
        # Keep SGLang's default padded capture-bs bucket list (1,2,4,..,max_bs):
        # that is ~12 graphs up to bs=64 and fits the ~3 GB capture budget.
        # Do NOT set --disable-cuda-graph-padding, which switches to a
        # per-concrete-bs list (1..64 = 64 graphs) and OOMs the capture pool.
        defaults.update(
            {
                "cuda_graph_backend_decode": "full",
                "cuda_graph_backend_prefill": "disabled",
                "cuda_graph_max_bs_decode": max_bs,
            }
        )
    else:
        # disable_cuda_graph (deprecated, == backend decode+prefill disabled)
        # is the explicit, greppable eager switch; pass it as a store_true flag.
        defaults.update(
            {
                "disable_cuda_graph": True,
                "cuda_graph_backend_decode": "disabled",
                "cuda_graph_backend_prefill": "disabled",
            }
        )
    return defaults


def bootstrap_pythonpath() -> str | None:
    """Directory containing the sm70 sitecustomize, or None."""
    import os

    return os.path.join(os.path.dirname(__file__), "_sm70bootstrap")

