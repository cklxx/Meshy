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

Attention and sampling are selected separately with server flags
(``--attention-backend torch_native|triton --sampling-backend pytorch``) and
CUDA graph is disabled; those are CLI choices, not monkeypatches.

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


def apply_sm70_patch() -> bool:
    """Apply all sm70 compatibility patches. Returns True if applied."""
    global APPLIED
    if APPLIED:
        return True
    if not _is_sm70():
        return False

    _install_common_ops_stub()
    _patch_gate()
    _force_native_fused_ops()
    APPLIED = True
    logger.info("[sglang-sm70] applied sm70 compatibility patch")
    return True


def sm70_server_defaults() -> dict:
    """SGLang server flags required on sm70 (caller values win).

    Pure-torch fused ops, SDPA attention (triton also selectable), pytorch
    sampling, eager decode/prefill (sm70 graph capture poisons the allocator).
    """
    return {
        "dtype": "float16",
        "attention_backend": "torch_native",
        "sampling_backend": "pytorch",
        "cuda_graph_backend_decode": "disabled",
        "cuda_graph_backend_prefill": "disabled",
    }


def bootstrap_pythonpath() -> str | None:
    """Directory containing the sm70 sitecustomize, or None."""
    import os

    return os.path.join(os.path.dirname(__file__), "_sm70bootstrap")

