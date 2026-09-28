"""Interpreter-startup hook for the SGLang sm70 compatibility patch.

Added to PYTHONPATH by ``SGLangService`` (and the standalone smoke launcher),
this runs in every Python process of the server tree — HTTP server and the
spawned scheduler/worker — before SGLang is imported. It is gated on the
``MESHY_SGLANG_SM70`` env var and on actually running on sm70, so normal GPU
runs and unrelated Python processes are untouched.
"""

import os


def _bootstrap() -> None:
    if os.environ.get("MESHY_SGLANG_SM70") != "1":
        return
    try:
        from meshy.backend.sglang_sm70 import apply_sm70_patch

        apply_sm70_patch()
    except Exception:  # never break interpreter startup; real error surfaces at serve
        import traceback

        traceback.print_exc()


_bootstrap()
