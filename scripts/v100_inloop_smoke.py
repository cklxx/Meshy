#!/usr/bin/env python3
"""5-question smoke for the in-loop eval hook against a standalone SGLang.

Exercises the exact production path (tokenize -> input_ids -> engine.generate
-> lenient scoring -> jsonl/summary), which the text-based eval_gsm8k.py does
not. Env: XRL_EVAL_N=5 XRL_EVAL_EVERY=1. Prints the summary and asserts the
rows are real int-id generations (catches the BatchEncoding 400 regression).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_gsm8k import start_server  # noqa: E402

MODEL = os.environ.get("XRL_MODEL", "/data00/meshy/models/Qwen3-0.6B")
PORT = int(os.environ.get("XRL_SMOKE_PORT", "30021"))


async def main() -> None:
    log = os.path.join(tempfile.mkdtemp(prefix="inloopsmoke_"), "server.log")
    proc = start_server(MODEL, PORT, 0.6, log)
    runtime = tempfile.mkdtemp(prefix="inloopsmoke_run_")
    os.environ["XRL_RUNTIME_DIR"] = runtime
    os.environ["XRL_EVAL_N"] = "5"
    os.environ["XRL_EVAL_EVERY"] = "1"
    try:
        from meshy.engine.sglang import SGLangEngine
        import recipe.v100_inloop as inloop

        engine = SGLangEngine([f"http://127.0.0.1:{PORT}"])
        await inloop.version_hook(version=0, engine=engine, model_path=MODEL)
        await engine.close()

        rows = [json.loads(l) for l in open(os.path.join(runtime, "eval", "step0.jsonl"))]
        summary = json.loads(open(os.path.join(runtime, "eval", "summary.jsonl")).read().strip())
        assert len(rows) == 5, rows
        assert all(isinstance(r["tokens"], int) and r["tokens"] > 0 for r in rows), rows
        print("SMOKE_OK", json.dumps(summary))
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
