"""Regression test: no module-level name may be shadowed inside
``TitanTrainer.__init__`` before its first use.

The GradScaler lives near the top of ``__init__``; a later function-local
``import torch._dynamo`` binds ``torch`` for the *whole* function and made
the earlier GradScaler construction raise ``UnboundLocalError`` (fixed in
favour of ``from torch import _dynamo``). The direct bytecode check fails
on that exact regression; the fake-backend construction below is the
end-to-end guard and runs on CPU (no GPU, no tensordict needed).
"""

from __future__ import annotations

import ast
import dis
import os


def test_init_does_not_locally_bind_torch_before_use():
    path = os.path.join(
        os.path.dirname(__file__), "..", "meshy", "backend", "titan", "trainer.py"
    )
    src = open(os.path.abspath(path)).read()
    tree = ast.parse(src)
    init = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "TitanTrainer"
        for n in n.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    # Any torch import inside __init__ must bind a submodule name, never
    # ``torch`` itself (which would shadow the module-level import).
    for node in ast.walk(init):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                assert bound != "torch", (
                    f"line {node.lineno}: `{ast.unparse(node)}` shadows the "
                    "module-level `torch` in TitanTrainer.__init__; use "
                    "`from torch import <submodule>` instead"
                )


def test_init_first_torch_use_is_global_load():
    """The GradScaler line must emit LOAD_GLOBAL torch, not LOAD_FAST."""
    from meshy.backend.titan import trainer as trainer_mod

    init = trainer_mod.TitanTrainer.__init__
    # Find the first instruction loading the name torch.
    first_torch = next(
        (ins for ins in dis.get_instructions(init) if ins.argval == "torch"),
        None,
    )
    assert first_torch is not None, "expected a torch reference in __init__"
    assert first_torch.opname == "LOAD_GLOBAL", (
        f"torch resolves via {first_torch.opname} at offset "
        f"{first_torch.offset}: a function-local binding shadows the module "
        "import (the import-torch-._dynamo regression)"
    )


def test_construct_trainer_on_cpu_fake_backend(monkeypatch):
    """Actually build TitanTrainer (debugmodel) under the fake dist backend.

    This is the end-to-end guard: with the shadowing bug, __init__ raised
    UnboundLocalError before reaching model construction.
    """
    for k, v in (("LOCAL_RANK", "0"), ("RANK", "0"), ("WORLD_SIZE", "1"),
                 ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29581")):
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    import torch
    import sys

    if sys.platform != "linux" or torch.cuda.is_available() or getattr(
        torch.backends, "mps", None
    ) and torch.backends.mps.is_available():
        import pytest

        pytest.skip("needs a Linux CPU box (forge engine rejects MPS/CUDA here)")

    from torchtitan.distributed import utils as dist_utils
    from torchtitan.config.configs import CommConfig

    dist_utils.init_distributed(CommConfig())

    from meshy.backend.titan.config import build_forge_config
    from meshy.backend.titan.trainer import TitanTrainer
    from meshy.config import TrainerConfig

    cfg = TrainerConfig(
        model_name="qwen3", model_flavor="debugmodel", seq_len=64,
        attn_backend="sdpa", dp_shard_degree=-1, enable_checkpoint=False,
        compile_model=False, dump_folder="/tmp/xrl_shadow_test", lr=0.0,
        dtype="float32",
    )
    trainer = TitanTrainer(
        build_forge_config(cfg),
        micro_batch_size=1, mini_batch_size=1, batch_layout="padded",
        seq_bucket=64, timer_enabled=False,
    )
    # The GradScaler exists (fp32 storage -> disabled, but constructed).
    assert trainer.grad_scaler is not None
    import torch.distributed as dist

    dist.destroy_process_group()
