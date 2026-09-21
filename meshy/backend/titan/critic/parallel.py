"""Parallelisation for :class:`CriticModel`.

Mirrors ``torchtitan.models.llama3.parallelize.parallelize_llama`` (CP → TP →
AC → compile → FSDP) but the FSDP unit grouping differs: the actor groups
``[norm, lm_head]`` together, the critic has no ``lm_head`` and instead groups
``[norm, value_head]`` so the tiny value head gets its own fully-sharded unit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import ActivationCheckpointConfig, CompileConfig, ParallelismConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.tools.logging import logger

from .model import CriticModel


def parallelize_critic(
    model: CriticModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply TP / AC / compile / FSDP to a critic model (no pipeline support)."""
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"seq_len {training.seq_len} not divisible by {parallel_dims.seq_len_divisor}"

    # CP: wrap inner attention before parallelize() so CP runs inside local_map.
    if parallel_dims.cp_enabled:
        apply_cp_to_forward(
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        model.parallelize(tp_mesh)
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    model_compile_enabled = compile_config.enable and "model" in compile_config.components

    if ac_config.mode != "none":
        apply_ac(
            model, ac_config, model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    if model_compile_enabled:
        apply_compile(model, compile_config)

    names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    dp_mesh = parallel_dims.get_mesh(names)
    _apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
    )

    if parallel_dims.dp_replicate_enabled:
        logger.info("Applied HSDP to the critic")
    else:
        logger.info("Applied FSDP to the critic")
    return model


def _apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, None
    )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    # No weight tying: value head is a distinct FSDP unit that prefixes
    # immediately after norm, so group norm+value_head together.
    if model.norm is not None:
        fully_shard(
            [model.norm, model.value_head],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    for layer_id, transformer_block in model.layers.items():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)


__all__ = ["parallelize_critic"]
