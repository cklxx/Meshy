"""Independent PPO critic network (Backbone + value head).

``CriticModel`` is a self-contained TorchTitan model, **not** an attachment to
the actor trainer. It reuses the actor's *architecture* (the same
``tok_embeddings`` / ``rope`` / ``layers`` / ``norm`` sub-configs from a
:class:`~torchtitan.protocols.model_spec.ModelSpec`) but replaces the
``lm_head`` with a small ``value_head`` linear mapping the hidden dimension to
a scalar per token. It deliberately does *not* enable weight tying: weight
tying in the actor binds ``tok_embeddings`` to ``lm_head``, and the critic has
no ``lm_head``. Backbone weights are meant to be loaded from the same base
model as the actor (the value head stays randomly initialised).

Why a separate model instead of a forward hook on the actor: :meth:`Decoder.forward`
only returns post-``lm_head`` logits, never the post-norm hidden state a value
head needs. A forward hook that taps ``self.norm`` works but is fragile under
``fully_shard`` / ``torch.compile`` / CP wrappers and couples the critic's
lifecycle to the actor's. A standalone model keeps the critic an independent
ForgeEngine replica (see :mod:`meshy.backend.titan.critic.engine`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import ModuleDict

__all__ = ["CriticModel"]


class CriticModel(BaseModel):
    """Decoder backbone (to ``norm``) followed by a scalar value head.

    ``forward`` returns per-token values shaped like the actor's per-token
    tensors (``[rows, S]`` under the same layout/CP conventions), so values can
    be token-aligned with ``input_ids`` / ``mask`` / ``doc_ids`` downstream.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        """Backbone fields identical to the actor, but with no ``lm_head``.

        Because there is no ``lm_head`` there is no weight tying, so the
        embedding is a standalone parameter that can be loaded straight from
        the base model.
        """

        dim: int
        vocab_size: int
        tok_embeddings: Embedding.Config
        norm: RMSNorm.Config
        rope: RoPE.Config
        layers: list  # list[TransformerBlock.Config]
        # The actor ties embeddings to an lm_head and often leaves the
        # embedding initialiser at ``skip_param_init``; a critic has no lm_head
        # so the embedding must be initialised explicitly.
        embedding_param_init: dict | None = None

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            # Extend the RoPE cache to the training ``seq_len`` just like the
            # actor (see ``Qwen3Model.Config.update_from_config``).
            training = getattr(trainer_config, "training", None)
            if training is not None:
                import dataclasses

                if training.seq_len > self.rope.max_seq_len:
                    self.rope = dataclasses.replace(
                        self.rope, max_seq_len=training.seq_len
                    )

        def get_nparams_and_flops(
            self, model: "nn.Module", seq_len: int
        ) -> tuple[int, int]:
            # Rough param count; flops estimation is irrelevant for a critic.
            params = sum(p.numel() for p in model.parameters())
            return params, 0

    def __init__(self, config: Config):
        # Intentionally do NOT call Decoder.__init__: that builds an lm_head,
        # which a critic must not have. Assemble the backbone by hand.
        nn.Module.__init__(self)
        self.config = config

        self.tok_embeddings = (
            config.tok_embeddings.build() if config.tok_embeddings is not None else None
        )
        self.rope = config.rope.build()
        self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        self.layers = ModuleDict()
        for i, layer_config in enumerate(config.layers):
            self.layers[str(i)] = layer_config.build()

        self.norm = config.norm.build() if config.norm is not None else None

        self.value_head = nn.Linear(config.dim, 1)
        # ``value_head`` is a submodule, so ``named_parameters(recurse=False)``
        # does not see its params; seed it explicitly in ``init_states``.
        self._value_head_init_std = 0.02

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self.freqs_cis.device
        # Recursively initialise the backbone children (layers, norms, rope),
        # then seed the value head (``to_empty`` leaves it all zeros).
        super().init_states(buffer_device=buffer_device)
        with torch.no_grad():
            nn.init.normal_(self.value_head.weight, std=self._value_head_init_std)
            if self.value_head.bias is not None:
                nn.init.zeros_(self.value_head.bias)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        # ``RoPE._init_self_buffers`` (already run by the recursion above)
        # *reassigns* ``self.rope.cache``; the buffer registered in
        # ``__init__`` still points at the tensor ``to_empty`` left behind --
        # uninitialised, and no longer following ``rope.cache`` across device
        # moves. Re-point it, exactly as ``Decoder._init_self_buffers`` does.
        assert buffer_device is None or buffer_device.type != "meta", (
            f"buffer_device must not be meta, got {buffer_device}; buffers are "
            "initialised on a real device after to_empty()."
        )
        self.freqs_cis = self.rope.cache

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: Any = None,
        positions: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        # Replicate Decoder.forward up to and including ``norm``, then feed the
        # post-norm hidden state to the value head instead of an lm_head.
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks, positions)
        h = self.norm(h) if self.norm is not None else h
        # Run the value head in fp32 so the scalar output is never quantised to
        # bf16 precision, matching miles' LinearForLastLayer. The *weight* has
        # to be cast too, not just the input: the head is built under the
        # bf16 default dtype and FSDP's ``MixedPrecisionPolicy`` all-gathers it
        # as ``param_dtype`` regardless, so ``value_head(h.float())`` is a
        # guaranteed fp32-times-bf16 dtype mismatch. The cast is on a [1, dim]
        # weight, i.e. free next to the activation it multiplies.
        weight = self.value_head.weight.float()
        bias = self.value_head.bias
        values = F.linear(h.float(), weight, None if bias is None else bias.float())
        if return_hidden:
            return values, h  # [rows, S, 1], [rows, S, dim]
        return values  # [rows, S, 1]


__all__ = ["CriticModel"]
