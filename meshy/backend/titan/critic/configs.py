"""Build a :class:`CriticModel.Config` from an actor :class:`~torchtitan.protocols.model_spec.ModelSpec`.

The critic reuses the actor's architecture but is not the actor. It keeps the
backbone sub-configs (``tok_embeddings``, ``rope``, per-layer blocks, ``norm``)
so a change to the actor's flavor automatically applies to the critic, but it
drops ``lm_head`` and adds a ``value_head`` (added by :class:`CriticModel`).

``enable_weight_tying`` binds ``tok_embeddings`` to ``lm_head`` in the actor;
the critic has no ``lm_head``, so it is set ``False`` and the embedding becomes
a standalone parameter loadable from the base model.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any

import torch.nn as nn

from .model import CriticModel


def critic_config_from_actor(actor_config: Any) -> CriticModel.Config:
    """Derive a ``CriticModel.Config`` from an actor ``*Model.Config``.

    ``actor_config`` is anything deriving ``torchtitan.models.common.decoder
    .Decoder.Config`` (``Qwen3Model.Config``, ``Llama3Model.Config``, ...). Its
    ``tok_embeddings`` / ``rope`` / ``norm`` / ``layers`` are reused verbatim;
    ``lm_head`` is dropped (the critic adds a ``value_head`` instead).

    If the actor relied on weight tying (``skip_param_init`` embedding), the
    embedding config is rewritten with a plain normal initialiser so the critic
    embedding is seeded rather than left as ``to_empty``'s zeros.
    """
    import torch.nn.init as init

    tok = actor_config.tok_embeddings
    param_init = tok.param_init if tok is not None else None
    if param_init is not None and any(
        callable(v) and getattr(v, "__name__", "") == "skip_param_init"
        for v in param_init.values()
    ):
        from torchtitan.models.common.param_init import skip_param_init

        param_init = {
            k: (partial(init.trunc_normal_, std=0.02) if k == "weight" else init.zeros_)
            for k in param_init
        }

    return CriticModel.Config(
        vocab_size=actor_config.vocab_size,
        dim=actor_config.dim,
        tok_embeddings=replace(actor_config.tok_embeddings, param_init=param_init)
        if actor_config.tok_embeddings is not None
        else None,
        norm=actor_config.norm,
        rope=replace(actor_config.rope),
        layers=actor_config.layers,
        embedding_param_init=param_init,
    )


def build_critic_model(actor_config: Any) -> CriticModel:
    """Construct a :class:`CriticModel` from an actor config (on meta device).

    Callers should construct it under ``torch.device("meta")`` + the training
    dtype, mirroring ``ForgeEngine.__init__``, then ``to_empty`` / ``init_states``
    and load the backbone weights.
    """
    return critic_config_from_actor(actor_config).build()


__all__ = ["CriticModel", "build_critic_model", "critic_config_from_actor"]
