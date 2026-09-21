"""Independent PPO critic engine (own ForgeEngine-style replica).

Not an attachment to the actor trainer: the critic owns its own value network,
its own parallelisation, its own optimiser, and trains its own value head. Its
only coupling to the actor is the advantage signal, carried through rollout
sample fields.
"""

from .configs import build_critic_model, critic_config_from_actor
from .data import (
    TrajectorySample,
    load_samples,
    make_batch,
    sample_from_row,
    samples_from_rows,
    split_by_prompt,
)
from .engine import CriticEngine
from .gae import (
    advantages_to_per_sequence,
    compute_returns,
    compute_vapo_gae,
)
from .metrics import (
    Accumulator,
    AccumulatorState,
    CriticDiagnostics,
    critic_diagnostics,
    diagnostics_from_states,
)
from .model import CriticModel
from .parallel import parallelize_critic

__all__ = [
    "CriticEngine",
    "CriticModel",
    "build_critic_model",
    "critic_config_from_actor",
    "parallelize_critic",
    # GAE / VAPO-λ and value targets
    "compute_vapo_gae",
    "compute_returns",
    "advantages_to_per_sequence",
    # trajectory.jsonl / live TQ rows -> batches
    "TrajectorySample",
    "load_samples",
    "sample_from_row",
    "samples_from_rows",
    "split_by_prompt",
    "make_batch",
    # critic health
    "Accumulator",
    "AccumulatorState",
    "CriticDiagnostics",
    "critic_diagnostics",
    "diagnostics_from_states",
]
