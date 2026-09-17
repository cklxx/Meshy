"""Single-class wrapper around torchtitan's context-parallel primitives.

:class:`CpSharder` is the only place the trainer talks to CP:

* :meth:`shard_seq` — slice a batch of ``[B, S]`` per-token tensors into the
  CP-local ``[B, S_local]`` head-tail layout (no-op when CP is disabled).
* :meth:`gather_seq` — the exact inverse: reassemble ``[B, S_local]`` shards
  into ``[B, S]`` in true temporal order (no-op when CP is disabled).
* :meth:`all_reduce_sum` — a gradient-free SUM over the CP group, used for
  the per-sequence token counts and for metrics.

Loss reductions deliberately do **not** all-reduce the differentiable
numerator across CP: every CP rank backpropagates its local partial sum
divided by the *global* denominator, and FSDP's gradient reduce (SUM over
``dp_shard * cp``; torchtitan disables the automatic division) assembles the
global gradient. That is exactly the same contract as data parallelism.

Determinism note
----------------
Every call to :meth:`shard_seq` constructs a fresh ``_HeadTailLoadBalancer``
under the same ``(seq_len, cp_world_size)``, which yields a deterministic
permutation. As long as all per-token tensors of a micro-batch are sharded
through one call they are token-for-token aligned on every CP rank — which
is what makes ``new_lp - old_lp`` meaningful under CP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

# ``all_gather_into_tensor`` was renamed in newer torch; prefer the new name
# when present so the call does not emit a deprecation warning every step.
_all_gather_into = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import ParallelDims


class CpSharder:
    def __init__(
        self,
        parallel_dims: "ParallelDims",
        load_balancer: str | None = "headtail",
    ) -> None:
        self._parallel_dims = parallel_dims
        self._load_balancer = load_balancer

    @property
    def enabled(self) -> bool:
        return self._parallel_dims.cp_enabled

    @property
    def load_balancer(self) -> str | None:
        return self._load_balancer

    def shard_seq(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Shard each tensor along ``seq_dim=1`` across the CP mesh.

        All tensors are sharded together with a single
        ``_context_parallel_shard`` call, so they receive identical head-tail
        permutations and end up token-for-token aligned on every CP rank.

        No-op (returns the inputs unchanged) when CP is disabled.
        """
        if not self.enabled:
            return tensors

        from torchtitan.distributed.context_parallel import cp_shard

        cp_mesh = self._parallel_dims.get_mesh("cp")
        sharded, _ = cp_shard(
            cp_mesh,
            tuple(tensors),
            None,
            self._load_balancer,
            input_seq_dim=1,
        )
        return sharded

    @torch.no_grad()
    def gather_seq(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Inverse of :meth:`shard_seq`: ``[B, S_local]`` → ``[B, S]``, in order.

        Needed by anything that is *sequential in time* rather than pointwise.
        Losses are pointwise, so the trainer never needs this: every rank can
        reduce its own shard. GAE is not — it is a backward recursion over
        adjacent timesteps, and the head-tail layout deliberately makes
        neighbouring positions non-adjacent, so the recursion has to see the
        whole sequence.

        Two steps, exactly undoing what ``cp_shard`` did:

        1. all-gather the CP-local shards and concatenate them in rank order,
           which reproduces the *rearranged* sequence (``distribute_tensor``
           with ``Shard(1)`` gave rank ``r`` contiguous chunk ``r``);
        2. apply the load balancer's ``restore`` indices, defined so that
           ``rearranged[restore] == original``.

        No-op (returns the inputs unchanged) when CP is disabled.
        """
        if not self.enabled:
            return tensors
        if not tensors:
            return tensors

        cp_mesh = self._parallel_dims.get_mesh("cp")
        group = cp_mesh.get_group()
        cp_size = cp_mesh.size(0)

        gathered = []
        for t in tensors:
            rows, s_local = t.shape
            # ``all_gather_into_tensor`` concatenates along dim 0, so the output
            # is [cp * rows, S_local]; view it as [cp, rows, S_local], move the
            # rank axis next to the sequence axis, and flatten. Rank r holds
            # chunk r, so the rank axis is the outer (slower) one.
            buf = torch.empty((cp_size * rows, s_local), dtype=t.dtype, device=t.device)
            _all_gather_into(buf, t.contiguous(), group=group)
            gathered.append(
                buf.view(cp_size, rows, s_local).transpose(0, 1).reshape(rows, cp_size * s_local)
            )

        restore = self._restore_indices(
            seq_len=gathered[0].shape[1], cp_size=cp_size, device=gathered[0].device
        )
        if restore is None:
            return tuple(gathered)
        return tuple(t.index_select(1, restore) for t in gathered)

    def _restore_indices(
        self, *, seq_len: int, cp_size: int, device: torch.device
    ) -> torch.Tensor | None:
        """Head-tail restore permutation, or ``None`` when nothing was permuted."""
        if self._load_balancer is None:
            return None
        if self._load_balancer != "headtail":
            # "ptrr" derives its permutation from the attention mask, which we
            # do not have here.
            raise NotImplementedError(
                f"gather_seq supports the 'headtail' load balancer and None, "
                f"got {self._load_balancer!r}"
            )
        # Same import path torchtitan's own ``cp_shard`` uses, so the two stay
        # on the same implementation.
        from torch.distributed.tensor.experimental._attention import (
            _HeadTailLoadBalancer,
        )

        lb = _HeadTailLoadBalancer(seq_len, cp_size, device)
        return lb._generate_indices(restore=True)[0].to(device=device, dtype=torch.long)

    @torch.no_grad()
    def all_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        """SUM ``value`` over the CP group (no autograd). Identity when CP is off."""
        if not self.enabled:
            return value
        out = value.detach().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self._parallel_dims.get_mesh("cp").get_group())
        return out
