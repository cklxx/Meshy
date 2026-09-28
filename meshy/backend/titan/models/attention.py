"""Inner-attention backends for compute capabilities torchtitan does not cover.

Background: torchtitan's :class:`ScaledDotProductAttention` hard-codes the
backend priority ``[CUDNN_ATTENTION, FLASH_ATTENTION, MATH]``. On sm70 (V100,
no bf16 tensor cores) the first two have no kernel, so ``F.scaled_dot_product_attention``
falls through to the MATH backend, which materialises the full
``[heads, seq, seq]`` score matrix (O(seq^2) memory; a 5120-token fp16 row
asks for ~12.5 GiB). That is the backwards-OOM and slow-attention root cause
on V100.

PyTorch's mem-efficient SDPA kernel (xFormers-style) *is* built for sm50+ and
runs on sm70 in fp16 with O(seq) memory, but torchtitan never lists it.
:class:`Sm70ScaledDotProductAttention` inserts ``EFFICIENT_ATTENTION`` ahead
of ``MATH`` only on the affected capability, leaving every other
architecture on torchtitan's original order.
"""

from __future__ import annotations

import torch
from torchtitan.models.common.attention import ScaledDotProductAttention
from torch.nn.attention import SDPBackend


def sm70_sdpa_backends() -> list[SDPBackend]:
    """SDPA backend order for the current device's compute capability.

    sm7x (Volta/Turing, fp16-capable, no flash/cudnn SDPA kernel) prefers the
    mem-efficient kernel over the score-materialising MATH backend. Every other
    capability keeps torchtitan's upstream order exactly.
    """
    upstream = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (7, 0):
        return [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
    return upstream


class Sm70ScaledDotProductAttention(ScaledDotProductAttention):
    """SDPA inner attention whose backend list includes mem-efficient on sm70."""

    # Class attribute read by torchtitan's ScaledDotProductAttention.forward
    # via ``self.sdpa_backends``; overriding it is enough (no forward copy).
    sdpa_backends: list[SDPBackend] = sm70_sdpa_backends()
