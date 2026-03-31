# SPDX-License-Identifier: Apache-2.0
"""MiniKV-adapter entrypoint for TurboQuant fused decode attention.

This module provides a stable API surface for "MiniKV-style fused decode"
experiments while delegating execution to TurboQuant's production fused
Triton decode kernel.
"""

from __future__ import annotations

import torch

from vllm.v1.attention.ops.triton_turboquant_paged_attn import (
    turboquant_fused_paged_decode,
)


def turboquant_minikv_fused_decode(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    rotation: torch.Tensor,
    scale: float,
    skip_output_inverse_rotation: bool = False,
    rotate_inside_decode: bool = False,
) -> torch.Tensor:
    """Decode with fused in-register TurboQuant decompression + attention.

    The adapter naming is intentional: it lets us benchmark and evolve
    MiniKV-inspired integration points without duplicating kernel code.
    """
    return turboquant_fused_paged_decode(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        rotation=rotation,
        scale=scale,
        skip_output_inverse_rotation=skip_output_inverse_rotation,
        rotate_inside_decode=rotate_inside_decode,
    )

