# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TurboQuant fused Triton kernels for decode attention.

Ports the 0xSero/turboquant fully-fused Triton kernels that compute
attention scores directly from packed TurboQuant-compressed data.

The main bottleneck during decode is computing attention scores from the
packed TurboQuant representation. Without fusion, the PyTorch path is:

  1. Unpack MSE indices (bit-shift)
  2. Lookup centroids (gather)
  3. Rotate back (d×d matmul)
  4. Scale by norms
  5. Dot with query (another matmul)
  ──
  6. Sketch query through S (d×d matmul)
  7. Unpack QJL signs (bit-shift)
  8. Dot sketched query with signs
  9. Scale by residual norms

With fusion, we avoid materializing the full d-dim dequantized vectors.
Instead we compute the score directly from packed data.
"""

from ..triton_turboquant_fused_decode import (
    turboquant_fused_decode,
    turboquant_mse_score,
    turboquant_qjl_score,
    _get_packing_params,
)

__all__ = [
    "turboquant_fused_decode",
    "turboquant_mse_score",
    "turboquant_qjl_score",
    "_get_packing_params",
]