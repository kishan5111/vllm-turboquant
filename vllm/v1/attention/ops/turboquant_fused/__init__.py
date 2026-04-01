# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TurboQuant MSE+QJL fused attention components.

Ports 0xSero/turboquant implementation with fully-fused Triton kernels
for compute-efficient attention over compressed KV cache.
"""

from .quantizer import (
    TurboQuantMSE,
    TurboQuantProd,
    MSEQuantized,
    ProdQuantized,
    _pack_indices,
    _unpack_indices,
)
from .kv_cache import (
    ValueQuantized,
    quantize_values,
    dequantize_values,
    unpack_values,
)
from .rotation import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    rotate_forward,
    rotate_backward,
)
from .codebook import (
    get_codebook_tensors,
)
from .triton_kernels import (
    turboquant_fused_decode,
    turboquant_mse_score,
    turboquant_qjl_score,
)
from .store import (
    FlatCache,
    CompressedKVStore,
)
from .capture import (
    RingBuffer,
    KVCaptureEngine,
)

__all__ = [
    "TurboQuantMSE",
    "TurboQuantProd",
    "MSEQuantized",
    "ProdQuantized",
    "ValueQuantized",
    "FlatCache",
    "CompressedKVStore",
    "RingBuffer",
    "KVCaptureEngine",
    "_pack_indices",
    "_unpack_indices",
    "quantize_values",
    "dequantize_values",
    "unpack_values",
    "generate_rotation_matrix",
    "generate_qjl_matrix",
    "rotate_forward",
    "rotate_backward",
    "get_codebook_tensors",
    "turboquant_fused_decode",
    "turboquant_mse_score",
    "turboquant_qjl_score",
]