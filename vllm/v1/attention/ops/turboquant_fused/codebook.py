# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Lloyd-Max codebook for TurboQuant MSE quantization.

Uses precomputed codebooks for common head dimensions.
Codebooks are computed offline using scipy and stored as JSON.
"""

import os
import json
import torch


# ── Codebook cache ──────────────────────────────────────────────────────
_CODEBOOK_CACHE: dict[tuple[int, int], dict] = {}


def get_codebook(d: int, bits: int) -> dict:
    """Get a precomputed codebook from cache or disk."""
    key = (d, bits)
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key]

    # Try loading from disk
    codebook_dir = os.path.join(os.path.dirname(__file__), "codebooks")
    path = os.path.join(codebook_dir, f"codebook_d{d}_b{bits}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            cb = json.load(f)
        _CODEBOOK_CACHE[key] = cb
        return cb

    # Fallback: use uniform centroids if precomputed codebook not available
    n_clusters = 2**bits
    centroids = []
    for i in range(n_clusters):
        t = (2 * i + 1) / (2 * n_clusters)
        centroids.append(1.0 - 2.0 * t)
    boundaries = [-1.0] + [
        (centroids[i] + centroids[i + 1]) / 2.0
        for i in range(n_clusters - 1)
    ] + [1.0]

    cb = {
        "centroids": centroids,
        "boundaries": boundaries,
        "mse_per_coord": 0.0,
        "mse_total": 0.0,
        "d": d,
        "bits": bits,
    }
    _CODEBOOK_CACHE[key] = cb
    return cb


def get_codebook_tensors(
    d: int, bits: int, device: torch.device, dtype: torch.dtype = torch.float32
):
    """Get codebook as GPU tensors ready for quantization."""
    cb = get_codebook(d, bits)
    centroids = torch.tensor(cb["centroids"], device=device, dtype=dtype)
    boundaries = torch.tensor(cb["boundaries"], device=device, dtype=dtype)
    return centroids, boundaries