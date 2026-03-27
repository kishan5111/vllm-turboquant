# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import torch

try:
    from qjl_kernel import qjl_kernel
    from qjl_kernel import quantization
    from qjl_kernel.new_pack import triton_quantize_and_pack_along_last_dim
except ImportError as exc:  # pragma: no cover - exercised in runtime env only
    raise ImportError(
        "QJL prototype ops require the author kernels on PYTHONPATH. "
        "Set PYTHONPATH=/workspace/QJL and build /workspace/QJL/qjl_kernel."
    ) from exc


def qjl_packed_slot_size(
    *,
    sketch_dim: int,
    outlier_count: int,
    head_size: int,
    value_group_size: int = 32,
    value_bits: int = 2,
) -> int:
    value_groups = head_size // value_group_size
    return (
        sketch_dim // 8
        + sketch_dim // 8
        + 2
        + 2
        + outlier_count
        + head_size * value_bits // 8
        + value_groups * 2
        + value_groups * 2
    )


def cuda_quantized_bmm_gqa_dynamic(
    group_size: int,
    fA: torch.Tensor,
    qB: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Quantized value matmul without expanding KV heads to Q heads."""
    assert fA.ndim == 4 and qB.ndim == 4
    batch, q_heads, m_size, k_size = fA.shape
    batch_qb, kv_heads, qB_k, packed_dim = qB.shape
    assert batch_qb == batch
    assert qB_k == k_size
    assert q_heads % kv_heads == 0

    q_per_kv = q_heads // kv_heads
    feat_per_int = 32 // bits
    n_size = packed_dim * feat_per_int

    fA_flat = fA.reshape(-1, m_size, k_size).contiguous()
    qB_flat = qB.reshape(-1, k_size, packed_dim).transpose(1, 2).contiguous()
    scales_flat = scales.reshape(-1, k_size, scales.shape[-1]).transpose(
        1, 2
    ).contiguous()
    zeros_flat = zeros.reshape(-1, k_size, zeros.shape[-1]).transpose(
        1, 2
    ).contiguous()

    if fA.dtype == torch.float16:
        result = quantization.batchedQuantizedMultiplyAccumulate_half(
            fA_flat, qB_flat, scales_flat, zeros_flat,
            bits, group_size, q_per_kv, True,
        )
    elif fA.dtype == torch.float32:
        result = quantization.batchedQuantizedMultiplyAccumulate_float(
            fA_flat, qB_flat, scales_flat, zeros_flat,
            bits, group_size, q_per_kv, True,
        )
    elif fA.dtype == torch.bfloat16:
        result = quantization.batchedQuantizedMultiplyAccumulate_bf16(
            fA_flat, qB_flat, scales_flat, zeros_flat,
            bits, group_size, q_per_kv, True,
        )
    else:  # pragma: no cover - runtime type guard
        raise TypeError("Unsupported dtype for tensor fA.")

    return result.reshape(batch, q_heads, m_size, n_size)


def _pack_2byte_float_to_bytes(
    x: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return x.to(dtype).contiguous().view(torch.uint8)


def _unpack_2byte_float_from_bytes(
    x: torch.Tensor,
    source_dtype: torch.dtype,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    return x.contiguous().view(source_dtype).to(target_dtype)


def make_qjl_projection(
    head_size: int,
    sketch_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the QJL projection pair used by the author kernels.

    Returns:
        proj_dir_score: [head_size, sketch_dim]
        proj_dir_quant: [sketch_dim, head_size]
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    proj = torch.randn(
        head_size,
        sketch_dim,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    chunks: list[torch.Tensor] = []
    for start in range(0, sketch_dim, head_size):
        stop = min(start + head_size, sketch_dim)
        q, _ = torch.linalg.qr(proj[:, start:stop], mode="reduced")
        chunks.append(q)
    proj_dir_score = torch.cat(chunks, dim=-1)[:, :sketch_dim]
    proj_dir_score = (proj_dir_score * math.sqrt(head_size)).to(
        dtype=dtype, device=device
    ).contiguous()
    proj_dir_quant = proj_dir_score.transpose(0, 1).contiguous()
    return proj_dir_score, proj_dir_quant


def pack_paged_qjl_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    proj_dir_quant: torch.Tensor,
    value_group_size: int = 32,
    value_bits: int = 2,
    outlier_count: int = 8,
    outlier_sketch_dim: int | None = None,
) -> dict[str, torch.Tensor]:
    """Pack paged K/V tensors into a QJL-style cache.

    Args:
        key_cache:   [batch, kv_heads, num_blocks, block_size, head_size]
        value_cache: [batch, kv_heads, num_blocks, block_size, head_size]
    """
    assert key_cache.ndim == 5
    assert value_cache.shape == key_cache.shape
    batch, kv_heads, num_blocks, block_size, head_size = key_cache.shape
    assert head_size <= 255, "Author kernels store outlier dims as uint8."
    assert 0 < outlier_count <= head_size

    if outlier_sketch_dim is None:
        outlier_sketch_dim = proj_dir_quant.shape[0]
    outlier_sketch_dim = min(outlier_sketch_dim, proj_dir_quant.shape[0])

    # Match the author implementation: select a few outlier dimensions per
    # token-group using the groupwise channel norms.
    outlier_norms = key_cache.norm(dim=-2)
    _, outlier_idx = outlier_norms.topk(outlier_count, dim=-1)
    outlier_idx = outlier_idx.to(torch.uint8).contiguous()
    key_hash, key_outlier_hash, key_outlier_norm = qjl_kernel.qjl_quant(
        key_cache.contiguous(),
        outlier_idx.contiguous(),
        proj_dir_quant,
        outlier_sketch_dim,
    )
    key_norm = torch.linalg.vector_norm(key_cache, dim=-1)

    value_flat = value_cache.reshape(batch, kv_heads, num_blocks * block_size, head_size)
    value_pack, value_scale, value_min = triton_quantize_and_pack_along_last_dim(
        value_flat.contiguous(),
        value_group_size,
        value_bits,
    )

    return {
        "key_hash": key_hash.contiguous(),
        "key_outlier_hash": key_outlier_hash.contiguous(),
        "key_norm": key_norm.contiguous(),
        "key_outlier_norm": key_outlier_norm.contiguous(),
        "outlier_idx": outlier_idx,
        "value_pack": value_pack.reshape(
            batch, kv_heads, num_blocks, block_size, value_pack.shape[-1]
        ).contiguous(),
        "value_scale": value_scale.reshape(
            batch, kv_heads, num_blocks, block_size, value_scale.shape[-1]
        ).contiguous(),
        "value_min": value_min.reshape(
            batch, kv_heads, num_blocks, block_size, value_min.shape[-1]
        ).contiguous(),
    }


def pack_qjl_prefix_tail_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    proj_dir_quant: torch.Tensor,
    tail_tokens: int = 128,
    value_group_size: int = 32,
    value_bits: int = 2,
    outlier_count: int = 8,
    outlier_sketch_dim: int | None = None,
) -> dict[str, torch.Tensor]:
    """Split paged K/V into a compressed prefix and an exact recent tail."""
    assert key_cache.ndim == 5
    batch, kv_heads, num_blocks, block_size, head_size = key_cache.shape
    total_tokens = num_blocks * block_size

    tail_blocks = min(num_blocks, math.ceil(max(tail_tokens, 0) / block_size))
    prefix_blocks = max(num_blocks - tail_blocks, 0)

    cache = pack_paged_qjl_cache(
        key_cache[:, :, :prefix_blocks],
        value_cache[:, :, :prefix_blocks],
        proj_dir_quant=proj_dir_quant,
        value_group_size=value_group_size,
        value_bits=value_bits,
        outlier_count=outlier_count,
        outlier_sketch_dim=outlier_sketch_dim,
    ) if prefix_blocks > 0 else {
        "key_hash": key_cache.new_empty(
            (batch, kv_heads, 0, block_size, proj_dir_quant.shape[0] // 8),
            dtype=torch.uint8,
        ),
        "key_outlier_hash": key_cache.new_empty(
            (batch, kv_heads, 0, block_size, proj_dir_quant.shape[0] // 8),
            dtype=torch.uint8,
        ),
        "key_norm": key_cache.new_empty((batch, kv_heads, 0, block_size)),
        "key_outlier_norm": key_cache.new_empty((batch, kv_heads, 0, block_size)),
        "outlier_idx": key_cache.new_empty(
            (batch, kv_heads, 0, outlier_count), dtype=torch.uint8
        ),
        "value_pack": key_cache.new_empty(
            (batch, kv_heads, 0, block_size, head_size * value_bits // 32),
            dtype=torch.int32,
        ),
        "value_scale": key_cache.new_empty(
            (batch, kv_heads, 0, block_size, head_size // value_group_size)
        ),
        "value_min": key_cache.new_empty(
            (batch, kv_heads, 0, block_size, head_size // value_group_size)
        ),
    }

    tail_key = key_cache[:, :, prefix_blocks:].reshape(batch, kv_heads, -1, head_size)
    tail_value = value_cache[:, :, prefix_blocks:].reshape(
        batch, kv_heads, -1, head_size
    )
    cache["tail_key"] = tail_key.contiguous()
    cache["tail_value"] = tail_value.contiguous()
    cache["prefix_blocks"] = torch.tensor(
        prefix_blocks, device=key_cache.device, dtype=torch.int32
    )
    cache["tail_len"] = torch.tensor(
        total_tokens - prefix_blocks * block_size,
        device=key_cache.device,
        dtype=torch.int32,
    )
    return cache


def _packed_qjl_cache_views(
    packed_cache: torch.Tensor,
    *,
    sketch_dim: int,
    outlier_count: int,
    head_size: int,
    scalar_dtype: torch.dtype,
    value_group_size: int = 32,
    value_bits: int = 2,
) -> dict[str, torch.Tensor]:
    """Return typed views over the packed QJL cache without copying."""
    _, _, _, slot_size = packed_cache.shape
    hash_bytes = sketch_dim // 8
    value_pack_bytes = head_size * value_bits // 8
    value_groups = head_size // value_group_size
    assert slot_size == qjl_packed_slot_size(
        sketch_dim=sketch_dim,
        outlier_count=outlier_count,
        head_size=head_size,
        value_group_size=value_group_size,
        value_bits=value_bits,
    )

    off = 0
    key_hash = packed_cache[..., off:off + hash_bytes]
    off += hash_bytes
    key_outlier_hash = packed_cache[..., off:off + hash_bytes]
    off += hash_bytes
    key_norm = packed_cache[..., off:off + 2].view(scalar_dtype).squeeze(-1)
    off += 2
    key_outlier_norm = packed_cache[..., off:off + 2].view(scalar_dtype).squeeze(-1)
    off += 2
    # Outlier indices are duplicated per token in a block; read the first token.
    outlier_idx = packed_cache[..., 0, off:off + outlier_count]
    off += outlier_count
    value_pack = packed_cache[..., off:off + value_pack_bytes].view(torch.int32)
    off += value_pack_bytes
    value_scale = packed_cache[..., off:off + value_groups * 2].view(scalar_dtype)
    off += value_groups * 2
    value_min = packed_cache[..., off:off + value_groups * 2].view(scalar_dtype)

    return {
        "key_hash": key_hash,
        "key_outlier_hash": key_outlier_hash,
        "key_norm": key_norm,
        "key_outlier_norm": key_outlier_norm,
        "outlier_idx": outlier_idx,
        "value_pack": value_pack,
        "value_scale": value_scale,
        "value_min": value_min,
    }


def pack_full_blocks_to_packed_qjl_cache(
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    packed_cache: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    proj_dir_quant: torch.Tensor,
    value_group_size: int = 32,
    value_bits: int = 2,
    outlier_count: int = 8,
    outlier_sketch_dim: int | None = None,
) -> None:
    """Pack full `[block_size, head_size]` blocks into a uint8 cache."""
    assert key_blocks.ndim == 4
    assert value_blocks.shape == key_blocks.shape
    num_blocks, kv_heads, block_size, head_size = key_blocks.shape
    slot_size = packed_cache.shape[-1]
    expected_slot = qjl_packed_slot_size(
        sketch_dim=proj_dir_quant.shape[0],
        outlier_count=outlier_count,
        head_size=head_size,
        value_group_size=value_group_size,
        value_bits=value_bits,
    )
    assert slot_size == expected_slot

    cache = pack_paged_qjl_cache(
        key_blocks.unsqueeze(0),
        value_blocks.unsqueeze(0),
        proj_dir_quant=proj_dir_quant,
        value_group_size=value_group_size,
        value_bits=value_bits,
        outlier_count=outlier_count,
        outlier_sketch_dim=outlier_sketch_dim,
    )

    views = _packed_qjl_cache_views(
        packed_cache,
        sketch_dim=proj_dir_quant.shape[0],
        outlier_count=outlier_count,
        head_size=head_size,
        scalar_dtype=key_blocks.dtype,
        value_group_size=value_group_size,
        value_bits=value_bits,
    )
    views["key_hash"][block_ids] = cache["key_hash"].squeeze(0)
    views["key_outlier_hash"][block_ids] = cache["key_outlier_hash"].squeeze(0)
    views["key_norm"][block_ids] = cache["key_norm"].squeeze(0).to(key_blocks.dtype)
    views["key_outlier_norm"][block_ids] = cache["key_outlier_norm"].squeeze(0).to(
        key_blocks.dtype
    )
    views["outlier_idx"][block_ids] = cache["outlier_idx"].squeeze(0)
    views["value_pack"][block_ids] = cache["value_pack"].squeeze(0)
    views["value_scale"][block_ids] = cache["value_scale"].squeeze(0).to(
        key_blocks.dtype
    )
    views["value_min"][block_ids] = cache["value_min"].squeeze(0).to(key_blocks.dtype)


def decode_packed_qjl(
    query: torch.Tensor,
    *,
    packed_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    proj_dir_score: torch.Tensor,
    scale: float,
    outlier_count: int = 8,
    value_group_size: int = 32,
    value_bits: int = 2,
) -> torch.Tensor:
    """Decode directly from a packed uint8 cache with QJL-style layout."""
    scores, value_pack, value_scale, value_min = packed_qjl_prefix_scores(
        query,
        packed_cache=packed_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        proj_dir_score=proj_dir_score,
        scale=scale,
        outlier_count=outlier_count,
        value_group_size=value_group_size,
        value_bits=value_bits,
    )
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = cuda_quantized_bmm_gqa_dynamic(
        value_group_size,
        weights.unsqueeze(2),
        value_pack,
        value_scale,
        value_min,
        value_bits,
    )
    return out.squeeze(2)


def packed_qjl_prefix_scores(
    query: torch.Tensor,
    *,
    packed_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    proj_dir_score: torch.Tensor,
    scale: float,
    outlier_count: int = 8,
    value_group_size: int = 32,
    value_bits: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed-cache prefix scores and quantized V metadata."""
    batch, q_heads, head_size = query.shape
    _, kv_heads, block_size, slot_size = packed_cache.shape
    sketch_dim = proj_dir_score.shape[1]
    hash_bytes = sketch_dim // 8
    value_pack_bytes = head_size * value_bits // 8
    value_groups = head_size // value_group_size
    assert slot_size == qjl_packed_slot_size(
        sketch_dim=sketch_dim,
        outlier_count=outlier_count,
        head_size=head_size,
        value_group_size=value_group_size,
        value_bits=value_bits,
    )

    views = _packed_qjl_cache_views(
        packed_cache,
        sketch_dim=sketch_dim,
        outlier_count=outlier_count,
        head_size=head_size,
        scalar_dtype=query.dtype,
        value_group_size=value_group_size,
        value_bits=value_bits,
    )
    gather_ids = block_table.clamp(min=0).to(torch.long)

    key_hash = views["key_hash"][gather_ids].permute(0, 2, 1, 3, 4).contiguous()
    key_outlier_hash = views["key_outlier_hash"][gather_ids].permute(
        0, 2, 1, 3, 4
    ).contiguous()
    key_norm = views["key_norm"][gather_ids].permute(0, 2, 1, 3).contiguous()
    key_outlier_norm = views["key_outlier_norm"][gather_ids].permute(
        0, 2, 1, 3
    ).contiguous()
    outlier_idx = views["outlier_idx"][gather_ids].permute(0, 2, 1, 3).contiguous()
    value_pack = views["value_pack"][gather_ids].permute(0, 2, 1, 3, 4).reshape(
        batch, kv_heads, -1, value_pack_bytes // 4
    ).contiguous()
    value_scale = views["value_scale"][gather_ids].permute(0, 2, 1, 3, 4).reshape(
        batch, kv_heads, -1, value_groups
    ).contiguous()
    value_min = views["value_min"][gather_ids].permute(0, 2, 1, 3, 4).reshape(
        batch, kv_heads, -1, value_groups
    ).contiguous()

    query_4d = query.unsqueeze(2).contiguous()
    query_sketch = torch.matmul(query_4d.to(proj_dir_score.dtype), proj_dir_score)
    scores = qjl_kernel.qjl_gqa_score(
        key_hash,
        key_outlier_hash,
        key_norm,
        key_outlier_norm,
        outlier_idx,
        query_sketch,
        query_4d,
        proj_dir_score,
    ).squeeze(-1) * scale

    token_idx = torch.arange(
        block_table.shape[1] * block_size, device=query.device, dtype=seq_lens.dtype
    )
    valid = token_idx.unsqueeze(0) < seq_lens.unsqueeze(1)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    return scores, value_pack, value_scale, value_min


def _qjl_prefix_scores(
    query: torch.Tensor,
    *,
    qjl_cache: dict[str, torch.Tensor],
    block_table: torch.Tensor,
    prefix_lens: torch.Tensor,
    proj_dir_score: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, q_heads, _ = query.shape
    _, kv_heads, max_blocks, block_size, _ = qjl_cache["key_hash"].shape
    if max_blocks == 0:
        empty_scores = query.new_empty((batch, q_heads, 0), dtype=torch.float32)
        empty_values = qjl_cache["value_pack"].reshape(batch, kv_heads, 0, -1)
        empty_scale = qjl_cache["value_scale"].reshape(batch, kv_heads, 0, -1)
        empty_min = qjl_cache["value_min"].reshape(batch, kv_heads, 0, -1)
        return empty_scores, empty_values, empty_scale, empty_min

    batch_idx = torch.arange(batch, device=query.device)[:, None]
    gather_ids = block_table[:, :max_blocks].clamp(min=0)

    key_hash = qjl_cache["key_hash"][batch_idx, :, gather_ids].permute(
        0, 2, 1, 3, 4
    ).contiguous()
    key_outlier_hash = qjl_cache["key_outlier_hash"][batch_idx, :, gather_ids].permute(
        0, 2, 1, 3, 4
    ).contiguous()
    key_norm = qjl_cache["key_norm"][batch_idx, :, gather_ids].permute(
        0, 2, 1, 3
    ).contiguous()
    key_outlier_norm = qjl_cache["key_outlier_norm"][batch_idx, :, gather_ids].permute(
        0, 2, 1, 3
    ).contiguous()
    outlier_idx = qjl_cache["outlier_idx"][batch_idx, :, gather_ids].permute(
        0, 2, 1, 3
    ).contiguous()

    value_pack = qjl_cache["value_pack"][batch_idx, :, gather_ids].permute(
        0, 2, 1, 3, 4
    ).reshape(batch, kv_heads, max_blocks * block_size, -1)
    value_scale = qjl_cache["value_scale"][batch_idx, :, gather_ids].permute(
        0, 2, 1, 3, 4
    ).reshape(batch, kv_heads, max_blocks * block_size, -1)
    value_min = qjl_cache["value_min"][batch_idx, :, gather_ids].permute(
        0, 2, 1, 3, 4
    ).reshape(batch, kv_heads, max_blocks * block_size, -1)

    query_4d = query.unsqueeze(2).contiguous()
    query_sketch = torch.matmul(query_4d.to(proj_dir_score.dtype), proj_dir_score)
    scores = qjl_kernel.qjl_gqa_score(
        key_hash,
        key_outlier_hash,
        key_norm,
        key_outlier_norm,
        outlier_idx,
        query_sketch,
        query_4d,
        proj_dir_score,
    ).squeeze(-1)
    scores = scores * scale

    token_idx = torch.arange(
        max_blocks * block_size, device=query.device, dtype=prefix_lens.dtype
    )
    valid = token_idx.unsqueeze(0) < prefix_lens.unsqueeze(1)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    return scores, value_pack, value_scale, value_min


def decode_paged_qjl(
    query: torch.Tensor,
    *,
    qjl_cache: dict[str, torch.Tensor],
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    proj_dir_score: torch.Tensor,
    scale: float,
    value_bits: int = 2,
) -> torch.Tensor:
    """Decode one-token queries against a paged QJL-style cache.

    Args:
        query: [batch, q_heads, head_size]
        block_table: [batch, max_blocks] physical block ids
        seq_lens: [batch]
    """
    assert query.ndim == 3
    batch, q_heads, head_size = query.shape
    _, kv_heads, _, block_size, _ = qjl_cache["key_hash"].shape

    prefix_scores, value_pack, value_scale, value_min = _qjl_prefix_scores(
        query,
        qjl_cache=qjl_cache,
        block_table=block_table,
        prefix_lens=seq_lens,
        proj_dir_score=proj_dir_score,
        scale=scale,
    )
    weights = torch.softmax(prefix_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = cuda_quantized_bmm_gqa_dynamic(
        32,
        weights.unsqueeze(2),
        value_pack,
        value_scale,
        value_min,
        value_bits,
    )

    tail_key = qjl_cache.get("tail_key")
    tail_value = qjl_cache.get("tail_value")
    if tail_key is not None and tail_value is not None and tail_key.shape[2] > 0:
        tail_len = qjl_cache["tail_len"].item()
        tail_lens = torch.minimum(
            seq_lens,
            torch.full_like(seq_lens, tail_len),
        )
        prefix_lens = seq_lens - tail_lens

        tail_scores = torch.einsum(
            "bhd,bhkd->bhk",
            query,
            tail_key.repeat_interleave(q_heads // kv_heads, dim=1),
        ) * scale
        tail_idx = torch.arange(tail_key.shape[2], device=query.device, dtype=seq_lens.dtype)
        tail_valid = tail_idx.unsqueeze(0) < tail_lens.unsqueeze(1)
        tail_scores = tail_scores.masked_fill(~tail_valid.unsqueeze(1), float("-inf"))

        all_scores = torch.cat([prefix_scores, tail_scores], dim=-1)
        all_weights = torch.softmax(all_scores, dim=-1, dtype=torch.float32).to(query.dtype)

        prefix_weight_len = prefix_scores.shape[-1]
        prefix_weights = all_weights[..., :prefix_weight_len]
        tail_weights = all_weights[..., prefix_weight_len:]

        out = cuda_quantized_bmm_gqa_dynamic(
            32,
            prefix_weights.unsqueeze(2),
            value_pack,
            value_scale,
            value_min,
            value_bits,
        )
        tail_value_rep = tail_value.repeat_interleave(q_heads // kv_heads, dim=1)
        out = out + torch.einsum(
            "bhk,bhkd->bhd",
            tail_weights,
            tail_value_rep,
        ).unsqueeze(2)

    return out.squeeze(2)
