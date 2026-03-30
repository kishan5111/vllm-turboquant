# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.v1.attention.ops.triton_turboquant_kv import apply_rotation
from vllm.v1.attention.ops.qjl_kernels import (
    make_qjl_projection,
    qjl_score_stage12_triton,
    quantize_values_2bit,
)

_CONST_TENSOR_CACHE: dict[tuple[str, int | None, torch.dtype, tuple[int, ...]], torch.Tensor] = {}
_ARANGE_CACHE: dict[tuple[str, int | None, torch.dtype, int], torch.Tensor] = {}


def _device_cache_key(device: torch.device) -> tuple[str, int | None]:
    return (device.type, device.index)


def _get_const_tensor(
    values: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (*_device_cache_key(device), dtype, values)
    tensor = _CONST_TENSOR_CACHE.get(key)
    if tensor is None:
        tensor = torch.tensor(values, device=device, dtype=dtype)
        _CONST_TENSOR_CACHE[key] = tensor
    return tensor


def _get_arange(
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (*_device_cache_key(device), dtype, size)
    tensor = _ARANGE_CACHE.get(key)
    if tensor is None:
        tensor = torch.arange(size, device=device, dtype=dtype)
        _ARANGE_CACHE[key] = tensor
    return tensor


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
        head_size * 2 // 8
        + sketch_dim // 8
        + 2
        + 2
        + outlier_count
        + head_size * value_bits // 8
        + value_groups * 2
        + value_groups * 2
    )


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
    _, _, _, slot_size = packed_cache.shape
    key_pack_bytes = head_size * 2 // 8
    residual_hash_bytes = sketch_dim // 8
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
    key_hash = packed_cache[..., off:off + key_pack_bytes]
    off += key_pack_bytes
    key_outlier_hash = packed_cache[..., off:off + residual_hash_bytes]
    off += residual_hash_bytes
    key_norm = packed_cache[..., off:off + 2].view(scalar_dtype).squeeze(-1)
    off += 2
    key_outlier_norm = packed_cache[..., off:off + 2].view(scalar_dtype).squeeze(-1)
    off += 2
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


def _unpack_hash_bits(packed_hash: torch.Tensor, sketch_dim: int) -> torch.Tensor:
    bit_offsets = _get_const_tensor(
        (0, 1, 2, 3, 4, 5, 6, 7),
        device=packed_hash.device,
        dtype=torch.uint8,
    )
    bits = ((packed_hash.unsqueeze(-1) >> bit_offsets) & 1).reshape(
        *packed_hash.shape[:-1], -1
    )
    return bits[..., :sketch_dim].contiguous()


def _pack_bits_lastdim(bits: torch.Tensor) -> torch.Tensor:
    if bits.shape[-1] % 8 != 0:
        bits = F.pad(bits, (0, 8 - (bits.shape[-1] % 8)))
    bit_offsets = _get_const_tensor(
        (0, 1, 2, 3, 4, 5, 6, 7),
        device=bits.device,
        dtype=torch.uint8,
    )
    return torch.sum(
        (bits.reshape(*bits.shape[:-1], -1, 8) << bit_offsets).to(torch.int32),
        dim=-1,
    ).to(torch.uint8)


def _popcount_u8(x: torch.Tensor) -> torch.Tensor:
    lut = _get_const_tensor(
        tuple(i.bit_count() for i in range(256)),
        device=x.device,
        dtype=torch.uint8,
    )
    return lut[x.to(torch.long)]


def _pack_stage_a_2bit(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric per-token 2-bit quantizer used for the stage-A key sketch."""
    scale = x.abs().amax(dim=-1).clamp_min(1e-8)
    q = torch.clamp(
        torch.round(x / scale.unsqueeze(-1) * 1.5 + 1.5),
        0,
        3,
    ).to(torch.uint8)
    q_groups = q.reshape(*q.shape[:-1], -1, 4)
    packed = (
        q_groups[..., 0]
        | (q_groups[..., 1] << 2)
        | (q_groups[..., 2] << 4)
        | (q_groups[..., 3] << 6)
    ).contiguous()

    dequant = (q.to(torch.float32) / 1.5 - 1.0) * scale.unsqueeze(-1)
    residual = x.to(torch.float32) - dequant
    return packed, scale.to(x.dtype), residual


def _unpack_stage_a_2bit(
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    shifts = _get_const_tensor((0, 2, 4, 6), device=packed.device, dtype=torch.uint8)
    q = ((packed.unsqueeze(-1) >> shifts) & 0x3).reshape(
        *packed.shape[:-1], -1
    ).to(torch.float32)
    return (q / 1.5 - 1.0) * scale.to(torch.float32).unsqueeze(-1)


def _quantize_values_2bit_tensor(
    values: torch.Tensor,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized 2-bit quantization used by the QJL write path."""
    *outer_shape, head_size = values.shape
    num_groups = head_size // group_size
    groups = values.to(torch.float32).reshape(*outer_shape, num_groups, group_size)

    vmin = groups.amin(dim=-1)
    vmax = groups.amax(dim=-1)
    scale = (vmax - vmin) / 3.0
    safe_scale = scale.clamp_min(1e-8)

    q = torch.round(
        (groups - vmin.unsqueeze(-1)) / safe_scale.unsqueeze(-1)
    ).clamp_(0, 3)
    q = torch.where(
        (scale > 1e-8).unsqueeze(-1),
        q,
        torch.zeros_like(q),
    ).to(torch.uint8)

    q_groups = q.reshape(*outer_shape, num_groups, group_size // 4, 4)
    pack = (
        q_groups[..., 0]
        | (q_groups[..., 1] << 2)
        | (q_groups[..., 2] << 4)
        | (q_groups[..., 3] << 6)
    ).reshape(*outer_shape, head_size // 4).contiguous()

    return pack, scale.to(values.dtype), vmin.to(values.dtype)


def _score_stage12_batched(
    query_score: torch.Tensor,
    q_hash_packed: torch.Tensor,
    q_norms: torch.Tensor,
    key_pack: torch.Tensor,
    key_scale: torch.Tensor,
    residual_hash_packed: torch.Tensor,
    residual_norm: torch.Tensor,
    *,
    sketch_dim: int,
) -> torch.Tensor:
    batch, q_heads, head_size = query_score.shape
    _, kv_heads, total_tokens, _ = key_pack.shape
    assert q_heads % kv_heads == 0
    q_per_kv = q_heads // kv_heads

    query_grouped = query_score.to(torch.float32).view(
        batch, kv_heads, q_per_kv, head_size
    )
    q_hash_grouped = q_hash_packed.view(
        batch, kv_heads, q_per_kv, q_hash_packed.shape[-1]
    )
    q_norm_grouped = q_norms.view(batch, kv_heads, q_per_kv)

    scores = torch.empty(
        batch, q_heads, total_tokens,
        dtype=torch.float32,
        device=query_score.device,
    )
    scores_grouped = scores.view(batch, kv_heads, q_per_kv, total_tokens)

    chunk_tokens = 1024
    for start in range(0, total_tokens, chunk_tokens):
        end = min(start + chunk_tokens, total_tokens)
        key_stage_a = _unpack_stage_a_2bit(
            key_pack[:, :, start:end],
            key_scale[:, :, start:end],
        )
        coarse = torch.einsum(
            "bhqd,bhkd->bhqk",
            query_grouped,
            key_stage_a,
        )

        xor = q_hash_grouped.unsqueeze(3) ^ residual_hash_packed[:, :, start:end].unsqueeze(2)
        mismatches = _popcount_u8(xor).sum(dim=-1).to(torch.float32)
        centered = (sketch_dim - 2.0 * mismatches) / sketch_dim
        correction = centered * q_norm_grouped.unsqueeze(-1) * (
            residual_norm[:, :, start:end].to(torch.float32).unsqueeze(2)
        )
        scores_grouped[:, :, :, start:end] = coarse + correction

    return scores


def _score_stage12_multi_query(
    query_score: torch.Tensor,
    q_hash_packed: torch.Tensor,
    q_norms: torch.Tensor,
    key_pack: torch.Tensor,
    key_scale: torch.Tensor,
    residual_hash_packed: torch.Tensor,
    residual_norm: torch.Tensor,
    *,
    sketch_dim: int,
) -> torch.Tensor:
    q_len, q_heads, head_size = query_score.shape
    kv_heads, total_tokens, _ = key_pack.shape
    assert q_heads % kv_heads == 0
    q_per_kv = q_heads // kv_heads

    query_grouped = query_score.to(torch.float32).view(
        q_len, kv_heads, q_per_kv, head_size
    )
    q_hash_grouped = q_hash_packed.view(
        q_len, kv_heads, q_per_kv, q_hash_packed.shape[-1]
    )
    q_norm_grouped = q_norms.view(q_len, kv_heads, q_per_kv)

    scores = torch.empty(
        q_len, q_heads, total_tokens,
        dtype=torch.float32,
        device=query_score.device,
    )
    scores_grouped = scores.view(q_len, kv_heads, q_per_kv, total_tokens)

    chunk_tokens = 1024
    for start in range(0, total_tokens, chunk_tokens):
        end = min(start + chunk_tokens, total_tokens)
        key_stage_a = _unpack_stage_a_2bit(
            key_pack[:, start:end],
            key_scale[:, start:end],
        )
        coarse = torch.einsum(
            "qhgd,hkd->qhgk",
            query_grouped,
            key_stage_a,
        )

        xor = (
            q_hash_grouped.unsqueeze(3)
            ^ residual_hash_packed[:, start:end].unsqueeze(0).unsqueeze(2)
        )
        mismatches = _popcount_u8(xor).sum(dim=-1).to(torch.float32)
        centered = (sketch_dim - 2.0 * mismatches) / sketch_dim
        correction = centered * q_norm_grouped.unsqueeze(-1) * (
            residual_norm[:, start:end].to(torch.float32).unsqueeze(0).unsqueeze(2)
        )
        scores_grouped[:, :, :, start:end] = coarse + correction

    return scores


def _logical_block_table(
    batch: int,
    num_blocks: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return _get_arange(num_blocks, device=device, dtype=dtype).expand(batch, -1)


def cuda_quantized_bmm_gqa_dynamic(
    group_size: int,
    fA: torch.Tensor,
    qB: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Chunked GQA matmul over 2-bit packed values without external kernels."""
    if bits != 2:
        raise NotImplementedError("Local QJL adapter currently supports only 2-bit values.")

    assert fA.ndim == 4 and qB.ndim == 4
    batch, q_heads, m_size, k_size = fA.shape
    batch_qb, kv_heads, qB_k, packed_dim = qB.shape
    assert batch_qb == batch
    assert qB_k == k_size
    assert q_heads % kv_heads == 0

    q_per_kv = q_heads // kv_heads
    head_size = packed_dim * 16  # 4 bytes/int32 * 4 two-bit values/byte.
    shifts = _get_const_tensor((0, 2, 4, 6), device=qB.device, dtype=torch.int32)

    output = torch.zeros(
        batch, q_heads, m_size, head_size,
        dtype=torch.float32,
        device=fA.device,
    )
    output_grouped = output.view(batch, kv_heads, q_per_kv, m_size, head_size)

    chunk_tokens = 512
    for start in range(0, k_size, chunk_tokens):
        end = min(start + chunk_tokens, k_size)
        qB_chunk = qB[:, :, start:end].contiguous()
        scale_chunk = scales[:, :, start:end].to(torch.float32)
        zero_chunk = zeros[:, :, start:end].to(torch.float32)

        byte_view = qB_chunk.view(torch.uint8)
        qvals = ((byte_view.unsqueeze(-1).to(torch.int32) >> shifts) & 0x3).reshape(
            batch, kv_heads, end - start, head_size
        ).to(torch.float32)
        scale_full = scale_chunk.repeat_interleave(group_size, dim=-1)
        zero_full = zero_chunk.repeat_interleave(group_size, dim=-1)
        values = qvals * scale_full + zero_full

        weights = fA[..., start:end].to(torch.float32).view(
            batch, kv_heads, q_per_kv, m_size, end - start
        )
        output_grouped += torch.einsum(
            "bhqmk,bhkd->bhqmd",
            weights,
            values,
        )

    return output.to(fA.dtype)


def pack_full_blocks_to_packed_qjl_cache(
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    packed_cache: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    proj_dir_quant: torch.Tensor,
    rotation: torch.Tensor | None = None,
    value_group_size: int = 32,
    value_bits: int = 2,
    outlier_count: int = 8,
    outlier_sketch_dim: int | None = None,
) -> None:
    """Pack full blocks into the uint8 slot layout used by the backend."""
    del outlier_sketch_dim
    assert key_blocks.ndim == 4
    assert value_blocks.shape == key_blocks.shape
    assert value_bits == 2, "Local adapter currently supports only 2-bit values."

    num_blocks, kv_heads, block_size, head_size = key_blocks.shape
    sketch_dim = proj_dir_quant.shape[0]
    hash_bytes = sketch_dim // 8

    expected_slot = qjl_packed_slot_size(
        sketch_dim=sketch_dim,
        outlier_count=outlier_count,
        head_size=head_size,
        value_group_size=value_group_size,
        value_bits=value_bits,
    )
    assert packed_cache.shape[-1] == expected_slot

    key_rot = apply_rotation(key_blocks, rotation) if rotation is not None else key_blocks
    key_pack, key_scale, residual = _pack_stage_a_2bit(key_rot)
    residual_flat = residual.reshape(num_blocks * kv_heads * block_size, head_size)
    residual_sketch = torch.matmul(
        residual_flat,
        proj_dir_quant.to(torch.float32).t(),
    )
    residual_hash = (residual_sketch >= 0).to(torch.uint8).reshape(
        num_blocks, kv_heads, block_size, sketch_dim
    )
    key_hash = key_pack
    key_outlier_hash = _pack_bits_lastdim(residual_hash)
    key_norm = key_scale
    key_outlier_norm = torch.linalg.vector_norm(residual, dim=-1).to(key_blocks.dtype)

    outlier_idx = torch.zeros(
        num_blocks, kv_heads, outlier_count,
        dtype=torch.uint8,
        device=key_blocks.device,
    )

    value_pack_u8, value_scale, value_min = quantize_values_2bit(
        value_blocks,
        group_size=value_group_size,
    )
    value_pack = value_pack_u8.contiguous().view(torch.int32)

    views = _packed_qjl_cache_views(
        packed_cache,
        sketch_dim=sketch_dim,
        outlier_count=outlier_count,
        head_size=head_size,
        scalar_dtype=key_blocks.dtype,
        value_group_size=value_group_size,
        value_bits=value_bits,
    )
    views["key_hash"][block_ids] = key_hash
    views["key_outlier_hash"][block_ids] = key_outlier_hash
    views["key_norm"][block_ids] = key_norm
    views["key_outlier_norm"][block_ids] = key_outlier_norm
    views["outlier_idx"][block_ids] = outlier_idx
    views["value_pack"][block_ids] = value_pack
    views["value_scale"][block_ids] = value_scale.to(key_blocks.dtype)
    views["value_min"][block_ids] = value_min.to(key_blocks.dtype)


def pack_tokens_to_packed_qjl_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    packed_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    proj_dir_quant: torch.Tensor,
    rotation: torch.Tensor | None = None,
    value_group_size: int = 32,
    value_bits: int = 2,
    outlier_count: int = 8,
    assume_valid: bool = False,
) -> None:
    """Pack token-local K/V rows directly into the uint8 cache."""
    assert key.ndim == 3
    assert value.shape == key.shape
    assert value_bits == 2, "Local adapter currently supports only 2-bit values."

    if assume_valid:
        slot_mapping = slot_mapping.to(torch.long)
    else:
        valid_idx = torch.where(slot_mapping >= 0)[0]
        key = key.index_select(0, valid_idx)
        value = value.index_select(0, valid_idx)
        slot_mapping = slot_mapping.index_select(0, valid_idx).to(torch.long)

        if key.shape[0] == 0:
            return

    num_tokens, kv_heads, head_size = key.shape
    sketch_dim = proj_dir_quant.shape[0]
    block_size = packed_cache.shape[2]

    key_rot = apply_rotation(key, rotation) if rotation is not None else key
    key_pack, key_scale, residual = _pack_stage_a_2bit(key_rot)
    residual_flat = residual.reshape(num_tokens * kv_heads, head_size)
    residual_sketch = torch.matmul(
        residual_flat,
        proj_dir_quant.to(torch.float32).t(),
    )
    residual_hash = (residual_sketch >= 0).to(torch.uint8).reshape(
        num_tokens, kv_heads, sketch_dim
    )
    key_outlier_hash = _pack_bits_lastdim(residual_hash)
    key_outlier_norm = torch.linalg.vector_norm(residual, dim=-1).to(key.dtype)

    value_pack_u8, value_scale, value_min = _quantize_values_2bit_tensor(
        value,
        group_size=value_group_size,
    )
    value_pack = value_pack_u8.contiguous().view(torch.int32)

    block_ids = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_pos = slot_mapping.remainder(block_size)

    views = _packed_qjl_cache_views(
        packed_cache,
        sketch_dim=sketch_dim,
        outlier_count=outlier_count,
        head_size=head_size,
        scalar_dtype=key.dtype,
        value_group_size=value_group_size,
        value_bits=value_bits,
    )
    views["key_hash"][block_ids, :, block_pos] = key_pack
    views["key_outlier_hash"][block_ids, :, block_pos] = key_outlier_hash
    views["key_norm"][block_ids, :, block_pos] = key_scale
    views["key_outlier_norm"][block_ids, :, block_pos] = key_outlier_norm
    views["value_pack"][block_ids, :, block_pos] = value_pack
    views["value_scale"][block_ids, :, block_pos] = value_scale.to(key.dtype)
    views["value_min"][block_ids, :, block_pos] = value_min.to(key.dtype)


def packed_qjl_prefix_scores(
    query: torch.Tensor,
    *,
    packed_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    proj_dir_score: torch.Tensor,
    scale: float,
    rotation: torch.Tensor | None = None,
    outlier_count: int = 8,
    value_group_size: int = 32,
    value_bits: int = 2,
    gather_values: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    batch, q_heads, head_size = query.shape
    _, kv_heads, block_size, slot_size = packed_cache.shape
    sketch_dim = proj_dir_score.shape[1]
    key_pack_bytes = head_size * 2 // 8
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
    max_blocks = gather_ids.shape[1]

    key_pack = views["key_hash"][gather_ids].permute(0, 2, 1, 3, 4).reshape(
        batch, kv_heads, -1, key_pack_bytes
    ).contiguous()
    key_scale = views["key_norm"][gather_ids].permute(0, 2, 1, 3).reshape(
        batch, kv_heads, -1
    ).contiguous()
    residual_hash_packed = views["key_outlier_hash"][gather_ids].permute(
        0, 2, 1, 3, 4
    ).contiguous()
    residual_hash_packed = residual_hash_packed.reshape(
        batch, kv_heads, -1, residual_hash_packed.shape[-1]
    )
    residual_norm = views["key_outlier_norm"][gather_ids].permute(
        0, 2, 1, 3
    ).reshape(batch, kv_heads, -1).contiguous()

    if gather_values:
        value_pack = views["value_pack"][gather_ids].permute(0, 2, 1, 3, 4).reshape(
            batch, kv_heads, -1, views["value_pack"].shape[-1]
        ).contiguous()
        value_scale = views["value_scale"][gather_ids].permute(0, 2, 1, 3, 4).reshape(
            batch, kv_heads, -1, value_groups
        ).contiguous()
        value_min = views["value_min"][gather_ids].permute(0, 2, 1, 3, 4).reshape(
            batch, kv_heads, -1, value_groups
        ).contiguous()
    else:
        value_pack = None
        value_scale = None
        value_min = None

    query_score = apply_rotation(query, rotation) if rotation is not None else query
    q_hashes = (
        torch.matmul(query_score.to(torch.float32), proj_dir_score.to(torch.float32)) >= 0
    ).to(torch.uint8)
    q_hashes_packed = _pack_bits_lastdim(q_hashes)
    q_norms = torch.linalg.vector_norm(query_score.to(torch.float32), dim=-1)
    if query.is_cuda:
        scores = qjl_score_stage12_triton(
            query_score,
            q_hashes_packed,
            q_norms,
            key_pack,
            key_scale,
            residual_hash_packed,
            residual_norm,
            seq_lens,
            gqa_ratio=q_heads // kv_heads,
            scale=1.0,
            sketch_dim=sketch_dim,
        )
    else:
        scores = _score_stage12_batched(
            query_score,
            q_hashes_packed,
            q_norms,
            key_pack,
            key_scale,
            residual_hash_packed,
            residual_norm,
            sketch_dim=sketch_dim,
        )
    scores = scores * scale
    token_idx = _get_arange(
        max_blocks * block_size,
        device=query.device,
        dtype=seq_lens.dtype,
    )
    valid = token_idx.unsqueeze(0) < seq_lens.unsqueeze(1)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

    return scores, value_pack, value_scale, value_min


def packed_qjl_prefix_scores_multi_query(
    query: torch.Tensor,
    *,
    packed_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    proj_dir_score: torch.Tensor,
    scale: float,
    rotation: torch.Tensor | None = None,
    outlier_count: int = 8,
    value_group_size: int = 32,
    value_bits: int = 2,
    gather_values: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Return prefix scores for many query tokens against one shared prefix."""
    q_len, q_heads, head_size = query.shape
    _, kv_heads, _, slot_size = packed_cache.shape
    sketch_dim = proj_dir_score.shape[1]
    key_pack_bytes = head_size * 2 // 8
    value_groups = head_size // value_group_size

    assert block_table.ndim == 1
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
    max_blocks = gather_ids.shape[0]

    key_pack = views["key_hash"][gather_ids].permute(1, 0, 2, 3).reshape(
        kv_heads, -1, key_pack_bytes
    ).contiguous()
    key_scale = views["key_norm"][gather_ids].permute(1, 0, 2).reshape(
        kv_heads, -1
    ).contiguous()
    residual_hash_packed = views["key_outlier_hash"][gather_ids].permute(
        1, 0, 2, 3
    ).contiguous()
    residual_hash_packed = residual_hash_packed.reshape(
        kv_heads, -1, residual_hash_packed.shape[-1]
    )
    residual_norm = views["key_outlier_norm"][gather_ids].permute(
        1, 0, 2
    ).reshape(kv_heads, -1).contiguous()

    if gather_values:
        value_pack = views["value_pack"][gather_ids].permute(1, 0, 2, 3).reshape(
            kv_heads, -1, views["value_pack"].shape[-1]
        ).contiguous()
        value_scale = views["value_scale"][gather_ids].permute(1, 0, 2, 3).reshape(
            kv_heads, -1, value_groups
        ).contiguous()
        value_min = views["value_min"][gather_ids].permute(1, 0, 2, 3).reshape(
            kv_heads, -1, value_groups
        ).contiguous()
    else:
        value_pack = None
        value_scale = None
        value_min = None

    query_score = apply_rotation(query, rotation) if rotation is not None else query
    q_hashes = (
        torch.matmul(query_score.to(torch.float32), proj_dir_score.to(torch.float32)) >= 0
    ).to(torch.uint8)
    q_hashes_packed = _pack_bits_lastdim(q_hashes)
    q_norms = torch.linalg.vector_norm(query_score.to(torch.float32), dim=-1)
    scores = _score_stage12_multi_query(
        query_score,
        q_hashes_packed,
        q_norms,
        key_pack,
        key_scale,
        residual_hash_packed,
        residual_norm,
        sketch_dim=sketch_dim,
    )
    scores = scores * scale
    valid = _get_arange(
        max_blocks * packed_cache.shape[2],
        device=query.device,
        dtype=torch.long,
    ) < seq_len
    scores = scores.masked_fill(~valid.view(1, 1, -1), float("-inf"))

    return scores, value_pack, value_scale, value_min


def decode_packed_qjl(
    query: torch.Tensor,
    *,
    packed_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    proj_dir_score: torch.Tensor,
    scale: float,
    rotation: torch.Tensor | None = None,
    outlier_count: int = 8,
    value_group_size: int = 32,
    value_bits: int = 2,
) -> torch.Tensor:
    scores, value_pack, value_scale, value_min = packed_qjl_prefix_scores(
        query,
        packed_cache=packed_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        proj_dir_score=proj_dir_score,
        scale=scale,
        rotation=rotation,
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
