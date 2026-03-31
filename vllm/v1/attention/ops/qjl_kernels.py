# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
QJL (Quantized Jacobian-Lindenstrauss) Attention Kernels - Optimized Implementation.

Based on: https://github.com/amirzandieh/QJL

Key ideas:
1. Johnson-Lindenstrauss transform projects K/V to a lower-dimensional sketch space
2. In sketch space, K can be quantized to 1-bit (sign bits only!)
3. Score = popcount(XOR(Q_sketch, K_sketch)) / sketch_dim (cosine-like similarity)
4. Outlier dimensions (high-magnitude) stored exactly for accuracy
5. V stored as 2-bit quantized for memory efficiency

This avoids per-step KV dequantization - K is stored as bits, V is only
dequantized for the weighted sum (not score computation).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


# =============================================================================
# QJL Configuration and Types
# =============================================================================

class QJLConfig(NamedTuple):
    """QJL attention configuration."""
    sketch_dim: int = 256           # JL projection dimension
    value_bits: int = 2             # Bits for value quantization
    value_group_size: int = 32      # Group size for value quantization
    outlier_count: int = 8          # Number of outlier dimensions per head
    outlier_sketch_dim: int = 256  # Sketch dim for outlier detection


def compute_qjl_slot_size(
    head_size: int,
    sketch_dim: int,
    outlier_count: int,
    value_bits: int = 2,
    value_group_size: int = 32,
) -> int:
    """Compute the slot size for packed QJL cache.

    Layout per token:
    - key_hash: sketch_dim // 8 bytes (packed 1-bit per dim)
    - key_outlier_hash: outlier_count // 8 bytes
    - key_norm: 2 bytes (float16)
    - key_outlier_norm: outlier_count * 2 bytes (float16)
    - outlier_idx: outlier_count bytes (uint8 indices)
    - value_pack: head_size * value_bits / 8 bytes
    - value_scale: num_groups * 2 bytes (float16)
    - value_min: num_groups * 2 bytes (float16)
    """
    key_hash_bytes = (sketch_dim + 7) // 8
    key_outlier_bytes = (outlier_count + 7) // 8
    value_pack_bytes = head_size * value_bits // 8
    num_groups = head_size // value_group_size

    return (
        key_hash_bytes +
        key_outlier_bytes +
        2 +
        outlier_count * 2 +
        outlier_count +
        value_pack_bytes +
        num_groups * 2 +
        num_groups * 2
    )


# =============================================================================
# QJL Projection Matrix Generation
# =============================================================================

def make_qjl_projection(
    head_size: int,
    sketch_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the QJL projection pair with orthonormal columns."""
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


# =============================================================================
# Optimized Triton Kernels for QJL
# =============================================================================

@triton.jit
def _qjl_sign_project_kernel(
    x_ptr, proj_ptr, hash_out_ptr, norm_out_ptr,
    stride_x_t, stride_x_h, stride_proj_h,
    stride_hash_t, stride_norm_t,
    HEAD_SIZE: tl.constexpr, SKETCH_DIM: tl.constexpr,
):
    """Project to sketch space and sign-quantize.

    Grid: (num_tokens,)
    Each program handles one token.
    """
    token_id = tl.program_id(0)
    x_base = token_id * stride_x_t

    sketch = tl.zeros((SKETCH_DIM,), dtype=tl.float32)

    for h in range(HEAD_SIZE):
        x_h = tl.load(x_ptr + x_base + h * stride_x_h).to(tl.float32)
        w_base = h * stride_proj_h
        w_vals = tl.load(proj_ptr + w_base + tl.arange(0, SKETCH_DIM)).to(tl.float32)
        sketch = sketch + x_h * w_vals

    sketch_sign = (sketch >= 0).to(tl.int8)
    hash_base = token_id * stride_hash_t
    tl.store(hash_out_ptr + hash_base + tl.arange(0, SKETCH_DIM), sketch_sign.to(tl.uint8))

    x_offs = x_base + tl.arange(0, HEAD_SIZE) * stride_x_h
    x_vals = tl.load(x_ptr + x_offs).to(tl.float32)
    norm_sq = tl.sum(x_vals * x_vals, axis=0)
    norm = tl.sqrt(norm_sq + 1e-6)
    tl.store(norm_out_ptr + token_id * stride_norm_t, norm)


def qjl_sign_project_triton(
    x: torch.Tensor,
    proj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project tensors to QJL sketch space.

    Args:
        x: [batch, kv_heads, num_blocks, block_size, head_size]
        proj: [head_size, sketch_dim] projection matrix

    Returns:
        hashes: [batch, kv_heads, num_blocks, block_size, sketch_dim] sign bits
        norms: [batch, kv_heads, num_blocks, block_size]
    """
    batch, kv_heads, num_blocks, block_size, head_size = x.shape
    sketch_dim = proj.shape[1]

    # Flatten for kernel
    x_flat = x.reshape(-1, head_size)
    num_tokens = x_flat.shape[0]

    hashes = torch.zeros(
        num_tokens, sketch_dim, dtype=torch.uint8, device=x.device
    )
    norms = torch.zeros(num_tokens, dtype=x.dtype, device=x.device)

    _qjl_sign_project_kernel[(num_tokens,)](
        x_flat, proj, hashes, norms,
        x_flat.stride(0), 1,
        proj.stride(0),
        hashes.stride(0), norms.stride(0),
        HEAD_SIZE=head_size, SKETCH_DIM=sketch_dim,
        num_warps=4,
    )

    hashes = hashes.reshape(batch, kv_heads, num_blocks, block_size, sketch_dim)
    norms = norms.reshape(batch, kv_heads, num_blocks, block_size)

    return hashes, norms


@triton.jit
def _qjl_pack_bits_kernel(
    hash_in_ptr, pack_out_ptr,
    stride_h_b, stride_h_h, stride_h_blk, stride_h_tok, stride_h_dim,
    stride_p_b, stride_p_h, stride_p_blk, stride_p_tok, stride_p_byte,
    BATCH: tl.constexpr, KV_HEADS: tl.constexpr,
    HASH_BYTES: tl.constexpr, SKETCH_DIM: tl.constexpr,
):
    """Pack sign bits into bytes.

    Grid: (batch * kv_heads, num_blocks, block_size)
    """
    b_h = tl.program_id(0)
    blk = tl.program_id(1)
    tok = tl.program_id(2)

    if b_h >= BATCH * KV_HEADS:
        return

    # Compute actual batch and head indices from the flattened grid dimension.
    b = b_h // KV_HEADS
    h = b_h % KV_HEADS

    hash_base = (
        b * stride_h_b
        + h * stride_h_h
        + blk * stride_h_blk
        + tok * stride_h_tok
    )

    # Pack SKETCH_DIM bits into HASH_BYTES.
    for byte_i in range(HASH_BYTES):
        byte_val = tl.zeros((), dtype=tl.int32)
        for bit in range(8):
            dim_idx = byte_i * 8 + bit
            if dim_idx < SKETCH_DIM:
                bit_val = tl.load(hash_in_ptr + hash_base + dim_idx).to(tl.int32)
                byte_val = byte_val | (bit_val << bit)
        pack_off = (
            b * stride_p_b
            + h * stride_p_h
            + blk * stride_p_blk
            + tok * stride_p_tok
            + byte_i * stride_p_byte
        )
        tl.store(pack_out_ptr + pack_off, byte_val.to(tl.uint8))


def qjl_pack_bits(
    hashes: torch.Tensor,
) -> torch.Tensor:
    """Pack sign bits into bytes.

    Args:
        hashes:
            [batch, kv_heads, num_blocks, block_size, sketch_dim] uint8, or
            [num_blocks, kv_heads, block_size, sketch_dim] uint8

    Returns:
        packed:
            [batch, kv_heads, num_blocks, block_size, hash_bytes], or
            [num_blocks, kv_heads, block_size, hash_bytes]
    """
    if hashes.ndim == 4:
        hashes_in = hashes.unsqueeze(0)
        squeeze_batch = True
    elif hashes.ndim == 5:
        hashes_in = hashes
        squeeze_batch = False
    else:
        raise ValueError(
            'qjl_pack_bits expects a 4D or 5D tensor, got '
            f'shape={tuple(hashes.shape)}'
        )

    hashes_in = hashes_in.contiguous()
    batch, kv_heads, num_blocks, block_size, sketch_dim = hashes_in.shape
    hash_bytes = (sketch_dim + 7) // 8

    packed = torch.zeros(
        batch, kv_heads, num_blocks, block_size, hash_bytes,
        dtype=torch.uint8, device=hashes_in.device
    )

    _qjl_pack_bits_kernel[(batch * kv_heads, num_blocks, block_size)](
        hashes_in, packed,
        hashes_in.stride(0), hashes_in.stride(1), hashes_in.stride(2),
        hashes_in.stride(3), hashes_in.stride(4),
        packed.stride(0), packed.stride(1), packed.stride(2),
        packed.stride(3), packed.stride(4),
        BATCH=batch, KV_HEADS=kv_heads,
        HASH_BYTES=hash_bytes, SKETCH_DIM=sketch_dim,
        num_warps=1,
    )

    if squeeze_batch:
        return packed.squeeze(0)
    return packed

def qjl_compute_scores_python(
    q_hashes: torch.Tensor,
    k_hashes: torch.Tensor,
    q_norms: torch.Tensor,
    k_norms: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    gqa_ratio: int,
) -> torch.Tensor:
    """Compute QJL attention scores using Python (simple but works).

    Score = popcount(XOR(q_hash, k_hash)) / sketch_dim * scale * norm_ratio
    """
    batch, q_heads, sketch_dim = q_hashes.shape
    _, kv_heads, num_blocks, block_size, _ = k_hashes.shape
    max_tokens = num_blocks * block_size

    scores = torch.zeros(
        batch, q_heads, max_tokens,
        dtype=torch.float32, device=q_hashes.device
    )

    for b in range(batch):
        seq_len = seq_lens[b].item()
        for qh in range(q_heads):
            kh = qh // gqa_ratio
            q_hash = q_hashes[b, qh]  # [sketch_dim]

            for blk in range(num_blocks):
                blk_start = blk * block_size
                if blk_start >= seq_len:
                    break

                physical_block = block_table[b, blk].item()

                for tok in range(block_size):
                    token_idx = blk_start + tok
                    if token_idx >= seq_len:
                        break

                    k_hash = k_hashes[b, kh, physical_block, tok]  # [sketch_dim]

                    # XOR and popcount
                    xor_val = q_hash ^ k_hash
                    matches = (1 - xor_val).sum().item()  # Count matching bits

                    # Score
                    k_norm = k_norms[b, kh, physical_block, tok]
                    q_norm = q_norms[b, qh]
                    norm_ratio = k_norm / (q_norm + 1e-6)
                    score = (matches / sketch_dim) * scale * norm_ratio

                    scores[b, qh, token_idx] = score

    return scores


@triton.jit
def _qjl_score_kernel(
    q_hash_ptr, k_hash_ptr, q_norm_ptr, k_norm_ptr,
    block_table_ptr, seq_lens_ptr, scores_ptr,
    stride_qh_b, stride_qh_d,
    stride_kh_b, stride_kh_h, stride_kh_blk, stride_kh_tok, stride_kh_d,
    stride_qn_b, stride_qn_h,
    stride_kn_b, stride_kn_h, stride_kn_blk, stride_kn_tok,
    stride_bt_b, stride_bt_h,
    stride_s_b, stride_s_h, stride_s_tok,
    GQA_RATIO: tl.constexpr,
    SKETCH_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    attn_scale: tl.constexpr,
):
    """Compute attention scores from QJL hashes using XOR+popcount.

    Grid: (batch, q_heads, num_blocks)
    Each program computes scores for one (batch, q_head) against all tokens in one block.
    """
    b = tl.program_id(0)
    qh = tl.program_id(1)
    blk = tl.program_id(2)
    kh = qh // GQA_RATIO

    seq_len = tl.load(seq_lens_ptr + b)
    physical_block = tl.load(block_table_ptr + b * stride_bt_b + blk * stride_bt_h)

    # Load Q hash - all SKETCH_DIM values.
    q_hash_base = b * stride_qh_b + qh * stride_qh_d
    q_hash = tl.load(
        q_hash_ptr + q_hash_base + tl.arange(0, SKETCH_DIM)
    ).to(tl.int8)

    # Load Q norm
    q_norm = tl.load(q_norm_ptr + b * stride_qn_b + qh * stride_qn_h).to(tl.float32)

    # Compute scores for all tokens in this block
    scores_base = b * stride_s_b + qh * stride_s_h + blk * BLOCK_SIZE * stride_s_tok

    k_hash_base = (b * stride_kh_b + kh * stride_kh_h +
                   physical_block * stride_kh_blk)

    for tok in range(BLOCK_SIZE):
        token_idx = blk * BLOCK_SIZE + tok
        if token_idx < seq_len:
            # Load K hash
            k_off = k_hash_base + tok * stride_kh_tok
            k_hash = tl.load(
                k_hash_ptr + k_off + tl.arange(0, SKETCH_DIM)
            ).to(tl.int8)

            # XOR and popcount (count matching bits)
            xor_val = q_hash ^ k_hash
            matches = tl.sum(1 - xor_val)

            # Load K norm
            k_norm_off = (b * stride_kn_b + kh * stride_kn_h +
                          physical_block * stride_kn_blk + tok * stride_kn_tok)
            k_norm = tl.load(k_norm_ptr + k_norm_off).to(tl.float32)

            # Score with norm normalization
            norm_ratio = k_norm / (q_norm + 1e-6)
            score = (tl.cast(matches, tl.float32) / SKETCH_DIM) * attn_scale * norm_ratio

            tl.store(scores_ptr + scores_base + tok * stride_s_tok, score)


def qjl_compute_scores_triton(
    q_hashes: torch.Tensor,
    k_hashes: torch.Tensor,
    q_norms: torch.Tensor,
    k_norms: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    gqa_ratio: int,
) -> torch.Tensor:
    """Compute QJL attention scores using Triton.

    Args:
        q_hashes: [batch, q_heads, sketch_dim] sign bits
        k_hashes: [batch, kv_heads, num_blocks, block_size, sketch_dim] sign bits
        q_norms: [batch, q_heads]
        k_norms: [batch, kv_heads, num_blocks, block_size]
        block_table: [batch, max_blocks]
        seq_lens: [batch]
        scale: attention scale
        gqa_ratio: Q heads per KV head

    Returns:
        scores: [batch, q_heads, max_tokens]
    """
    batch, q_heads, sketch_dim = q_hashes.shape
    _, kv_heads, num_blocks, block_size, _ = k_hashes.shape
    max_tokens = num_blocks * block_size

    scores = torch.zeros(
        batch, q_heads, max_tokens,
        dtype=torch.float32, device=q_hashes.device
    )

    # Contiguous strides for efficient access
    q_hashes_c = q_hashes.contiguous()
    k_hashes_c = k_hashes.contiguous()

    q_norms_c = q_norms.contiguous()
    k_norms_c = k_norms.contiguous()
    block_table_c = block_table.contiguous()

    _qjl_score_kernel[(batch, q_heads, num_blocks)](
        q_hashes_c, k_hashes_c, q_norms_c, k_norms_c,
        block_table_c, seq_lens, scores,
        q_hashes_c.stride(0), q_hashes_c.stride(1),
        k_hashes_c.stride(0), k_hashes_c.stride(1), k_hashes_c.stride(2), k_hashes_c.stride(3), k_hashes_c.stride(4),
        q_norms_c.stride(0), q_norms_c.stride(1),
        k_norms_c.stride(0), k_norms_c.stride(1), k_norms_c.stride(2), k_norms_c.stride(3),
        block_table_c.stride(0), block_table_c.stride(1),
        scores.stride(0), scores.stride(1), scores.stride(2),
        GQA_RATIO=gqa_ratio,
        SKETCH_DIM=sketch_dim,
        BLOCK_SIZE=block_size,
        attn_scale=scale,
        num_warps=4,
    )

    return scores


@triton.jit
def _qjl_stage12_score_kernel(
    query_ptr,
    q_hash_ptr,
    q_norm_ptr,
    key_pack_ptr,
    key_scale_ptr,
    residual_hash_ptr,
    residual_norm_ptr,
    seq_lens_ptr,
    total_tokens,
    scores_ptr,
    stride_q_b, stride_q_h, stride_q_d,
    stride_qhash_b, stride_qhash_h, stride_qhash_byte,
    stride_qnorm_b, stride_qnorm_h,
    stride_kp_b, stride_kp_h, stride_kp_t, stride_kp_byte,
    stride_ks_b, stride_ks_h, stride_ks_t,
    stride_rh_b, stride_rh_h, stride_rh_t, stride_rh_byte,
    stride_rn_b, stride_rn_h, stride_rn_t,
    stride_s_b, stride_s_h, stride_s_t,
    GQA_RATIO: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    SKETCH_DIM: tl.constexpr,
    HASH_BYTES: tl.constexpr,
    PACK_BYTES: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    SCORE_SCALE: tl.constexpr,
):
    """Stage-1 2-bit coarse score + stage-2 packed residual correction.

    Grid: (batch, q_heads, ceil_div(total_tokens, BLOCK_TOKENS))
    """
    b = tl.program_id(0)
    qh = tl.program_id(1)
    tok_blk = tl.program_id(2)
    kh = qh // GQA_RATIO

    tok_offsets = tok_blk * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    tok_in_bounds = tok_offsets < total_tokens
    seq_len = tl.load(seq_lens_ptr + b)
    tok_mask = tok_offsets < seq_len

    q_base = b * stride_q_b + qh * stride_q_h
    qhash_base = b * stride_qhash_b + qh * stride_qhash_h
    q_norm = tl.load(q_norm_ptr + b * stride_qnorm_b + qh * stride_qnorm_h).to(
        tl.float32
    )

    coarse = tl.zeros((BLOCK_TOKENS,), dtype=tl.float32)
    mismatches = tl.zeros((BLOCK_TOKENS,), dtype=tl.int32)
    key_scale = tl.load(
        key_scale_ptr
        + b * stride_ks_b
        + kh * stride_ks_h
        + tok_offsets * stride_ks_t,
        mask=tok_in_bounds,
        other=0,
    ).to(tl.float32)

    for byte_i in range(PACK_BYTES):
        packed = tl.load(
            key_pack_ptr
            + b * stride_kp_b
            + kh * stride_kp_h
            + tok_offsets * stride_kp_t
            + byte_i * stride_kp_byte,
            mask=tok_in_bounds,
            other=0,
        ).to(tl.int32)

        q0 = tl.load(query_ptr + q_base + (byte_i * 4 + 0) * stride_q_d).to(tl.float32)
        q1 = tl.load(query_ptr + q_base + (byte_i * 4 + 1) * stride_q_d).to(tl.float32)
        q2 = tl.load(query_ptr + q_base + (byte_i * 4 + 2) * stride_q_d).to(tl.float32)
        q3 = tl.load(query_ptr + q_base + (byte_i * 4 + 3) * stride_q_d).to(tl.float32)

        v0 = (((packed >> 0) & 0x3).to(tl.float32) / 1.5 - 1.0) * key_scale
        v1 = (((packed >> 2) & 0x3).to(tl.float32) / 1.5 - 1.0) * key_scale
        v2 = (((packed >> 4) & 0x3).to(tl.float32) / 1.5 - 1.0) * key_scale
        v3 = (((packed >> 6) & 0x3).to(tl.float32) / 1.5 - 1.0) * key_scale

        coarse += q0 * v0 + q1 * v1 + q2 * v2 + q3 * v3

    for byte_i in range(HASH_BYTES):
        q_byte = tl.load(q_hash_ptr + qhash_base + byte_i * stride_qhash_byte).to(tl.int32)
        k_byte = tl.load(
            residual_hash_ptr
            + b * stride_rh_b
            + kh * stride_rh_h
            + tok_offsets * stride_rh_t
            + byte_i * stride_rh_byte,
            mask=tok_in_bounds,
            other=0,
        ).to(tl.int32)
        x = q_byte ^ k_byte
        x = x - ((x >> 1) & 0x55)
        x = (x & 0x33) + ((x >> 2) & 0x33)
        x = (x + (x >> 4)) & 0x0F
        mismatches += x

    residual_norm = tl.load(
        residual_norm_ptr
        + b * stride_rn_b
        + kh * stride_rn_h
        + tok_offsets * stride_rn_t,
        mask=tok_in_bounds,
        other=0,
    ).to(tl.float32)
    centered = (
        (SKETCH_DIM - 2.0 * mismatches.to(tl.float32)) / SKETCH_DIM
    )
    score = (coarse + centered * q_norm * residual_norm) * SCORE_SCALE
    score = tl.where(tok_mask, score, float("-inf"))

    tl.store(
        scores_ptr
        + b * stride_s_b
        + qh * stride_s_h
        + tok_offsets * stride_s_t,
        score,
        mask=tok_in_bounds,
    )


@triton.jit
def _qjl_stage12_score_multi_query_kernel(
    query_ptr,
    q_hash_ptr,
    q_norm_ptr,
    key_pack_ptr,
    key_scale_ptr,
    residual_hash_ptr,
    residual_norm_ptr,
    seq_len,
    total_tokens,
    scores_ptr,
    stride_q_t, stride_q_h, stride_q_d,
    stride_qhash_t, stride_qhash_h, stride_qhash_byte,
    stride_qnorm_t, stride_qnorm_h,
    stride_kp_h, stride_kp_t, stride_kp_byte,
    stride_ks_h, stride_ks_t,
    stride_rh_h, stride_rh_t, stride_rh_byte,
    stride_rn_h, stride_rn_t,
    stride_s_t, stride_s_h, stride_s_tok,
    GQA_RATIO: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    SKETCH_DIM: tl.constexpr,
    HASH_BYTES: tl.constexpr,
    PACK_BYTES: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    SCORE_SCALE: tl.constexpr,
):
    """Stage-1/2 fused scorer for many query tokens against one shared prefix.

    Grid: (q_len, q_heads, ceil_div(total_tokens, BLOCK_TOKENS))
    """
    q_tok = tl.program_id(0)
    qh = tl.program_id(1)
    tok_blk = tl.program_id(2)
    kh = qh // GQA_RATIO

    tok_offsets = tok_blk * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    tok_in_bounds = tok_offsets < total_tokens
    tok_mask = tok_offsets < seq_len

    q_base = q_tok * stride_q_t + qh * stride_q_h
    qhash_base = q_tok * stride_qhash_t + qh * stride_qhash_h
    q_norm = tl.load(q_norm_ptr + q_tok * stride_qnorm_t + qh * stride_qnorm_h).to(
        tl.float32
    )

    coarse = tl.zeros((BLOCK_TOKENS,), dtype=tl.float32)
    mismatches = tl.zeros((BLOCK_TOKENS,), dtype=tl.int32)
    key_scale = tl.load(
        key_scale_ptr
        + kh * stride_ks_h
        + tok_offsets * stride_ks_t,
        mask=tok_in_bounds,
        other=0,
    ).to(tl.float32)

    for byte_i in range(PACK_BYTES):
        packed = tl.load(
            key_pack_ptr
            + kh * stride_kp_h
            + tok_offsets * stride_kp_t
            + byte_i * stride_kp_byte,
            mask=tok_in_bounds,
            other=0,
        ).to(tl.int32)

        q0 = tl.load(query_ptr + q_base + (byte_i * 4 + 0) * stride_q_d).to(tl.float32)
        q1 = tl.load(query_ptr + q_base + (byte_i * 4 + 1) * stride_q_d).to(tl.float32)
        q2 = tl.load(query_ptr + q_base + (byte_i * 4 + 2) * stride_q_d).to(tl.float32)
        q3 = tl.load(query_ptr + q_base + (byte_i * 4 + 3) * stride_q_d).to(tl.float32)

        v0 = (((packed >> 0) & 0x3).to(tl.float32) / 1.5 - 1.0) * key_scale
        v1 = (((packed >> 2) & 0x3).to(tl.float32) / 1.5 - 1.0) * key_scale
        v2 = (((packed >> 4) & 0x3).to(tl.float32) / 1.5 - 1.0) * key_scale
        v3 = (((packed >> 6) & 0x3).to(tl.float32) / 1.5 - 1.0) * key_scale

        coarse += q0 * v0 + q1 * v1 + q2 * v2 + q3 * v3

    for byte_i in range(HASH_BYTES):
        q_byte = tl.load(q_hash_ptr + qhash_base + byte_i * stride_qhash_byte).to(tl.int32)
        k_byte = tl.load(
            residual_hash_ptr
            + kh * stride_rh_h
            + tok_offsets * stride_rh_t
            + byte_i * stride_rh_byte,
            mask=tok_in_bounds,
            other=0,
        ).to(tl.int32)
        x = q_byte ^ k_byte
        x = x - ((x >> 1) & 0x55)
        x = (x & 0x33) + ((x >> 2) & 0x33)
        x = (x + (x >> 4)) & 0x0F
        mismatches += x

    residual_norm = tl.load(
        residual_norm_ptr
        + kh * stride_rn_h
        + tok_offsets * stride_rn_t,
        mask=tok_in_bounds,
        other=0,
    ).to(tl.float32)
    centered = (SKETCH_DIM - 2.0 * mismatches.to(tl.float32)) / SKETCH_DIM
    score = (coarse + centered * q_norm * residual_norm) * SCORE_SCALE
    score = tl.where(tok_mask, score, float("-inf"))

    tl.store(
        scores_ptr
        + q_tok * stride_s_t
        + qh * stride_s_h
        + tok_offsets * stride_s_tok,
        score,
        mask=tok_in_bounds,
    )


def qjl_score_stage12_triton(
    query: torch.Tensor,
    q_hash_packed: torch.Tensor,
    q_norms: torch.Tensor,
    key_pack: torch.Tensor,
    key_scale: torch.Tensor,
    residual_hash_packed: torch.Tensor,
    residual_norm: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    gqa_ratio: int,
    scale: float,
    sketch_dim: int,
) -> torch.Tensor:
    """Fused stage-1/stage-2 score computation on packed cache tensors."""
    batch, q_heads, head_size = query.shape
    _, _, total_tokens, pack_bytes = key_pack.shape
    hash_bytes = q_hash_packed.shape[-1]

    scores = torch.empty(
        batch, q_heads, total_tokens,
        dtype=torch.float32,
        device=query.device,
    )

    query_c = query.contiguous()
    q_hash_c = q_hash_packed.contiguous()
    q_norms_c = q_norms.contiguous()
    key_pack_c = key_pack.contiguous()
    key_scale_c = key_scale.contiguous()
    residual_hash_c = residual_hash_packed.contiguous()
    residual_norm_c = residual_norm.contiguous()
    seq_lens_c = seq_lens.contiguous()

    block_tokens = 32
    grid = (batch, q_heads, triton.cdiv(total_tokens, block_tokens))
    _qjl_stage12_score_kernel[grid](
        query_c,
        q_hash_c,
        q_norms_c,
        key_pack_c,
        key_scale_c,
        residual_hash_c,
        residual_norm_c,
        seq_lens_c,
        total_tokens,
        scores,
        query_c.stride(0), query_c.stride(1), query_c.stride(2),
        q_hash_c.stride(0), q_hash_c.stride(1), q_hash_c.stride(2),
        q_norms_c.stride(0), q_norms_c.stride(1),
        key_pack_c.stride(0), key_pack_c.stride(1), key_pack_c.stride(2), key_pack_c.stride(3),
        key_scale_c.stride(0), key_scale_c.stride(1), key_scale_c.stride(2),
        residual_hash_c.stride(0), residual_hash_c.stride(1), residual_hash_c.stride(2), residual_hash_c.stride(3),
        residual_norm_c.stride(0), residual_norm_c.stride(1), residual_norm_c.stride(2),
        scores.stride(0), scores.stride(1), scores.stride(2),
        GQA_RATIO=gqa_ratio,
        HEAD_SIZE=head_size,
        SKETCH_DIM=sketch_dim,
        HASH_BYTES=hash_bytes,
        PACK_BYTES=pack_bytes,
        BLOCK_TOKENS=block_tokens,
        SCORE_SCALE=scale,
        num_warps=4,
        num_stages=1,
    )

    return scores


def qjl_score_stage12_multi_query_triton(
    query: torch.Tensor,
    q_hash_packed: torch.Tensor,
    q_norms: torch.Tensor,
    key_pack: torch.Tensor,
    key_scale: torch.Tensor,
    residual_hash_packed: torch.Tensor,
    residual_norm: torch.Tensor,
    *,
    seq_len: int,
    gqa_ratio: int,
    scale: float,
    sketch_dim: int,
) -> torch.Tensor:
    """Fused stage-1/stage-2 score for many query tokens vs one shared prefix."""
    q_len, q_heads, head_size = query.shape
    _, total_tokens, pack_bytes = key_pack.shape
    hash_bytes = q_hash_packed.shape[-1]

    scores = torch.empty(
        q_len, q_heads, total_tokens,
        dtype=torch.float32,
        device=query.device,
    )

    query_c = query.contiguous()
    q_hash_c = q_hash_packed.contiguous()
    q_norms_c = q_norms.contiguous()
    key_pack_c = key_pack.contiguous()
    key_scale_c = key_scale.contiguous()
    residual_hash_c = residual_hash_packed.contiguous()
    residual_norm_c = residual_norm.contiguous()

    block_tokens = 32
    grid = (q_len, q_heads, triton.cdiv(total_tokens, block_tokens))
    _qjl_stage12_score_multi_query_kernel[grid](
        query_c,
        q_hash_c,
        q_norms_c,
        key_pack_c,
        key_scale_c,
        residual_hash_c,
        residual_norm_c,
        seq_len,
        total_tokens,
        scores,
        query_c.stride(0), query_c.stride(1), query_c.stride(2),
        q_hash_c.stride(0), q_hash_c.stride(1), q_hash_c.stride(2),
        q_norms_c.stride(0), q_norms_c.stride(1),
        key_pack_c.stride(0), key_pack_c.stride(1), key_pack_c.stride(2),
        key_scale_c.stride(0), key_scale_c.stride(1),
        residual_hash_c.stride(0), residual_hash_c.stride(1), residual_hash_c.stride(2),
        residual_norm_c.stride(0), residual_norm_c.stride(1),
        scores.stride(0), scores.stride(1), scores.stride(2),
        GQA_RATIO=gqa_ratio,
        HEAD_SIZE=head_size,
        SKETCH_DIM=sketch_dim,
        HASH_BYTES=hash_bytes,
        PACK_BYTES=pack_bytes,
        BLOCK_TOKENS=block_tokens,
        SCORE_SCALE=scale,
        num_warps=4,
        num_stages=1,
    )
    return scores


@triton.jit
def _qjl_local_causal_attn_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    lse_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_lse_t,
    stride_lse_h,
    kv_len,
    q_start,
    GQA_RATIO: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    MAX_K_BLOCKS: tl.constexpr,
    SCORE_SCALE: tl.constexpr,
):
    """Fused local causal attention with online softmax.

    Grid: (q_slice_len, q_heads)
    """
    q_local = tl.program_id(0)
    qh = tl.program_id(1)
    kh = qh // GQA_RATIO
    q_tok = q_start + q_local

    d_offsets = tl.arange(0, HEAD_SIZE)
    q_ptr = query_ptr + q_local * stride_q_t + qh * stride_q_h + d_offsets * stride_q_d
    q_vec = tl.load(q_ptr).to(tl.float32)

    m = tl.full((), -float("inf"), dtype=tl.float32)
    l = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((HEAD_SIZE,), dtype=tl.float32)

    for k_blk in range(MAX_K_BLOCKS):
        k_offsets = k_blk * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        valid = (k_offsets < kv_len) & (k_offsets <= q_tok)
        valid_exp = valid[:, None]

        k_ptr = (
            key_ptr
            + k_offsets[:, None] * stride_k_t
            + kh * stride_k_h
            + d_offsets[None, :] * stride_k_d
        )
        v_ptr = (
            value_ptr
            + k_offsets[:, None] * stride_v_t
            + kh * stride_v_h
            + d_offsets[None, :] * stride_v_d
        )
        k_block = tl.load(k_ptr, mask=valid_exp, other=0).to(tl.float32)
        v_block = tl.load(v_ptr, mask=valid_exp, other=0).to(tl.float32)

        scores = tl.sum(k_block * q_vec[None, :], axis=1) * SCORE_SCALE
        scores = tl.where(valid, scores, -float("inf"))
        block_m = tl.max(scores, axis=0)
        m_new = tl.maximum(m, block_m)
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new)

        acc = acc * alpha + tl.sum(p[:, None] * v_block, axis=0)
        l = l * alpha + tl.sum(p, axis=0)
        m = m_new

    out_vec = acc / l
    lse = m + tl.log(l)

    o_ptr = output_ptr + q_local * stride_o_t + qh * stride_o_h + d_offsets * stride_o_d
    tl.store(o_ptr, out_vec)
    tl.store(lse_ptr + q_local * stride_lse_t + qh * stride_lse_h, lse)


def qjl_local_causal_attn_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    gqa_ratio: int,
    scale: float,
    q_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local causal attention output and per-row logsumexp.

    Args:
        query: [q_len, q_heads, head_size]
        key: [q_len, kv_heads, head_size]
        value: [q_len, kv_heads, head_size]
    Returns:
        output: [q_len, q_heads, head_size]
        lse: [q_len, q_heads] (float32)
    """
    q_slice_len, q_heads, head_size = query.shape
    kv_len = key.shape[0]
    assert value.shape[0] == kv_len

    query_c = query.contiguous()
    key_c = key.contiguous()
    value_c = value.contiguous()

    output = torch.empty_like(query_c)
    lse = torch.empty(
        (q_slice_len, q_heads), dtype=torch.float32, device=query.device
    )

    block_tokens = 32
    max_k_blocks = triton.cdiv(kv_len, block_tokens)
    grid = (q_slice_len, q_heads)
    _qjl_local_causal_attn_kernel[grid](
        query_c,
        key_c,
        value_c,
        output,
        lse,
        query_c.stride(0), query_c.stride(1), query_c.stride(2),
        key_c.stride(0), key_c.stride(1), key_c.stride(2),
        value_c.stride(0), value_c.stride(1), value_c.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        kv_len,
        q_start,
        GQA_RATIO=gqa_ratio,
        HEAD_SIZE=head_size,
        BLOCK_TOKENS=block_tokens,
        MAX_K_BLOCKS=max_k_blocks,
        SCORE_SCALE=scale,
        num_warps=4,
        num_stages=1,
    )
    return output, lse


# =============================================================================
# Value Quantization Kernels
# =============================================================================

@triton.jit
def _quantize_values_2bit_kernel(
    values_ptr, scale_out_ptr, min_out_ptr, pack_out_ptr,
    stride_v_b, stride_v_h, stride_v_tok, stride_v_d,
    stride_s_b, stride_s_h, stride_s_tok, stride_s_g,
    stride_m_b, stride_m_h, stride_m_tok, stride_m_g,
    stride_p_b, stride_p_h, stride_p_tok, stride_p_byte,
    HEAD_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
):
    """Quantize values to 2-bit with per-group scale and min.

    Grid: (batch, kv_heads, num_tokens)
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    tok = tl.program_id(2)

    num_groups = HEAD_SIZE // GROUP_SIZE

    # Load values
    v_base = b * stride_v_b + h * stride_v_h + tok * stride_v_tok
    v_vals = tl.load(values_ptr + v_base + tl.arange(0, HEAD_SIZE)).to(tl.float32)

    # Quantize per group
    for g in range(num_groups):
        g_start = g * GROUP_SIZE
        g_vals = v_vals[g_start:g_start + GROUP_SIZE]

        g_min = tl.min(g_vals, axis=0)
        g_max = tl.max(g_vals, axis=0)
        scale = (g_max - g_min) / 3.0
        g_min_val = tl.cast(g_min, tl.float32)

        # Store scale and min
        s_off = b * stride_s_b + h * stride_s_h + tok * stride_s_tok + g * stride_s_g
        m_off = b * stride_m_b + h * stride_m_h + tok * stride_m_tok + g * stride_m_g
        tl.store(scale_out_ptr + s_off, scale)
        tl.store(min_out_ptr + m_off, g_min_val)

        # Quantize and pack
        if scale > 1e-8:
            qvals = ((g_vals - g_min) / scale).round().to(tl.int32).clamp(0, 3)
        else:
            qvals = tl.zeros((GROUP_SIZE,), dtype=tl.int32)

        # Pack 4 2-bit values per byte
        for byte_i in range(VALS_PER_BYTE):
            byte_val = 0
            for bit in range(4):
                idx = byte_i * 4 + bit
                if idx < GROUP_SIZE:
                    byte_val |= (qvals[idx] & 0x3) << (bit * 2)
            p_off = b * stride_p_b + h * stride_p_h + tok * stride_p_tok + byte_i * stride_p_byte
            tl.store(pack_out_ptr + p_off, byte_val.to(tl.uint8))


def quantize_values_2bit(
    values: torch.Tensor,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize values to 2-bit using PyTorch.

    Args:
        values: [..., head_size]
        group_size: quantization group size

    Returns:
        value_pack: [..., head_size // 4] uint8 packed values
        value_scale: [..., num_groups]
        value_min: [..., num_groups]
    """
    *outer_shape, head_size = values.shape
    num_groups = head_size // group_size
    vals_per_byte = 4  # 4 values per byte for 2-bit
    pack_bytes = (head_size + 3) // 4

    values_flat = values.reshape(-1, head_size)
    num_tokens = values_flat.shape[0]

    pack = torch.zeros(
        num_tokens, pack_bytes, dtype=torch.uint8, device=values.device
    )
    scale = torch.zeros(
        num_tokens, num_groups, dtype=torch.float32, device=values.device
    )
    vmin = torch.zeros(
        num_tokens, num_groups, dtype=torch.float32, device=values.device
    )

    for t in range(num_tokens):
        v = values_flat[t]
        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, head_size)
            group = v[start:end]

            gmin = group.min()
            gmax = group.max()
            scale_val = (gmax - gmin) / 3.0

            vmin[t, g] = gmin
            scale[t, g] = scale_val

            if scale_val > 1e-8:
                qvals = ((group - gmin) / scale_val).round().clamp(0, 3).to(torch.uint8)
            else:
                qvals = torch.zeros(end - start, dtype=torch.uint8, device=values.device)

            for i in range(0, end - start, 4):
                byte_val = (qvals[i].item() & 0x3)
                if i + 1 < end - start:
                    byte_val |= (qvals[i + 1].item() & 0x3) << 2
                if i + 2 < end - start:
                    byte_val |= (qvals[i + 2].item() & 0x3) << 4
                if i + 3 < end - start:
                    byte_val |= (qvals[i + 3].item() & 0x3) << 6
                byte_idx = (start + i) // 4
                pack[t, byte_idx] = byte_val

    if outer_shape:
        pack = pack.view(*outer_shape, pack_bytes)
        scale = scale.view(*outer_shape, num_groups)
        vmin = vmin.view(*outer_shape, num_groups)

    return pack, scale, vmin


@triton.jit
def _qjl_weighted_sum_kernel(
    weights_ptr, value_pack_ptr, value_scale_ptr, value_min_ptr,
    block_table_ptr, seq_lens_ptr, output_ptr,
    stride_w_b, stride_w_h, stride_w_tok,
    stride_vp_b, stride_vp_h, stride_vp_blk, stride_vp_tok, stride_vp_byte,
    stride_vs_b, stride_vs_h, stride_vs_blk, stride_vs_tok, stride_vs_g,
    stride_vm_b, stride_vm_h, stride_vm_blk, stride_vm_tok, stride_vm_g,
    stride_bt_b, stride_bt_h,
    stride_o_b, stride_o_h, stride_o_d,
    GQA_RATIO: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    """Compute weighted sum of dequantized values.

    Grid: (batch, q_heads)
    """
    b = tl.program_id(0)
    qh = tl.program_id(1)
    kh = qh // GQA_RATIO

    seq_len = tl.load(seq_lens_ptr + b)
    o_base = b * stride_o_b + qh * stride_o_h

    # Accumulate and store one quantization group at a time.
    for g in range(NUM_GROUPS):
        g_offsets = g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        byte_idx = g_offsets // 4
        bit_shift = (g_offsets % 4) * 2
        group_acc = tl.zeros((GROUP_SIZE,), dtype=tl.float32)

        for blk in range(MAX_BLOCKS):
            blk_start = blk * BLOCK_SIZE
            if blk_start < seq_len:
                physical_block = tl.load(
                    block_table_ptr + b * stride_bt_b + blk * stride_bt_h
                )

                for tok in range(BLOCK_SIZE):
                    token_idx = blk_start + tok
                    if token_idx < seq_len:
                        w = tl.load(
                            weights_ptr
                            + b * stride_w_b
                            + qh * stride_w_h
                            + token_idx * stride_w_tok
                        )

                        vp_base = (
                            b * stride_vp_b
                            + kh * stride_vp_h
                            + physical_block * stride_vp_blk
                            + tok * stride_vp_tok
                        )
                        vs_base = (
                            b * stride_vs_b
                            + kh * stride_vs_h
                            + physical_block * stride_vs_blk
                            + tok * stride_vs_tok
                        )
                        vm_base = (
                            b * stride_vm_b
                            + kh * stride_vm_h
                            + physical_block * stride_vm_blk
                            + tok * stride_vm_tok
                        )

                        scale = tl.load(
                            value_scale_ptr + vs_base + g * stride_vs_g
                        ).to(tl.float32)
                        min_val = tl.load(
                            value_min_ptr + vm_base + g * stride_vm_g
                        ).to(tl.float32)
                        qbytes = tl.load(
                            value_pack_ptr
                            + vp_base
                            + byte_idx * stride_vp_byte
                        )
                        qvals = (qbytes >> bit_shift) & 0x3
                        group_acc += w * (
                            tl.cast(qvals, tl.float32) * scale + min_val
                        )

        tl.store(
            output_ptr + o_base + g_offsets * stride_o_d,
            group_acc.to(tl.float16),
        )


def qjl_weighted_sum_triton(
    weights: torch.Tensor,
    value_pack: torch.Tensor,
    value_scale: torch.Tensor,
    value_min: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    gqa_ratio: int,
) -> torch.Tensor:
    """Compute weighted sum of dequantized values using Triton.

    Args:
        weights: [batch, q_heads, max_tokens] attention weights
        value_pack: [batch, kv_heads, num_blocks, block_size, pack_bytes]
        value_scale: [batch, kv_heads, num_blocks, block_size, num_groups]
        value_min: [batch, kv_heads, num_blocks, block_size, num_groups]
        block_table: [batch, max_blocks]
        seq_lens: [batch]
        gqa_ratio: Q heads per KV head

    Returns:
        output: [batch, q_heads, head_size]
    """
    batch, q_heads, max_tokens = weights.shape
    _, kv_heads, num_blocks, block_size, pack_bytes = value_pack.shape
    num_groups = value_scale.shape[-1]
    head_size = pack_bytes * 4  # 2-bit values: 4 elements per packed byte.
    group_size = head_size // num_groups
    pack_bytes = value_pack.shape[-1]

    output = torch.zeros(
        batch, q_heads, head_size,
        dtype=torch.float16, device=weights.device
    )

    weights_c = weights.contiguous()
    value_pack_c = value_pack.contiguous()
    value_scale_c = value_scale.contiguous()
    value_min_c = value_min.contiguous()
    block_table_c = block_table.contiguous()

    _qjl_weighted_sum_kernel[(batch, q_heads)](
        weights_c, value_pack_c, value_scale_c, value_min_c,
        block_table_c, seq_lens, output,
        weights_c.stride(0), weights_c.stride(1), weights_c.stride(2),
        value_pack_c.stride(0), value_pack_c.stride(1), value_pack_c.stride(2),
        value_pack_c.stride(3), value_pack_c.stride(4),
        value_scale_c.stride(0), value_scale_c.stride(1), value_scale_c.stride(2),
        value_scale_c.stride(3), value_scale_c.stride(4),
        value_min_c.stride(0), value_min_c.stride(1), value_min_c.stride(2),
        value_min_c.stride(3), value_min_c.stride(4),
        block_table_c.stride(0), block_table_c.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        GQA_RATIO=gqa_ratio,
        MAX_BLOCKS=num_blocks,
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        GROUP_SIZE=group_size,
        NUM_GROUPS=num_groups,
        num_warps=4,
    )

    return output


def qjl_weighted_sum_paged_triton(
    weights: torch.Tensor,
    value_pack: torch.Tensor,
    value_scale: torch.Tensor,
    value_min: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    gqa_ratio: int,
) -> torch.Tensor:
    """Compute weighted sum from a shared paged cache using Triton.

    Args:
        weights: [batch, q_heads, max_tokens] attention weights
        value_pack: [num_blocks, kv_heads, block_size, pack_bytes] uint8
        value_scale: [num_blocks, kv_heads, block_size, num_groups]
        value_min: [num_blocks, kv_heads, block_size, num_groups]
        block_table: [batch, max_blocks]
        seq_lens: [batch]
        gqa_ratio: Q heads per KV head

    Returns:
        output: [batch, q_heads, head_size]
    """
    batch, q_heads, _ = weights.shape
    num_blocks, kv_heads, block_size, pack_bytes = value_pack.shape
    num_groups = value_scale.shape[-1]
    head_size = pack_bytes * 4
    group_size = head_size // num_groups
    max_blocks = block_table.shape[1]

    output = torch.zeros(
        batch, q_heads, head_size,
        dtype=torch.float16, device=weights.device
    )

    weights_c = weights.contiguous()
    value_pack_c = value_pack.contiguous()
    value_scale_c = value_scale.contiguous()
    value_min_c = value_min.contiguous()
    block_table_c = block_table.contiguous()

    _qjl_weighted_sum_kernel[(batch, q_heads)](
        weights_c, value_pack_c, value_scale_c, value_min_c,
        block_table_c, seq_lens, output,
        weights_c.stride(0), weights_c.stride(1), weights_c.stride(2),
        0, value_pack_c.stride(1), value_pack_c.stride(0),
        value_pack_c.stride(2), value_pack_c.stride(3),
        0, value_scale_c.stride(1), value_scale_c.stride(0),
        value_scale_c.stride(2), value_scale_c.stride(3),
        0, value_min_c.stride(1), value_min_c.stride(0),
        value_min_c.stride(2), value_min_c.stride(3),
        block_table_c.stride(0), block_table_c.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        GQA_RATIO=gqa_ratio,
        MAX_BLOCKS=max_blocks,
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        GROUP_SIZE=group_size,
        NUM_GROUPS=num_groups,
        num_warps=4,
    )

    return output


def qjl_weighted_sum_pytorch(
    weights: torch.Tensor,
    value_pack: torch.Tensor,
    value_scale: torch.Tensor,
    value_min: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    gqa_ratio: int,
) -> torch.Tensor:
    """Compute weighted sum of dequantized values using PyTorch.

    Args:
        weights: [batch, q_heads, max_tokens] attention weights
        value_pack: [batch, kv_heads, num_blocks, block_size, pack_bytes]
        value_scale: [batch, kv_heads, num_blocks, block_size, num_groups]
        value_min: [batch, kv_heads, num_blocks, block_size, num_groups]
        block_table: [batch, max_blocks]
        seq_lens: [batch]
        gqa_ratio: Q heads per KV head

    Returns:
        output: [batch, q_heads, head_size]
    """
    batch, q_heads, max_tokens = weights.shape
    _, kv_heads, num_blocks, block_size, num_groups = value_scale.shape
    # head_size is derived from pack_bytes: pack_bytes = head_size // 4 for 2-bit
    pack_bytes = value_pack.shape[-1]
    head_size = pack_bytes * 4  # 2-bit = 4 values per byte
    group_size = head_size // num_groups

    output = torch.zeros(
        batch, q_heads, head_size,
        dtype=torch.float32, device=weights.device
    )

    for b in range(batch):
        seq_len = seq_lens[b].item()
        for qh in range(q_heads):
            kh = qh // gqa_ratio
            for blk in range(num_blocks):
                blk_start = blk * block_size
                if blk_start >= seq_len:
                    break

                physical_block = block_table[b, blk].item()

                for tok in range(block_size):
                    token_idx = blk_start + tok
                    if token_idx >= seq_len:
                        break

                    w = weights[b, qh, token_idx].item()

                    # Get value data for this token
                    vp = value_pack[b, kh, physical_block, tok]  # [pack_bytes]
                    vs = value_scale[b, kh, physical_block, tok]  # [num_groups]
                    vm = value_min[b, kh, physical_block, tok]  # [num_groups]

                    # Dequantize and accumulate
                    for g in range(num_groups):
                        scale = vs[g].item()
                        min_val = vm[g].item()
                        g_start = g * group_size

                        for i in range(group_size):
                            dim_idx = g_start + i
                            if dim_idx >= head_size:
                                break

                            byte_idx = dim_idx // 4
                            bit_shift = (dim_idx % 4) * 2
                            qval = (vp[byte_idx].item() >> bit_shift) & 0x3
                            val = qval * scale + min_val
                            output[b, qh, dim_idx] += w * val

    return output.to(torch.float16)


# =============================================================================
# Outlier Detection
# =============================================================================

def detect_outliers_per_head(
    keys: torch.Tensor,
    outlier_count: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detect outlier dimensions per head using mean absolute activation.

    Args:
        keys: [batch, kv_heads, num_tokens, head_size]
        outlier_count: number of outlier dimensions per head

    Returns:
        outlier_indices: [batch, kv_heads, outlier_count]
        outlier_values: [batch, kv_heads, outlier_count] (mean abs values)
    """
    batch, kv_heads, num_tokens, head_size = keys.shape
    device = keys.device

    mean_abs = keys.abs().mean(dim=2)
    outlier_indices = torch.argsort(mean_abs, dim=-1, descending=True)[..., :outlier_count]

    batch_idx = torch.arange(batch, device=device).view(batch, 1, 1).expand(-1, kv_heads, outlier_count)
    kh_idx = torch.arange(kv_heads, device=device).view(1, kv_heads, 1).expand(batch, -1, outlier_count)
    outlier_values = mean_abs[batch_idx, kh_idx, outlier_indices]

    return outlier_indices, outlier_values


# =============================================================================
# Full QJL Attention Forward
# =============================================================================

def qjl_attention_forward(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    proj_dir_score: torch.Tensor,
    proj_dir_quant: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    config: QJLConfig,
) -> torch.Tensor:
    """Full QJL attention forward pass.

    Args:
        queries: [batch, q_heads, head_size]
        keys: [batch, kv_heads, num_blocks, block_size, head_size]
        values: same shape as keys
        proj_dir_score: [head_size, sketch_dim]
        proj_dir_quant: [sketch_dim, head_size]
        block_table: [batch, max_blocks]
        seq_lens: [batch]
        scale: attention scale
        config: QJLConfig

    Returns:
        outputs: [batch, q_heads, head_size]
    """
    batch, q_heads, head_size = queries.shape
    _, kv_heads, num_blocks, block_size, _ = keys.shape
    gqa_ratio = q_heads // kv_heads

    # Step 1: Detect outliers per head
    keys_flat = keys.reshape(batch, kv_heads, num_blocks * block_size, head_size)
    outlier_indices, outlier_values = detect_outliers_per_head(keys_flat, config.outlier_count)

    # Step 2: Project queries to sketch space
    queries_f = queries.to(torch.float32)
    q_sketches = torch.matmul(queries_f, proj_dir_score.to(torch.float32))
    q_hashes = (q_sketches >= 0).to(torch.uint8)
    q_norms = torch.norm(queries_f, dim=-1)

    # Step 3: Project keys to sketch space
    keys_f = keys.to(torch.float32)
    keys_flat_f = keys_f.reshape(batch * kv_heads * num_blocks * block_size, head_size)
    k_sketches = torch.matmul(keys_flat_f, proj_dir_quant.to(torch.float32).t())
    k_hashes = (k_sketches >= 0).to(torch.uint8)
    k_hashes = k_hashes.reshape(batch, kv_heads, num_blocks, block_size, -1)
    k_norms = torch.norm(keys_f, dim=-1)

    # Step 4: Compute scores. Keep a CPU fallback for local smoke tests.
    if queries.is_cuda:
        scores = qjl_compute_scores_triton(
            q_hashes, k_hashes, q_norms, k_norms,
            block_table, seq_lens, scale, gqa_ratio
        )
    else:
        scores = qjl_compute_scores_python(
            q_hashes, k_hashes, q_norms, k_norms,
            block_table, seq_lens, scale, gqa_ratio
        )

    # Step 5: Softmax
    max_scores = scores.max(dim=-1, keepdim=True)[0]
    scores_exp = torch.exp(scores - max_scores)
    weights = scores_exp / scores_exp.sum(dim=-1, keepdim=True)

    # Step 6: Quantize values
    values_for_quant = values.reshape(batch, kv_heads, num_blocks * block_size, head_size)

    # Quantize
    value_pack, value_scale, value_min = quantize_values_2bit(
        values_for_quant, group_size=config.value_group_size
    )

    # Reshape quant outputs to match block structure
    value_pack = value_pack.reshape(batch, kv_heads, num_blocks, block_size, -1)
    value_scale = value_scale.reshape(batch, kv_heads, num_blocks, block_size, -1)
    value_min = value_min.reshape(batch, kv_heads, num_blocks, block_size, -1)

    # Step 7: Weighted sum. Keep a CPU fallback for local smoke tests.
    if queries.is_cuda:
        outputs = qjl_weighted_sum_triton(
            weights, value_pack, value_scale, value_min,
            block_table, seq_lens, gqa_ratio
        )
    else:
        outputs = qjl_weighted_sum_pytorch(
            weights, value_pack, value_scale, value_min,
            block_table, seq_lens, gqa_ratio
        )

    return outputs.to(queries.dtype)


# =============================================================================
# Packed Cache Format Adapter for vLLM Integration
# =============================================================================

def pack_qjl_cache(
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    outlier_indices: torch.Tensor,
    proj_dir_quant: torch.Tensor,
    value_group_size: int = 32,
    value_bits: int = 2,
    outlier_count: int = 8,
) -> dict[str, torch.Tensor]:
    """Pack key/value blocks into QJL cache format.

    Args:
        key_blocks: [num_blocks, kv_heads, block_size, head_size]
        value_blocks: [num_blocks, kv_heads, block_size, head_size]
        outlier_indices: [batch, kv_heads, outlier_count] per-head outlier indices
        proj_dir_quant: [head_size, sketch_dim]
        value_group_size: value quantization group size
        value_bits: value quantization bits
        outlier_count: number of outliers per head

    Returns:
        dict with packed cache components:
        - key_hash: [num_blocks, kv_heads, block_size, hash_bytes] packed sign bits
        - key_norm: [num_blocks, kv_heads, block_size]
        - outlier_idx: [num_blocks, kv_heads, outlier_count]
        - outlier_norm: [num_blocks, kv_heads, outlier_count]
        - value_pack, value_scale, value_min
    """
    num_blocks, kv_heads, block_size, head_size = key_blocks.shape
    sketch_dim = proj_dir_quant.shape[0]
    hash_bytes = (sketch_dim + 7) // 8

    # Project keys to sketch and sign-quantize
    key_flat = key_blocks.reshape(num_blocks * kv_heads * block_size, head_size)
    k_sketches = torch.matmul(key_flat.to(torch.float32), proj_dir_quant.to(torch.float32).t())
    k_hashes = (k_sketches >= 0).to(torch.uint8)
    k_hashes = k_hashes.reshape(num_blocks, kv_heads, block_size, sketch_dim)

    # Pack bits
    key_hash = qjl_pack_bits(k_hashes)

    # Compute norms
    key_norm = torch.norm(key_blocks.to(torch.float32), dim=-1)

    # Extract outlier values shared per KV head across blocks.
    outlier_idx = outlier_indices.squeeze(0)  # [kv_heads, outlier_count]
    outlier_index = outlier_idx.unsqueeze(0).unsqueeze(2).expand(
        num_blocks, -1, block_size, -1
    )
    outlier_values = torch.gather(key_blocks, dim=3, index=outlier_index)
    outlier_norm = outlier_values.abs().mean(dim=2)  # [num_blocks, kv_heads, outlier_count]

    # Quantize values
    value_for_quant = value_blocks.reshape(num_blocks * kv_heads * block_size, head_size)
    value_pack, value_scale, value_min = quantize_values_2bit(
        value_for_quant, group_size=value_group_size
    )

    # Reshape outputs
    value_pack = value_pack.reshape(num_blocks, kv_heads, block_size, -1)
    value_scale = value_scale.reshape(num_blocks, kv_heads, block_size, -1)
    value_min = value_min.reshape(num_blocks, kv_heads, block_size, -1)

    return {
        "key_hash": key_hash,
        "key_norm": key_norm,
        "outlier_idx": outlier_idx,
        "outlier_norm": outlier_norm,
        "value_pack": value_pack,
        "value_scale": value_scale,
        "value_min": value_min,
    }


def unpack_qjl_cache(
    packed: dict[str, torch.Tensor],
    sketch_dim: int,
    outlier_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack QJL cache to get hashes and norms.

    Returns:
        k_hashes: [num_blocks, kv_heads, block_size, sketch_dim]
        k_norms: [num_blocks, kv_heads, block_size]
    """
    key_hash = packed["key_hash"]
    num_blocks, kv_heads, block_size, hash_bytes = key_hash.shape

    # Unpack bytes to bits
    k_hashes = torch.zeros(
        num_blocks, kv_heads, block_size, sketch_dim,
        dtype=torch.uint8, device=key_hash.device
    )

    for byte_i in range(hash_bytes):
        for bit in range(8):
            dim_idx = byte_i * 8 + bit
            if dim_idx < sketch_dim:
                k_hashes[..., dim_idx] = (key_hash[..., byte_i] >> bit) & 1

    return k_hashes, packed["key_norm"]
