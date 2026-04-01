# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused TurboQuant decode attention using MSE+QJL quantization.

This module ports the 0xSero/turboquant fully-fused Triton kernels that
compute attention scores directly from packed TurboQuant-compressed data
without materializing full dequantized vectors.

Key advantage over PolarQuant (4-bit rotation):
  - PolarQuant: 6 arithmetic ops per nibble for dequantization
  - MSE+QJL: 1 multiply per dimension for QJL contribution
  - Score computed directly from packed data

Kernel 1: turboquant_mse_score
  Fuses MSE dequantization + centroid lookup + dot product

Kernel 2: turboquant_qjl_score
  Fuses QJL sign extraction + sketch query dot product

Kernel 3: turboquant_fused_decode_attention
  Full fusion: online softmax + value aggregation in single pass

Reference:
  0xSero/turboquant: https://github.com/0xSero/turboquant
  TurboQuant paper: Online Vector Quantization with Near-optimal Distortion Rate
"""

import math
import torch
from vllm.triton_utils import tl, triton


# ---------------------------------------------------------------------------
# Kernel 1: MSE score computation
# ---------------------------------------------------------------------------
#
# Given:
#   query_rot: (BH, D) — query @ Pi^T (pre-rotated)
#   mse_packed: (BH, N, packed_d) — bit-packed MSE indices
#   norms: (BH, N) — original vector norms
#   centroids: (n_clusters,) — Lloyd-Max codebook centroids
#
# Computes: scores[b,n] = sum_j q_rot[j] * centroid[idx[n,j]] * norms[n]
#   (avoids materializing full dequantized key vectors)

@triton.jit
def _turboquant_mse_score_kernel(
    Q_ptr,           # (BH, D) query vectors (pre-rotated: q @ Pi^T)
    MSE_ptr,         # (BH, N, packed_d) bit-packed MSE indices
    NORMS_ptr,       # (BH, N) original norms
    CENTROIDS_ptr,   # (n_clusters,) centroid values
    OUT_ptr,         # (BH, N) output scores
    # Strides
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_n_bh, stride_n_n,
    stride_o_bh, stride_o_n,
    # Dimensions
    BH: tl.constexpr,
    N,                # number of KV tokens
    D: tl.constexpr,
    PACKED_D: tl.constexpr,
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Load pre-rotated query: (D,)
    q = tl.load(Q_ptr + pid_bh * stride_q_bh + tl.arange(0, D) * stride_q_d).to(tl.float32)

    # Accumulate score for each token in the block
    scores = tl.zeros([BLOCK_N], dtype=tl.float32)

    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    # Process packed bytes — each byte contains VALS_PER_BYTE indices
    for byte_idx in range(PACKED_D):
        packed = tl.load(
            MSE_ptr + pid_bh * stride_m_bh + n_offs * stride_m_n + byte_idx * stride_m_d,
            mask=n_mask, other=0
        ).to(tl.int32)

        for sub in range(VALS_PER_BYTE):
            coord_idx = byte_idx * VALS_PER_BYTE + sub
            if coord_idx < D:
                idx = (packed >> (sub * BITS)) & BIT_MASK
                centroid_val = tl.load(CENTROIDS_ptr + idx)
                scores += q[coord_idx] * centroid_val

    # Multiply by norms
    norms = tl.load(
        NORMS_ptr + pid_bh * stride_n_bh + n_offs * stride_n_n,
        mask=n_mask, other=0.0
    ).to(tl.float32)
    scores = scores * norms

    tl.store(
        OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n,
        scores, mask=n_mask
    )


# ---------------------------------------------------------------------------
# Kernel 2: QJL score computation
# ---------------------------------------------------------------------------
#
# Given:
#   q_sketched: (BH, D) — precomputed q @ S^T
#   qjl_signs: (BH, N, D//8) — packed sign bits
#   residual_norms: (BH, N)
#   qjl_scale: sqrt(pi/2) / D
#
# Computes: scores[b,n] = qjl_scale * res_norms[n] * sum_j q_sketched[j] * sign[n,j]

@triton.jit
def _turboquant_qjl_score_kernel(
    Q_SKETCH_ptr,    # (BH, D) pre-sketched query
    SIGNS_ptr,       # (BH, N, packed_d_signs) packed sign bits
    RES_NORMS_ptr,   # (BH, N) residual norms
    OUT_ptr,         # (BH, N) output (added to existing)
    # Strides
    stride_qs_bh, stride_qs_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_rn_bh, stride_rn_n,
    stride_o_bh, stride_o_n,
    # Dims
    N,
    D: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,  # D // 8
    QJL_SCALE,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Accumulate dot product of q_sketched with sign vectors
    dot = tl.zeros([BLOCK_N], dtype=tl.float32)

    for byte_idx in range(PACKED_D_SIGNS):
        packed = tl.load(
            SIGNS_ptr + pid_bh * stride_s_bh + n_offs * stride_s_n + byte_idx * stride_s_d,
            mask=n_mask, other=0
        ).to(tl.int32)

        for bit in range(8):
            coord_idx = byte_idx * 8 + bit
            if coord_idx < D:
                sign_bit = (packed >> bit) & 1
                sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                q_val = tl.load(Q_SKETCH_ptr + pid_bh * stride_qs_bh + coord_idx * stride_qs_d).to(tl.float32)
                dot += q_val * sign_val

    # Scale by residual norms and QJL constant
    res_norms = tl.load(
        RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n,
        mask=n_mask, other=0.0
    ).to(tl.float32)
    qjl_scores = dot * res_norms * QJL_SCALE

    # Add to existing MSE scores
    existing = tl.load(
        OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n,
        mask=n_mask, other=0.0
    )
    tl.store(
        OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n,
        existing + qjl_scores, mask=n_mask
    )


# ---------------------------------------------------------------------------
# Kernel 3: Fully fused decode attention (online softmax over TQ keys + values)
# ---------------------------------------------------------------------------

@triton.jit
def _turboquant_fused_decode_kernel(
    # Query (pre-rotated for MSE, pre-sketched for QJL)
    Q_ROT_ptr,       # (BH, D) q @ Pi^T
    Q_SKETCH_ptr,    # (BH, D) q @ S^T
    # Quantized keys
    MSE_ptr,         # (BH, N, packed_d_mse) packed MSE indices
    SIGNS_ptr,       # (BH, N, packed_d_signs) packed QJL signs
    NORMS_ptr,       # (BH, N) key norms
    RES_NORMS_ptr,   # (BH, N) residual norms
    CENTROIDS_ptr,   # (n_clusters,) codebook
    # Values (group-quantized)
    V_DATA_ptr,      # (BH, N, D) uint8 quantized values
    V_SCALES_ptr,    # (BH, N, n_groups) value scales
    V_ZEROS_ptr,     # (BH, N, n_groups) value zeros
    # Recent tokens (float32, exact from ring buffer)
    RECENT_K_ptr,    # (BH, N_recent, D) float32 keys
    RECENT_V_ptr,    # (BH, N_recent, D) float32 values
    # Output
    OUT_ptr,         # (BH, D) output
    # Strides
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_n_bh, stride_n_n,
    stride_rn_bh, stride_rn_n,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_vs_bh, stride_vs_n, stride_vs_g,
    stride_vz_bh, stride_vz_n, stride_vz_g,
    stride_rk_bh, stride_rk_n, stride_rk_d,
    stride_rv_bh, stride_rv_n, stride_rv_d,
    stride_o_bh, stride_o_d,
    # Dims
    N,
    N_recent,
    D: tl.constexpr,
    PACKED_D_MSE: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    N_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    QJL_SCALE,
    SM_SCALE,  # 1/sqrt(d)
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    # Online softmax state
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    # Total tokens = history (N) + recent (N_recent)
    N_total = N + N_recent
    num_blocks_hist = tl.cdiv(N, BLOCK_N)
    num_blocks_recent = tl.cdiv(N_recent, BLOCK_N) if N_recent > 0 else 0
    total_blocks = num_blocks_hist + num_blocks_recent

    for block_idx in range(total_blocks):
        n_start = block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)

        # Determine if this is a history block or recent block
        is_history = block_idx < num_blocks_hist

        if is_history:
            # ---- History block: MSE+QJL dequantization ----
            n_mask = n_offs < N
            n_offs_hist = n_offs  # 0-based within history

            # MSE score
            mse_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
            for byte_idx in range(PACKED_D_MSE):
                packed = tl.load(
                    MSE_ptr + pid_bh * stride_m_bh + n_offs_hist * stride_m_n + byte_idx * stride_m_d,
                    mask=n_mask, other=0
                ).to(tl.int32)
                for sub in range(VALS_PER_BYTE):
                    coord_idx = byte_idx * VALS_PER_BYTE + sub
                    if coord_idx < D:
                        idx = (packed >> (sub * BITS)) & BIT_MASK
                        centroid_val = tl.load(CENTROIDS_ptr + idx)
                        q_val = tl.load(Q_ROT_ptr + pid_bh * stride_q_bh + coord_idx * stride_q_d).to(tl.float32)
                        mse_scores += q_val * centroid_val

            key_norms = tl.load(
                NORMS_ptr + pid_bh * stride_n_bh + n_offs_hist * stride_n_n,
                mask=n_mask, other=0.0
            ).to(tl.float32)
            mse_scores = mse_scores * key_norms

            # QJL score
            qjl_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
            for byte_idx in range(PACKED_D_SIGNS):
                packed = tl.load(
                    SIGNS_ptr + pid_bh * stride_s_bh + n_offs_hist * stride_s_n + byte_idx * stride_s_d,
                    mask=n_mask, other=0
                ).to(tl.int32)
                for bit in range(8):
                    coord_idx = byte_idx * 8 + bit
                    if coord_idx < D:
                        sign_bit = (packed >> bit) & 1
                        sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                        q_val = tl.load(Q_SKETCH_ptr + pid_bh * stride_q_bh + coord_idx * stride_q_d).to(tl.float32)
                        qjl_dot += q_val * sign_val

            res_norms = tl.load(
                RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs_hist * stride_rn_n,
                mask=n_mask, other=0.0
            ).to(tl.float32)
            qjl_scores = qjl_dot * res_norms * QJL_SCALE

            # Combined score
            scores = (mse_scores + qjl_scores) * SM_SCALE
            scores = tl.where(n_mask, scores, float("-inf"))

            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(scores, 0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)

            l_i = l_i * alpha + tl.sum(p, 0)
            acc = acc * alpha

            # Dequantize values for this block and accumulate
            d_offs = tl.arange(0, D)
            v_quant = tl.load(
                V_DATA_ptr + pid_bh * stride_v_bh
                + n_offs_hist[:, None] * stride_v_n + d_offs[None, :] * stride_v_d,
                mask=n_mask[:, None], other=0
            ).to(tl.float32)

            g_offs = d_offs // GROUP_SIZE
            v_scale = tl.load(
                V_SCALES_ptr + pid_bh * stride_vs_bh
                + n_offs_hist[:, None] * stride_vs_n + g_offs[None, :] * stride_vs_g,
                mask=n_mask[:, None], other=1.0
            ).to(tl.float32)
            v_zero = tl.load(
                V_ZEROS_ptr + pid_bh * stride_vz_bh
                + n_offs_hist[:, None] * stride_vz_n + g_offs[None, :] * stride_vz_g,
                mask=n_mask[:, None], other=0.0
            ).to(tl.float32)

            v_dequant = v_quant * v_scale + v_zero
            acc += tl.sum(p[:, None] * v_dequant, 0)

            m_i = m_new

        else:
            # ---- Recent block: float32 matmul (no dequantization) ----
            recent_block_idx = block_idx - num_blocks_hist
            n_start_recent = recent_block_idx * BLOCK_N
            n_offs_recent = n_start_recent + tl.arange(0, BLOCK_N)
            n_mask = n_offs_recent < N_recent

            # Float32 QK^T for recent tokens
            d_offs = tl.arange(0, D)
            q_vals = tl.load(
                Q_ROT_ptr + pid_bh * stride_q_bh + d_offs * stride_q_d,
            ).to(tl.float32)  # (D,)

            recent_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
            for d_idx in range(D):
                k_val = tl.load(
                    RECENT_K_ptr + pid_bh * stride_rk_bh
                    + n_offs_recent * stride_rk_n + d_idx * stride_rk_d,
                    mask=n_mask, other=0.0
                ).to(tl.float32)
                recent_scores += q_vals[d_idx] * k_val

            recent_scores = recent_scores * SM_SCALE

            # Online softmax update (continuing from history state)
            m_new = tl.maximum(m_i, tl.max(recent_scores, 0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(recent_scores - m_new)

            l_i = l_i * alpha + tl.sum(p, 0)
            acc = acc * alpha

            # Accumulate recent values
            v_vals = tl.load(
                RECENT_V_ptr + pid_bh * stride_rv_bh
                + n_offs_recent * stride_rv_n + d_offs[None, :] * stride_rv_d,
                mask=n_mask[:, None], other=0.0
            ).to(tl.float32)

            acc += tl.sum(p[:, None] * v_vals, 0)

            m_i = m_new

    # Final normalization
    acc = acc / l_i

    # Store output
    d_offs = tl.arange(0, D)
    tl.store(OUT_ptr + pid_bh * stride_o_bh + d_offs * stride_o_d, acc)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def _get_packing_params(bits: int):
    """Get packing parameters matching bit-packaging logic."""
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        return 4, 2  # 3-bit rounds up to 4-bit packing
    else:
        return 8, 1


def turboquant_mse_score(
    query_rot: torch.Tensor,     # (BH, D) — q @ Pi^T
    mse_packed: torch.Tensor,    # (BH, N, packed_d) uint8
    norms: torch.Tensor,         # (BH, N) float
    centroids: torch.Tensor,     # (n_clusters,) float32
    mse_bits: int,
) -> torch.Tensor:
    """
    Compute MSE attention scores using Triton kernel.

    Returns: (BH, N) attention logits (before scaling by 1/sqrt(d)).
    """
    if query_rot.dim() == 3:
        query_rot = query_rot.squeeze(1)

    BH, D = query_rot.shape
    N = mse_packed.shape[1]
    packed_d = mse_packed.shape[2]
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    out = torch.zeros(BH, N, device=query_rot.device, dtype=torch.float32)

    BLOCK_N = min(128, triton.next_power_of_2(N))
    grid = (BH, triton.cdiv(N, BLOCK_N))

    _turboquant_mse_score_kernel[grid](
        query_rot, mse_packed, norms, centroids, out,
        query_rot.stride(0), query_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        norms.stride(0), norms.stride(1),
        out.stride(0), out.stride(1),
        BH=BH, N=N, D=D, PACKED_D=packed_d,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        BLOCK_N=BLOCK_N,
    )
    return out


def turboquant_qjl_score(
    q_sketched: torch.Tensor,       # (BH, D) — q @ S^T
    qjl_signs: torch.Tensor,        # (BH, N, D//8) uint8 packed signs
    residual_norms: torch.Tensor,   # (BH, N)
    qjl_scale: float,               # sqrt(pi/2) / D
    out: torch.Tensor = None,       # (BH, N) — ADDED to if provided
) -> torch.Tensor:
    """
    Compute QJL attention score contribution.
    """
    if q_sketched.dim() == 3:
        q_sketched = q_sketched.squeeze(1)

    BH, D = q_sketched.shape
    N = qjl_signs.shape[1]
    packed_d_signs = qjl_signs.shape[2]

    if out is None:
        out = torch.zeros(BH, N, device=q_sketched.device, dtype=torch.float32)

    BLOCK_N = min(128, triton.next_power_of_2(N))
    grid = (BH, triton.cdiv(N, BLOCK_N))

    _turboquant_qjl_score_kernel[grid](
        q_sketched, qjl_signs, residual_norms, out,
        q_sketched.stride(0), q_sketched.stride(1),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        residual_norms.stride(0), residual_norms.stride(1),
        out.stride(0), out.stride(1),
        N=N, D=D, PACKED_D_SIGNS=packed_d_signs,
        QJL_SCALE=qjl_scale,
        BLOCK_N=BLOCK_N,
    )
    return out


def turboquant_fused_decode(
    query: torch.Tensor,               # (BH, 1, D) or (BH, D)
    quantized_key,                      # ProdQuantized namedtuple
    value_quantized,                    # ValueQuantized namedtuple
    Pi: torch.Tensor,                   # (D, D) rotation matrix
    S: torch.Tensor,                    # (D, D) QJL matrix
    centroids: torch.Tensor,           # (n_clusters,)
    mse_bits: int,
    qjl_scale: float,
    sm_scale: float,
    group_size: int = 32,
    gqa_ratio: int = 1,               # GQA ratio (H_q // H_kv)
    recent_k: torch.Tensor = None,     # (BH, N_recent, D) float32 keys from ring buffer
    recent_v: torch.Tensor = None,     # (BH, N_recent, D) float32 values from ring buffer
) -> torch.Tensor:
    """
    Fully fused decode attention: scores + softmax + value aggregation.
    Single pass over compressed KV (MSE+QJL) + recent (float32), flash-attention style.

    Supports two-segment attention:
    - History segment: packed MSE+QJL format (dequantized inside kernel)
    - Recent segment: float32 from ring buffer (direct matmul)

    For GQA (gqa_ratio > 1):
      - KV tensors are (H_kv, N, ...) but kernel expects (BH, N, ...)
      - We expand KV by repeating each KV head gqa_ratio times
      - This gives (BH=H_kv*gqa_ratio, N, ...) which is correct
      - Each Q-head variant within the same group attends to the same KV
        (correct for grouped-query attention)

    Returns: (BH, D) attention output.
    """
    if query.dim() == 3:
        query = query.squeeze(1)
    BH, D = query.shape

    # Precompute rotated and sketched queries (one-time per decode step)
    q_rot = torch.matmul(query.float(), Pi.T)
    q_sketch = torch.matmul(query.float(), S.T)

    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    # Handle batch dims
    if mse_packed.dim() > 3:
        BH_shape = mse_packed.shape[:2]
        BH_actual = BH_shape[0] * BH_shape[1]
        mse_packed = mse_packed.reshape(BH_actual, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH_actual, *qjl_signs.shape[2:])
        norms = norms.reshape(BH_actual, -1)
        res_norms = res_norms.reshape(BH_actual, -1)

    N = mse_packed.shape[1]
    packed_d_mse = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]

    # ----- GQA expansion -----
    # For GQA, each Q-head group shares the same KV head.
    # We expand (repeat) each KV head's data gqa_ratio times to match
    # the kernel's (BH=H_kv*gqa_ratio, N, ...) layout.
    if gqa_ratio > 1:
        H_kv = mse_packed.shape[0]  # e.g., 8 for GPT-OSS
        G = gqa_ratio               # e.g., 8 for GPT-OSS

        # Reshape and transpose query from (BH, D) to (H_kv, G, D)
        # so Q-heads that share the same KV head are contiguous in memory.
        # (BH, D) -> (H_kv, G, D) -> (BH, D) with reordered stride
        q_rot = q_rot.view(H_kv, G, D).transpose(0, 1).contiguous().view(BH, D)
        q_sketch = q_sketch.view(H_kv, G, D).transpose(0, 1).contiguous().view(BH, D)

        # Expand each KV head G times along the head dim (dim 0):
        # (H_kv, N, P) -> (H_kv*G, N, P)
        mse_packed = mse_packed.repeat_interleave(gqa_ratio, dim=0).contiguous()
        qjl_signs = qjl_signs.repeat_interleave(gqa_ratio, dim=0).contiguous()
        norms = norms.repeat_interleave(gqa_ratio, dim=0).contiguous()
        res_norms = res_norms.repeat_interleave(gqa_ratio, dim=0).contiguous()

    v_data = value_quantized.data
    v_scales = value_quantized.scales
    v_zeros = value_quantized.zeros

    # Unpack bit-packed values if needed
    v_bits = value_quantized.bits if len(value_quantized) > 3 else 2
    if v_bits == 2 and v_data.shape[-1] != D:
        from .kv_cache import unpack_values
        v_data = unpack_values(value_quantized)
    elif v_bits == 4 and v_data.shape[-1] != D:
        from .kv_cache import unpack_values
        v_data = unpack_values(value_quantized)

    if v_data.dim() > 3:
        v_data = v_data.reshape(BH, N, -1)
        v_scales = v_scales.reshape(BH, N, -1)
        v_zeros = v_zeros.reshape(BH, N, -1)

    # Expand value tensors for GQA
    if gqa_ratio > 1:
        v_data = v_data.repeat_interleave(gqa_ratio, dim=0).contiguous()
        v_scales = v_scales.repeat_interleave(gqa_ratio, dim=0).contiguous()
        v_zeros = v_zeros.repeat_interleave(gqa_ratio, dim=0).contiguous()

    # ----- Handle recent tokens (float32 from ring buffer) -----
    N_recent = 0
    if recent_k is not None and recent_v is not None:
        recent_k = recent_k.contiguous()
        recent_v = recent_v.contiguous()
        N_recent = recent_k.shape[1]
        if gqa_ratio > 1:
            # Expand recent KV for GQA: (H_kv, N_recent, D) -> (BH=H_kv*G, N_recent, D)
            recent_k = recent_k.repeat_interleave(gqa_ratio, dim=0).contiguous()
            recent_v = recent_v.repeat_interleave(gqa_ratio, dim=0).contiguous()

    N_GROUPS = D // group_size
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    out = torch.zeros(BH, D, device=query.device, dtype=torch.float32)

    BLOCK_N = min(64, triton.next_power_of_2(max(N + N_recent, 1)))
    grid = (BH,)

    # Prepare recent strides (even when None to keep kernel call valid)
    stride_rk = (recent_k.stride(0), recent_k.stride(1), recent_k.stride(2)) if N_recent > 0 else (0, 0, 0)
    stride_rv = (recent_v.stride(0), recent_v.stride(1), recent_v.stride(2)) if N_recent > 0 else (0, 0, 0)

    _turboquant_fused_decode_kernel[grid](
        q_rot, q_sketch,
        mse_packed, qjl_signs, norms, res_norms, centroids,
        v_data, v_scales, v_zeros,
        recent_k if N_recent > 0 else torch.zeros(1, 1, D, device=query.device, dtype=torch.float32),
        recent_v if N_recent > 0 else torch.zeros(1, 1, D, device=query.device, dtype=torch.float32),
        out,
        # Q strides
        q_rot.stride(0), q_rot.stride(1),
        # MSE strides
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        # Signs strides
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        # Norms strides
        norms.stride(0), norms.stride(1),
        # Res norms strides
        res_norms.stride(0), res_norms.stride(1),
        # Value strides
        v_data.stride(0), v_data.stride(1), v_data.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        # Recent strides
        stride_rk[0], stride_rk[1], stride_rk[2],
        stride_rv[0], stride_rv[1], stride_rv[2],
        # Out strides
        out.stride(0), out.stride(1),
        # Dims
        N=N, N_recent=N_recent, D=D, PACKED_D_MSE=packed_d_mse, PACKED_D_SIGNS=packed_d_signs,
        N_GROUPS=N_GROUPS, GROUP_SIZE=group_size,
        # Quant params
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        QJL_SCALE=qjl_scale,
        SM_SCALE=sm_scale,
        BLOCK_N=BLOCK_N,
    )
    return out