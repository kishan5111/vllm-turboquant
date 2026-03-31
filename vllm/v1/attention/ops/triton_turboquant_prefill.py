# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TurboQuant prefill fusion building blocks.

This module intentionally starts with a small, testable primitive:
dequantizing packed TurboQuant rows directly on device without materializing a
full [num_blocks, block_size, heads, head_dim] tensor.

The next step is to consume this primitive inside a fused prefill attention
kernel (QK + softmax + V) for long-context prefill.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _tq_unpack_rows_kernel(
    packed_ptr,      # [num_rows, comp_head] uint8
    out_lo_ptr,      # [num_rows, half] fp16/bf16
    out_hi_ptr,      # [num_rows, half] fp16/bf16
    stride_pr: tl.int64,
    stride_pc: tl.int64,
    stride_olr: tl.int64,
    stride_olc: tl.int64,
    stride_ohr: tl.int64,
    stride_ohc: tl.int64,
    num_rows,
    half: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row_blk = tl.program_id(0)
    offs_r = row_blk * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    offs_h = tl.arange(0, half)
    row_mask = offs_r < num_rows

    # Load packed nibbles: low half in low 4 bits, high half in high 4 bits.
    packed = tl.load(
        packed_ptr + offs_r[:, None] * stride_pr + offs_h[None, :] * stride_pc,
        mask=row_mask[:, None],
        other=0,
    ).to(tl.uint8)

    # Per-row fp16 scale is stored at [half] and [half+1] bytes in comp_head.
    b0 = tl.load(
        packed_ptr + offs_r * stride_pr + half * stride_pc,
        mask=row_mask,
        other=0,
    ).to(tl.uint16)
    b1 = tl.load(
        packed_ptr + offs_r * stride_pr + (half + 1) * stride_pc,
        mask=row_mask,
        other=0,
    ).to(tl.uint16)
    scale = (b0 | (b1 << 8)).to(tl.float16, bitcast=True)
    scale_mul = scale * 0.13333333333333333  # 2/15

    lo_q = (packed & 0x0F).to(tl.float16)
    hi_q = ((packed >> 4) & 0x0F).to(tl.float16)

    lo = lo_q * scale_mul[:, None] - scale[:, None]
    hi = hi_q * scale_mul[:, None] - scale[:, None]

    tl.store(
        out_lo_ptr + offs_r[:, None] * stride_olr + offs_h[None, :] * stride_olc,
        lo,
        mask=row_mask[:, None],
    )
    tl.store(
        out_hi_ptr + offs_r[:, None] * stride_ohr + offs_h[None, :] * stride_ohc,
        hi,
        mask=row_mask[:, None],
    )


def turboquant_unpack_rows(
    packed_rows: torch.Tensor,  # [num_rows, comp_head] uint8
    out_lo: torch.Tensor,       # [num_rows, half]
    out_hi: torch.Tensor,       # [num_rows, half]
) -> None:
    """Unpack/dequantize TurboQuant rows into low/high halves.

    This is a reusable prefill primitive for future fused attention kernels.
    """
    assert packed_rows.is_cuda and out_lo.is_cuda and out_hi.is_cuda
    assert packed_rows.dtype == torch.uint8
    assert packed_rows.dim() == 2 and out_lo.dim() == 2 and out_hi.dim() == 2
    num_rows, comp_head = packed_rows.shape
    half = out_lo.shape[1]
    assert out_hi.shape == out_lo.shape
    assert comp_head == half + 2

    if num_rows == 0:
        return

    BLOCK_ROWS = 8
    grid = (triton.cdiv(num_rows, BLOCK_ROWS),)
    _tq_unpack_rows_kernel[grid](
        packed_ptr=packed_rows,
        out_lo_ptr=out_lo,
        out_hi_ptr=out_hi,
        stride_pr=packed_rows.stride(0),
        stride_pc=packed_rows.stride(1),
        stride_olr=out_lo.stride(0),
        stride_olc=out_lo.stride(1),
        stride_ohr=out_hi.stride(0),
        stride_ohc=out_hi.stride(1),
        num_rows=num_rows,
        half=half,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _tq_prefill_fwd_kernel(
    Q,
    K,
    V,
    K_cache,  # [num_blocks, num_kv_heads, block_size, comp_head] uint8
    V_cache,  # [num_blocks, num_kv_heads, block_size, comp_head] uint8
    B_Loc,    # [batch, max_blocks]
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_bl,
    stride_k_cache_c,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_bl,
    stride_v_cache_c,
    num_queries_per_kv: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    SKIP_DECODE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    HALF: tl.constexpr = BLOCK_DMODEL // 2
    cur_kv_head = cur_head // num_queries_per_kv

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    if SKIP_DECODE and cur_batch_query_len == 1:
        return

    offs_bs_n = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, HALF)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Query split halves.
    off_q_lo = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_h[None, :] * stride_qd
    )
    off_q_hi = off_q_lo + HALF * stride_qd
    q_lo = tl.load(
        Q + off_q_lo,
        mask=offs_m[:, None] < cur_batch_query_len,
        other=0.0,
    )
    q_hi = tl.load(
        Q + off_q_hi,
        mask=offs_m[:, None] < cur_batch_query_len,
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_lo = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)

    # Context (cached) portion from packed TurboQuant K/V.
    for start_n in tl.range(0, cur_batch_ctx_len, BLOCK_SIZE):
        token_indices = start_n + offs_bs_n
        bn_logical_indices = token_indices // BLOCK_SIZE
        bn = tl.load(
            B_Loc + cur_batch * stride_b_loc_b + bn_logical_indices * stride_b_loc_s
        ).to(tl.int64)
        internal_offsets = token_indices % BLOCK_SIZE
        valid_ctx = (start_n + offs_bs_n) < cur_batch_ctx_len

        base = (
            bn[None, :] * stride_k_cache_bs
            + cur_kv_head * stride_k_cache_h
            + internal_offsets[None, :] * stride_k_cache_bl
        )
        k_packed = tl.load(
            K_cache + base + offs_h[:, None] * stride_k_cache_c,
            mask=valid_ctx[None, :],
            other=0,
        ).to(tl.uint8)
        kb0 = tl.load(
            K_cache + base[0, :] + HALF * stride_k_cache_c,
            mask=valid_ctx,
            other=0,
        ).to(tl.uint16)
        kb1 = tl.load(
            K_cache + base[0, :] + (HALF + 1) * stride_k_cache_c,
            mask=valid_ctx,
            other=0,
        ).to(tl.uint16)
        ks = (kb0 | (kb1 << 8)).to(tl.float16, bitcast=True)
        ks_scale = ks * 0.13333333333333333
        k_lo = (k_packed & 0x0F).to(tl.float16) * ks_scale[None, :] - ks[None, :]
        k_hi = ((k_packed >> 4) & 0x0F).to(tl.float16) * ks_scale[None, :] - ks[None, :]

        qk = sm_scale * (
            tl.dot(q_lo, k_lo) + tl.dot(q_hi, k_hi)
        )
        qk = tl.where(valid_ctx[None, :], qk, float("-inf"))

        if SLIDING_WINDOW > 0:
            qk = tl.where(
                (cur_batch_ctx_len + offs_m[:, None]) - (start_n + offs_bs_n[None, :])
                < SLIDING_WINDOW,
                qk,
                float("-inf"),
            )

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc_lo = acc_lo * alpha[:, None]
        acc_hi = acc_hi * alpha[:, None]

        base_v = (
            bn[None, :] * stride_v_cache_bs
            + cur_kv_head * stride_v_cache_h
            + internal_offsets[None, :] * stride_v_cache_bl
        )
        v_packed = tl.load(
            V_cache + base_v + offs_h[:, None] * stride_v_cache_c,
            mask=valid_ctx[None, :],
            other=0,
        ).to(tl.uint8)
        vb0 = tl.load(
            V_cache + base_v[0, :] + HALF * stride_v_cache_c,
            mask=valid_ctx,
            other=0,
        ).to(tl.uint16)
        vb1 = tl.load(
            V_cache + base_v[0, :] + (HALF + 1) * stride_v_cache_c,
            mask=valid_ctx,
            other=0,
        ).to(tl.uint16)
        vs = (vb0 | (vb1 << 8)).to(tl.float16, bitcast=True)
        vs_scale = vs * 0.13333333333333333
        v_lo = (
            (v_packed & 0x0F).to(tl.float16) * vs_scale[None, :] - vs[None, :]
        )
        v_hi = (
            ((v_packed >> 4) & 0x0F).to(tl.float16) * vs_scale[None, :] - vs[None, :]
        )
        p_v = p.to(v_lo.dtype)
        acc_lo = tl.dot(p_v, tl.trans(v_lo), acc=acc_lo)
        acc_hi = tl.dot(p_v, tl.trans(v_hi), acc=acc_hi)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Self (query chunk) portion from current K/V tensors (already rotated by caller).
    off_k_lo = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_h[:, None] * stride_kd
    off_k_hi = off_k_lo + HALF * stride_kd
    off_v_lo = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_h[None, :] * stride_vd
    off_v_hi = off_v_lo + HALF * stride_vd
    k_ptrs_lo = K + off_k_lo
    k_ptrs_hi = K + off_k_hi
    v_ptrs_lo = V + off_v_lo
    v_ptrs_hi = V + off_v_hi

    block_start_loc = BLOCK_M * start_m
    block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)
    for start_n in tl.range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        k_lo = tl.load(
            k_ptrs_lo + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=(start_n + offs_n[None, :]) < cur_batch_query_len,
            other=0.0,
        )
        k_hi = tl.load(
            k_ptrs_hi + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=(start_n + offs_n[None, :]) < cur_batch_query_len,
            other=0.0,
        )

        qk = sm_scale * (tl.dot(q_lo, k_lo) + tl.dot(q_hi, k_hi))
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        if SLIDING_WINDOW > 0:
            qk = tl.where(
                offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW,
                qk,
                float("-inf"),
            )

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc_lo = acc_lo * alpha[:, None]
        acc_hi = acc_hi * alpha[:, None]

        v_lo = tl.load(
            v_ptrs_lo + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < cur_batch_query_len,
            other=0.0,
        )
        v_hi = tl.load(
            v_ptrs_hi + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < cur_batch_query_len,
            other=0.0,
        )
        p_v = p.to(v_lo.dtype)
        acc_lo = tl.dot(p_v, v_lo, acc=acc_lo)
        acc_hi = tl.dot(p_v, v_hi, acc=acc_hi)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc_lo = acc_lo / (l_i[:, None] + 1e-10)
    acc_hi = acc_hi / (l_i[:, None] + 1e-10)

    off_o_lo = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_h[None, :] * stride_od
    )
    off_o_hi = off_o_lo + HALF * stride_od
    out_mask = offs_m[:, None] < cur_batch_query_len
    tl.store(Out + off_o_lo, acc_lo, mask=out_mask)
    tl.store(Out + off_o_hi, acc_hi, mask=out_mask)


def turboquant_context_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    b_loc: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    skip_decode: bool = False,
) -> None:
    """TurboQuant packed-cache prefill attention (Triton).

    Notes:
    - Expects K/V cache in TurboQuant packed AoS layout:
      [num_blocks, num_kv_heads, block_size, comp_head]
    - Expects q/k/v to already be in the same rotated basis.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and o.is_cuda
    assert k_cache.is_cuda and v_cache.is_cuda and b_loc.is_cuda
    assert k_cache.dtype == torch.uint8 and v_cache.dtype == torch.uint8
    assert q.shape[-1] % 2 == 0
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]

    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    batch = b_seq_len.shape[0]
    head = q.shape[1]
    num_queries_per_kv = q.shape[1] // k.shape[1]
    head_size = q.shape[-1]
    block_size = k_cache.shape[2]
    comp_head = k_cache.shape[3]
    assert comp_head == (head_size // 2 + 2)

    BLOCK_M = 128
    BLOCK_N = 64
    grid = (batch, head, triton.cdiv(max_input_len, BLOCK_M))
    _tq_prefill_fwd_kernel[grid](
        Q=q,
        K=k,
        V=v,
        K_cache=k_cache,
        V_cache=v_cache,
        B_Loc=b_loc.to(torch.int32),
        sm_scale=sm_scale,
        B_Start_Loc=b_start_loc,
        B_Seqlen=b_seq_len,
        Out=o,
        stride_b_loc_b=b_loc.stride(0),
        stride_b_loc_s=b_loc.stride(1),
        stride_qbs=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kbs=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        stride_vbs=v.stride(0),
        stride_vh=v.stride(1),
        stride_vd=v.stride(2),
        stride_obs=o.stride(0),
        stride_oh=o.stride(1),
        stride_od=o.stride(2),
        stride_k_cache_bs=k_cache.stride(0),
        stride_k_cache_h=k_cache.stride(1),
        stride_k_cache_bl=k_cache.stride(2),
        stride_k_cache_c=k_cache.stride(3),
        stride_v_cache_bs=v_cache.stride(0),
        stride_v_cache_h=v_cache.stride(1),
        stride_v_cache_bl=v_cache.stride(2),
        stride_v_cache_c=v_cache.stride(3),
        num_queries_per_kv=num_queries_per_kv,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=head_size,
        BLOCK_SIZE=block_size,
        BLOCK_N=BLOCK_N,
        SLIDING_WINDOW=sliding_window,
        SKIP_DECODE=skip_decode,
        num_warps=4,
        num_stages=1,
    )
