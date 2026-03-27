# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused TurboQuant paged decode attention — GQA-fused split-KV + AoS cache layout.

Key optimisations:

  AoS cache layout
    Within each (block, kv_head, position) token slot:
      [HALF nibble bytes | 2 scale bytes]  stride = COMP_HEAD = HALF+2 = 66
    Non-power-of-2 stride forces Triton to generate row-by-row 64-byte loads
    which are pipeline-friendly (each row fits in registers without spilling).
    A power-of-2 stride (HALF=64) would cause Triton to generate a single
    contiguous 1024-byte load; with num_stages=4 this fills registers and
    causes register spilling → 57% slowdown.

  GQA head fusion  (primary throughput win)
    Grid = (num_seqs, num_kv_heads, KV_SPLITS).  Each program processes
    ALL GQA_RATIO Q heads that share the same KV head.  K and V are
    loaded ONCE per block and reused across GQA_RATIO dot products.
    For GQA_RATIO=4 this gives 4× less K/V HBM traffic vs the previous
    per-Q-head kernel.

    Row extraction from 2D register tensors uses:
      row_g = tl.sum(tensor * (arange == g).to(float32)[:, None], axis=0)
    and row updates use tl.where masking — both avoid constexpr tensor
    indexing which Triton does not support.

  Tensor-core QK + V accumulation  (for GQA_RATIO that is a multiple of 16)
    When GQA_RATIO >= 16, uses tl.dot for:
      scores [GQA_RATIO, BLOCK_SIZE] = Q_lo [GQA_RATIO, HALF] @ K_lo.T [HALF, BLOCK_SIZE]
      delta  [GQA_RATIO, HALF]       = exp_s [GQA_RATIO, BLOCK_SIZE] @ V_lo [BLOCK_SIZE, HALF]
    This hits H100 tensor cores (16×16×64 tiles) for up to 20× compute throughput.

  KV split (FlashDecoding-style)
    Grid = (num_seqs, num_kv_heads, KV_SPLITS).  Merge kernel combines
    partial softmax outputs using log-sum-exp.

  Software pipelining (num_stages=4)
    Triton overlaps next-block HBM loads with current-block ALU.

Math:
  K stored in rotated space: K_stored = quantise(R @ K)
  Caller pre-rotates Q once: q_rot = Q @ R
  Score_i = q_rot · k_dq_i    (k_dq_i = dequantised K_stored[i])
  V accumulated in rotated space; caller applies R^T once to output.
"""

import math

import torch
from vllm.triton_utils import tl, triton

# H100 SM count used for adaptive kv_splits heuristic.
# Override via TURBOQUANT_SM_COUNT env var if running on a different GPU.
import os as _os
_SM_COUNT: int = int(_os.environ.get("TURBOQUANT_SM_COUNT", "132"))


# ---------------------------------------------------------------------------
# GQA-fused split kernel — one program per (seq, kv_head, kv_split)
# ---------------------------------------------------------------------------

@triton.jit
def _tq_split_kv_kernel(
    q_ptr,          # [num_seqs, num_q_heads, HEAD_SIZE]  bf16
    k_cache_ptr,    # [num_blocks, num_kv_heads, block_size, COMP_HEAD]  uint8  AoS layout
    v_cache_ptr,    #   same shape
    rot_ptr,        # [HEAD_SIZE, HEAD_SIZE] bf16/fp16 or unused when Q_IS_ROTATED=True
    bt_ptr,         # [num_seqs, MAX_BLOCKS]  int32
    lens_ptr,       # [num_seqs]  int32
    # outputs:
    #   WRITE_DIRECT=False → float32 partial results for merge kernel
    #   WRITE_DIRECT=True  → bf16 normalised output (written directly, no merge)
    out_acc_ptr,    # [num_seqs, num_q_heads, KV_SPLITS, HEAD_SIZE]  (partial, float32)
    out_m_ptr,      # [num_seqs, num_q_heads, KV_SPLITS]              (partial m)
    out_l_ptr,      # [num_seqs, num_q_heads, KV_SPLITS]              (partial l)
    out_direct_ptr, # [num_seqs, num_q_heads, HEAD_SIZE]              (direct bf16 out)
    # strides
    stride_q_s: tl.int64,
    stride_q_h: tl.int64,
    stride_r0: tl.int64,
    stride_r1: tl.int64,
    stride_c_b: tl.int64,   # block stride
    stride_c_h: tl.int64,   # kv-head stride
    stride_c_p: tl.int64,   # position stride = COMP_HEAD = HALF+2 = 66
    stride_bt:  tl.int64,
    stride_oa_s: tl.int64,
    stride_oa_h: tl.int64,
    stride_oa_k: tl.int64,
    stride_ml_s: tl.int64,
    stride_ml_h: tl.int64,
    # direct-output strides (only used when WRITE_DIRECT=True)
    stride_od_s: tl.int64,
    stride_od_h: tl.int64,
    attn_scale,
    HEAD_SIZE:        tl.constexpr,
    HALF:             tl.constexpr,
    COMP_HEAD:        tl.constexpr,
    BLOCK_SIZE:       tl.constexpr,
    MAX_BLOCKS:       tl.constexpr,
    BLOCKS_PER_SPLIT:  tl.constexpr,
    BPS_POW2:          tl.constexpr,  # next power-of-2 >= BLOCKS_PER_SPLIT (for tl.arange)
    GQA_RATIO:         tl.constexpr,
    KV_SPLITS:         tl.constexpr,
    WRITE_DIRECT:      tl.constexpr,  # True → normalise + write output, skip merge
    USE_BT_PREFETCH:   tl.constexpr,  # preload all bt entries to break load-dep chain
    Q_IS_ROTATED:      tl.constexpr,
):
    seq     = tl.program_id(0)
    kv_head = tl.program_id(1)   # iterates over KV heads
    split   = tl.program_id(2)

    ctx_len = tl.load(lens_ptr + seq)

    half_offs = tl.arange(0, HALF)
    pos_offs  = tl.arange(0, BLOCK_SIZE)
    gqa_offs  = tl.arange(0, GQA_RATIO)   # [GQA_RATIO]

    # Load all GQA_RATIO Q heads for this KV head: [GQA_RATIO, HALF] each.
    # These are loaded ONCE and reused across all block iterations.
    q_group_base = seq * stride_q_s + kv_head * GQA_RATIO * stride_q_h
    q_lo_in = tl.load(
        q_ptr + q_group_base
        + gqa_offs[:, None] * stride_q_h
        + half_offs[None, :],
    )
    q_hi_in = tl.load(
        q_ptr + q_group_base
        + gqa_offs[:, None] * stride_q_h
        + HALF + half_offs[None, :],
    )

    if Q_IS_ROTATED:
        q_lo_grp = q_lo_in.to(tl.float32)
        q_hi_grp = q_hi_in.to(tl.float32)
    else:
        r00 = tl.load(
            rot_ptr
            + half_offs[:, None] * stride_r0
            + half_offs[None, :] * stride_r1,
        ).to(q_lo_in.dtype)
        r01 = tl.load(
            rot_ptr
            + half_offs[:, None] * stride_r0
            + (HALF + half_offs)[None, :] * stride_r1,
        ).to(q_lo_in.dtype)
        r10 = tl.load(
            rot_ptr
            + (HALF + half_offs)[:, None] * stride_r0
            + half_offs[None, :] * stride_r1,
        ).to(q_lo_in.dtype)
        r11 = tl.load(
            rot_ptr
            + (HALF + half_offs)[:, None] * stride_r0
            + (HALF + half_offs)[None, :] * stride_r1,
        ).to(q_lo_in.dtype)

        q_lo_grp = (
            tl.dot(q_lo_in, r00, input_precision="ieee")
            + tl.dot(q_hi_in, r10, input_precision="ieee")
        ).to(tl.float32)
        q_hi_grp = (
            tl.dot(q_lo_in, r01, input_precision="ieee")
            + tl.dot(q_hi_in, r11, input_precision="ieee")
        ).to(tl.float32)

    # Per-Q-head online softmax state. Keep the full GQA group live as 2D / 1D
    # tensors so GPT-OSS's GQA_RATIO=8 path can update every shared-Q head in
    # one matrix-style step instead of an 8x scalar loop.
    m_grp    = tl.full([GQA_RATIO], -1e38, dtype=tl.float32)
    l_grp    = tl.zeros([GQA_RATIO], dtype=tl.float32)
    acc_lo_g = tl.zeros([GQA_RATIO, HALF], dtype=tl.float32)
    acc_hi_g = tl.zeros([GQA_RATIO, HALF], dtype=tl.float32)

    split_blk_start = split * BLOCKS_PER_SPLIT

    # -----------------------------------------------------------------
    # Pre-load all block IDs for this split into registers in one batch.
    # This breaks the two-level dependent-load chain:
    #   bt[seq, blk] → bid → cache[bid, kv_head, ...]
    # that would otherwise prevent num_stages=4 from pipelining K/V loads
    # across iterations (the next K/V address can't be computed until the
    # bt load of the current iteration completes).  With bids in registers,
    # the K/V address for iteration i+4 is immediately computable, so
    # Triton's software pipeline can issue those loads 4 iterations early.
    # -----------------------------------------------------------------
    if USE_BT_PREFETCH:
        bt_base    = bt_ptr + seq * stride_bt + split_blk_start
        bps_arange = tl.arange(0, BPS_POW2)
        blk_in_bt  = split_blk_start + bps_arange
        bids_reg   = tl.load(
            bt_base + bps_arange,
            mask=(bps_arange < BLOCKS_PER_SPLIT) & (blk_in_bt < MAX_BLOCKS),
            other=0,
        ).to(tl.int64)

    for blk_in_split in range(BLOCKS_PER_SPLIT):
        actual_blk = split_blk_start + blk_in_split
        blk_start  = actual_blk * BLOCK_SIZE
        abs_pos    = blk_start + pos_offs        # [BLOCK_SIZE]
        valid      = (abs_pos < ctx_len) & (actual_blk < MAX_BLOCKS)

        if USE_BT_PREFETCH:
            bid = tl.sum(bids_reg * (bps_arange == blk_in_split).to(tl.int64))
        else:
            bid = tl.load(
                bt_ptr + seq * stride_bt + actual_blk,
                mask=(blk_start < ctx_len) & (actual_blk < MAX_BLOCKS),
                other=0,
            ).to(tl.int64)

        # AoS layout: each position token slot is [HALF nibbles | 2 scale bytes].
        # Position stride = COMP_HEAD = HALF+2 = 66 (non-power-of-2).
        base = bid * stride_c_b + kv_head * stride_c_h

        # ---- Load K once — AoS nibbles ----
        k_packed = tl.load(
            k_cache_ptr + base
            + pos_offs[:, None] * stride_c_p    # AoS: stride = COMP_HEAD = 66
            + half_offs[None, :],
            mask=valid[:, None],
            other=0,
        ).to(tl.uint8)                              # [BLOCK_SIZE, HALF]

        # AoS scales: at offset HALF within each token slot.
        k_b0 = tl.load(k_cache_ptr + base + pos_offs * stride_c_p + HALF,
                        mask=valid, other=0).to(tl.uint16)
        k_b1 = tl.load(k_cache_ptr + base + pos_offs * stride_c_p + HALF + 1,
                        mask=valid, other=0).to(tl.uint16)
        ks = (k_b0 | (k_b1 << 8)).to(tl.float16, bitcast=True)
        ks_scale = ks * 0.13333333333333333  # 2.0/15.0 as float literal; Triton auto-casts to fp16

        # FP16 intermediates: halve register usage vs float32 ([16,64] fp16 = 2KB
        # vs 4KB float32).  With num_stages pipelining, fewer in-flight tiles
        # fit in the warp register file without spilling.
        k_lo = (k_packed & 0x0F).to(tl.float16) * ks_scale[:, None] - ks[:, None]  # [BS, HALF] fp16
        k_hi = ((k_packed >> 4) & 0x0F).to(tl.float16) * ks_scale[:, None] - ks[:, None]

        # ---- Load V once — AoS nibbles ----
        v_packed = tl.load(
            v_cache_ptr + base
            + pos_offs[:, None] * stride_c_p    # AoS: stride = COMP_HEAD = 66
            + half_offs[None, :],
            mask=valid[:, None],
            other=0,
        ).to(tl.uint8)                              # [BLOCK_SIZE, HALF]

        v_b0 = tl.load(v_cache_ptr + base + pos_offs * stride_c_p + HALF,
                        mask=valid, other=0).to(tl.uint16)
        v_b1 = tl.load(v_cache_ptr + base + pos_offs * stride_c_p + HALF + 1,
                        mask=valid, other=0).to(tl.uint16)
        vs = (v_b0 | (v_b1 << 8)).to(tl.float16, bitcast=True)
        vs_scale = vs * 0.13333333333333333

        v_lo = (v_packed & 0x0F).to(tl.float16) * vs_scale[:, None] - vs[:, None]  # [BS, HALF] fp16
        v_hi = ((v_packed >> 4) & 0x0F).to(tl.float16) * vs_scale[:, None] - vs[:, None]

        # ---- Group softmax + accumulation ----
        # Process the entire GQA group in one matrix-style update. This avoids
        # the previous compile-time-unrolled per-head masked loop, which is
        # especially costly for GPT-OSS where GQA_RATIO=8.
        scores = (
            tl.dot(q_lo_grp.to(k_lo.dtype), tl.trans(k_lo), input_precision="ieee")
            + tl.dot(q_hi_grp.to(k_hi.dtype), tl.trans(k_hi), input_precision="ieee")
        ).to(tl.float32) * attn_scale                            # [GQA_RATIO, BLOCK_SIZE]
        scores = tl.where(valid[None, :], scores, -1e38)

        block_max = tl.max(scores, axis=1)                       # [GQA_RATIO]
        m_new     = tl.maximum(m_grp, block_max)
        alpha     = tl.exp(m_grp - m_new)                        # [GQA_RATIO]
        exp_sc    = tl.exp(scores - m_new[:, None])              # [GQA_RATIO, BLOCK_SIZE]
        exp_sc    = tl.where(valid[None, :], exp_sc, 0.0)

        delta_lo = tl.dot(exp_sc.to(v_lo.dtype), v_lo, input_precision="ieee").to(
            tl.float32
        )                                                        # [GQA_RATIO, HALF]
        delta_hi = tl.dot(exp_sc.to(v_hi.dtype), v_hi, input_precision="ieee").to(
            tl.float32
        )                                                        # [GQA_RATIO, HALF]

        acc_lo_g = acc_lo_g * alpha[:, None] + delta_lo
        acc_hi_g = acc_hi_g * alpha[:, None] + delta_hi
        l_grp    = l_grp * alpha + tl.sum(exp_sc, axis=1)
        m_grp    = m_new

    # Write results for all GQA Q heads.
    for g in tl.static_range(GQA_RATIO):
        q_head = kv_head * GQA_RATIO + g
        sel_g  = (gqa_offs == g).to(tl.float32)

        acc_lo_row = tl.sum(acc_lo_g * sel_g[:, None], axis=0)  # [HALF]
        acc_hi_row = tl.sum(acc_hi_g * sel_g[:, None], axis=0)  # [HALF]
        m_row      = tl.sum(m_grp * sel_g)                       # scalar
        l_row      = tl.sum(l_grp * sel_g)                       # scalar

        if WRITE_DIRECT:
            # kv_splits == 1: normalise here, write bf16 output directly.
            l_safe     = tl.maximum(l_row, 1e-8)
            acc_lo_row = acc_lo_row / l_safe
            acc_hi_row = acc_hi_row / l_safe
            od_base = seq * stride_od_s + q_head * stride_od_h
            tl.store(out_direct_ptr + od_base + half_offs,
                     acc_lo_row.to(tl.bfloat16))
            tl.store(out_direct_ptr + od_base + HALF + half_offs,
                     acc_hi_row.to(tl.bfloat16))
        else:
            oa_base = seq * stride_oa_s + q_head * stride_oa_h + split * stride_oa_k
            tl.store(out_acc_ptr + oa_base + half_offs,        acc_lo_row)
            tl.store(out_acc_ptr + oa_base + HALF + half_offs, acc_hi_row)

            ml_idx = seq * stride_ml_s + q_head * stride_ml_h + split
            tl.store(out_m_ptr + ml_idx, m_row)
            tl.store(out_l_ptr + ml_idx, l_row)


# ---------------------------------------------------------------------------
# Merge kernel — one program per (seq, q_head)
# ---------------------------------------------------------------------------

@triton.jit
def _tq_merge_splits_kernel(
    out_acc_ptr,    # [num_seqs, num_q_heads, KV_SPLITS, HEAD_SIZE]  float32
    out_m_ptr,      # [num_seqs, num_q_heads, KV_SPLITS]              float32
    out_l_ptr,      # [num_seqs, num_q_heads, KV_SPLITS]              float32
    out_rot_ptr,    # [num_seqs, num_q_heads, HEAD_SIZE]               bf16 (write)
    stride_oa_s: tl.int64,
    stride_oa_h: tl.int64,
    stride_oa_k: tl.int64,
    stride_ml_s: tl.int64,
    stride_ml_h: tl.int64,
    stride_q_s:  tl.int64,
    stride_q_h:  tl.int64,
    HEAD_SIZE:  tl.constexpr,
    HALF:       tl.constexpr,
    KV_SPLITS:  tl.constexpr,
):
    seq  = tl.program_id(0)
    head = tl.program_id(1)

    half_offs  = tl.arange(0, HALF)
    split_offs = tl.arange(0, KV_SPLITS)

    ml_base = seq * stride_ml_s + head * stride_ml_h
    m_all   = tl.load(out_m_ptr + ml_base + split_offs)
    l_all   = tl.load(out_l_ptr + ml_base + split_offs)

    m_global = tl.max(m_all, axis=0)
    scales   = tl.exp(m_all - m_global)
    l_global = tl.sum(l_all * scales)

    oa_base    = seq * stride_oa_s + head * stride_oa_h
    acc_lo_all = tl.load(
        out_acc_ptr + oa_base
        + split_offs[:, None] * stride_oa_k
        + half_offs[None, :],
    )
    acc_hi_all = tl.load(
        out_acc_ptr + oa_base
        + split_offs[:, None] * stride_oa_k
        + HALF + half_offs[None, :],
    )

    acc_lo_out = tl.sum(scales[:, None] * acc_lo_all, axis=0)
    acc_hi_out = tl.sum(scales[:, None] * acc_hi_all, axis=0)

    l_safe     = tl.maximum(l_global, 1e-8)
    acc_lo_out = acc_lo_out / l_safe
    acc_hi_out = acc_hi_out / l_safe

    ob = seq * stride_q_s + head * stride_q_h
    tl.store(out_rot_ptr + ob + half_offs,        acc_lo_out.to(tl.bfloat16))
    tl.store(out_rot_ptr + ob + HALF + half_offs, acc_hi_out.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def _adaptive_kv_splits(num_seqs: int, num_kv_heads: int,
                        max_blocks: int = 0) -> int:
    """Choose kv_splits to maximise SM utilisation for the given batch/context.

    Strategy (tuned on H100 — see benchmark sweep in progress.md):
      • Primary: fill the GPU — target ≥ 16 × SM_COUNT total Triton programs.
        At 132 SMs this is 2112 programs, giving ~16 waves of 132 programs each.
      • Cap at 32.  Beyond 32 the merge-kernel cost grows faster than the
        decode-kernel gains for long-context (ctx≥32K) workloads.
      • Always round DOWN to the nearest power-of-two so the constexpr
        specialisation is reused across identical calls.
      • kv_splits=1 → WRITE_DIRECT fast path (no merge kernel at all).
    """
    progs  = num_seqs * num_kv_heads

    target = _SM_COUNT * 32          # ~4224 programs for H100
    needed = max(1, (target + progs - 1) // progs)
    # GPT-OSS/H100 sweeps favor fewer splits at ~8k (512 blocks) but more
    # splits again once per-sequence context reaches ~16k (1024 blocks).
    if max_blocks >= 1024:
        hard_cap = 32
    else:
        hard_cap = 16
    capped = min(needed, hard_cap)

    p2 = 1
    while p2 * 2 <= capped:
        p2 *= 2
    return p2


def turboquant_fused_paged_decode(
    query:       torch.Tensor,        # [num_seqs, num_q_heads, head_size]  BF16/FP16
    key_cache:   torch.Tensor,        # [num_blocks, num_kv_heads, block_size, comp_head]  uint8
    value_cache: torch.Tensor,
    block_table: torch.Tensor,        # [num_seqs, max_blocks_per_seq]  int32
    seq_lens:    torch.Tensor,        # [num_seqs]  int32
    rotation:    torch.Tensor,        # [head_size, head_size]  BF16/FP16
    scale:       float,
    kv_splits:   int | None = None,   # None → auto-tune per batch/GPU
    skip_output_inverse_rotation: bool = False,
    rotate_inside_decode: bool = False,
    split_num_warps: int | None = None,
    split_num_stages: int | None = None,
    merge_num_warps: int | None = None,
) -> torch.Tensor:                    # [num_seqs, num_q_heads, head_size]  BF16/FP16
    """Fused TurboQuant paged decode — GQA-fused split-KV + AoS cache layout.

    AoS layout
    ----------
    Each (block, kv_head, position) token slot stores [HALF nibbles | 2 scale bytes]
    contiguously.  Position stride = COMP_HEAD = HALF+2 = 66.
    The non-power-of-2 stride forces Triton to generate row-by-row 64-byte loads
    which are pipeline-friendly (each row fits in registers without spilling).

    Adaptive kv_splits
    ------------------
    When kv_splits=None (default) the split count is chosen to keep total
    Triton programs ≥ 8 × SM_COUNT, ranging from 1 (large-batch short-ctx)
    to 16 (single-sequence long-ctx).  kv_splits=1 skips the merge kernel
    entirely — the split kernel normalises and writes output directly.
    """
    num_seqs,    num_q_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    block_size   = key_cache.shape[2]
    comp_head    = key_cache.shape[3]
    max_blocks   = block_table.shape[1]
    half         = head_size // 2
    assert num_q_heads % num_kv_heads == 0
    gqa_ratio    = num_q_heads // num_kv_heads

    if kv_splits is None:
        kv_splits = _adaptive_kv_splits(num_seqs, num_kv_heads, max_blocks)

    blocks_per_split = triton.cdiv(max_blocks, kv_splits)
    write_direct     = (kv_splits == 1)

    # Next power-of-2 >= blocks_per_split (required by tl.arange in bt-prefetch).
    # USE_BT_PREFETCH is only beneficial when BPS_POW2 ≤ 64: the mask-sum
    # extraction inside the loop costs O(BPS × BPS_POW2) ops, which is
    # negligible for small splits but adds more overhead than it saves
    # when BPS > ~64 (large contexts / few splits).
    bps_pow2 = 1
    while bps_pow2 < blocks_per_split:
        bps_pow2 *= 2
    use_bt_prefetch = (bps_pow2 <= 64)

    dtype = query.dtype
    rotation = rotation.to(dtype).contiguous()
    q_input = query.contiguous()
    q_is_rotated = not rotate_inside_decode
    if q_is_rotated:
        # Use the legacy eager path when explicitly requested by tests or
        # benchmarks. The backend uses rotate_inside_decode=True.
        q_input = (query.reshape(-1, head_size) @ rotation).reshape(
            num_seqs, num_q_heads, head_size
        ).contiguous()

    # Allocate partial buffers (used when kv_splits > 1).
    # When write_direct=True these are zero-size placeholders — the kernel
    # writes to out_direct instead and the merge step is skipped.
    if write_direct:
        dummy = torch.empty(0, dtype=torch.float32, device=query.device)
        partial_acc = dummy
        partial_m   = dummy
        partial_l   = dummy
        # Strides for unused buffers — pass 0 so the compiler can fold them.
        s_oa_s, s_oa_h, s_oa_k = 0, 0, 0
        s_ml_s, s_ml_h          = 0, 0
    else:
        partial_acc = torch.empty(
            (num_seqs, num_q_heads, kv_splits, head_size),
            dtype=torch.float32, device=query.device,
        )
        partial_m = torch.full(
            (num_seqs, num_q_heads, kv_splits),
            -1e38, dtype=torch.float32, device=query.device,
        )
        partial_l = torch.zeros(
            (num_seqs, num_q_heads, kv_splits),
            dtype=torch.float32, device=query.device,
        )
        s_oa_s, s_oa_h, s_oa_k = (
            partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2))
        s_ml_s, s_ml_h = partial_m.stride(0), partial_m.stride(1)

    # Direct output buffer — always bf16 (kernel always writes tl.bfloat16).
    # Used for the kv_splits=1 fast path; also passed as a valid pointer arg
    # to the kernel even when write_direct=False (the kernel ignores it).
    out_direct = torch.empty(
        (num_seqs, num_q_heads, head_size),
        dtype=torch.bfloat16, device=query.device,
    )

    # H100/GPT-OSS tuning: 4 split warps consistently wins in the 8k/16k
    # decode region we care about, even when blocks_per_split reaches 64.
    num_warps = (
        split_num_warps
        if split_num_warps is not None
        else 4
    )
    # Tune staging by per-program KV chunk size. Smaller chunks benefit from
    # deeper pipelining; longer chunks need lower register pressure.
    num_stages = (
        split_num_stages
        if split_num_stages is not None
        else (
            1 if blocks_per_split >= 128
            else 2 if blocks_per_split >= 64
            else 3
        )
    )

    _tq_split_kv_kernel[(num_seqs, num_kv_heads, kv_splits)](
        q_input, key_cache, value_cache, rotation, block_table, seq_lens,
        partial_acc, partial_m, partial_l, out_direct,
        q_input.stride(0), q_input.stride(1),
        rotation.stride(0), rotation.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),  # block, head, position
        block_table.stride(0),
        s_oa_s, s_oa_h, s_oa_k,
        s_ml_s, s_ml_h,
        out_direct.stride(0), out_direct.stride(1),
        scale,
        HEAD_SIZE=head_size,
        HALF=half,
        COMP_HEAD=comp_head,
        BLOCK_SIZE=block_size,
        MAX_BLOCKS=max_blocks,
        BLOCKS_PER_SPLIT=blocks_per_split,
        BPS_POW2=bps_pow2,
        GQA_RATIO=gqa_ratio,
        KV_SPLITS=kv_splits,
        WRITE_DIRECT=write_direct,
        USE_BT_PREFETCH=use_bt_prefetch,
        Q_IS_ROTATED=q_is_rotated,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if write_direct:
        # out_direct already contains normalised bf16 rotated output.
        out_rot = out_direct
    else:
        # Merge kernel always writes bfloat16.
        out_rot = torch.empty(
            (num_seqs, num_q_heads, head_size),
            dtype=torch.bfloat16, device=query.device,
        )
        # H100 sweeps favored 4 merge warps across the GPT-OSS gate workloads.
        merge_warps = (
            merge_num_warps
            if merge_num_warps is not None
            else 4
        )
        _tq_merge_splits_kernel[(num_seqs, num_q_heads)](
            partial_acc, partial_m, partial_l, out_rot,
            s_oa_s, s_oa_h, s_oa_k,
            s_ml_s, s_ml_h,
            out_rot.stride(0), out_rot.stride(1),
            HEAD_SIZE=head_size,
            HALF=half,
            KV_SPLITS=kv_splits,
            num_warps=merge_warps,
            num_stages=1,
        )

    if skip_output_inverse_rotation:
        return out_rot.to(dtype)

    # Apply inverse rotation R^T to recover unrotated V accumulation.
    # BF16 tensor-core path: same precision argument as Q rotation above.
    out = (out_rot.reshape(-1, head_size) @ rotation.t()).to(dtype)
    return out.reshape(num_seqs, num_q_heads, head_size)
