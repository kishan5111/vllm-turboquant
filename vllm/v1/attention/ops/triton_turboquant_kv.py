# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TurboQuant KV-cache compression / decompression Triton kernels.

Implements the MSE-optimal stage of TurboQuant (called "PolarQuant" in the
paper) for 4-bit KV-cache quantisation:

  1. Apply a per-layer random rotation  R ∈ ℝ^{D×D}  (stored once as a
     model buffer) to the head vector:  x_rot = R @ x
  2. Compute a per-head L∞ scale:  scale = max|x_rot| / 7.5
  3. Quantise each element to a 4-bit code in [0, 15]:
       code = round(x_rot / scale * 7.5 + 7.5)
  4. Pack two codes per uint8 byte (first-half → low nibble,
                                    second-half → high nibble).
  5. Append 2 bytes (float16) for the scale.

Compressed layout per (token, kv-head)  [AoS]:
  [ D//2  bytes : packed 4-bit codes ][ 2 bytes : float16 scale ]
  Total = D//2 + 2  bytes   (vs  D*2  bytes for BF16 → ~3.9× compression)

The AoS (Array of Structures) layout keeps all data for one (token, kv-head)
contiguous, giving a non-power-of-2 stride between positions (COMP_HEAD = HALF+2).
This stride prevents Triton from treating the [BLOCK_SIZE, HALF] nibble load as
a single contiguous 1024-byte access, which would increase register pressure
with num_stages=4 software pipelining.  The row-by-row load pattern is
preferable for pipeline-friendly warp execution.

The rotation R is applied in Python via a matmul before calling these
kernels so that it can be fused with the preceding linear projections.

Reference:
  Zandieh et al., "TurboQuant: Online Vector Quantization with
  Near-optimal Distortion Rate", arXiv 2504.19874, ICLR 2026.
"""

import math

import torch

from vllm.triton_utils import tl, triton


# ---------------------------------------------------------------------------
# Compress kernel (write path)
# ---------------------------------------------------------------------------

@triton.jit
def _turboquant_pack_kernel(
    kv_ptr,            # [num_tokens, num_heads, head_size]  BF16/FP16  (rotated)
    cache_ptr,         # [num_blocks, num_kv_heads, block_size, comp_head]  UINT8
    slot_ptr,          # [num_tokens]  INT64 slot indices
    num_heads,         # runtime int
    block_size,        # runtime int
    stride_t: tl.int64,   # kv_ptr: stride over tokens
    stride_h: tl.int64,   # kv_ptr: stride over heads
    stride_cb: tl.int64,  # cache_ptr: stride over blocks
    stride_cp: tl.int64,  # cache_ptr: stride over page positions
    stride_ch: tl.int64,  # cache_ptr: stride over heads
    HEAD_SIZE: tl.constexpr,   # must be even and ≥ 2
    COMP_HEAD: tl.constexpr,   # HEAD_SIZE // 2 + 2
):
    """Pack one (token, head) pair into the compressed cache slot."""
    pid_t = tl.program_id(0)   # token index
    pid_h = tl.program_id(1)   # kv-head index

    slot = tl.load(slot_ptr + pid_t).to(tl.int64)
    if slot < 0:
        return

    blk = slot // block_size
    off = slot % block_size

    HALF: tl.constexpr = HEAD_SIZE // 2
    half_offs = tl.arange(0, HALF)

    src_base = pid_t * stride_t + pid_h * stride_h

    # Load first half and second half of the rotated head vector.
    lo_fp = tl.load(kv_ptr + src_base + half_offs).to(tl.float32)
    hi_fp = tl.load(kv_ptr + src_base + half_offs + HALF).to(tl.float32)

    # Per-head scale = max absolute value.  Maps [-scale, +scale] ↔ [0, 15].
    abs_max = tl.maximum(tl.max(tl.abs(lo_fp)), tl.max(tl.abs(hi_fp)))
    scale = tl.where(abs_max > 1e-8, abs_max, 1.0)

    # Encode:  code = round( x / scale * 7.5 + 7.5 )  ∈ [0, 15]
    # floor(x + 0.5) is the portable round-to-nearest in Triton.
    lo_q = tl.maximum(0.0, tl.minimum(15.0,
                       tl.floor(lo_fp / scale * 7.5 + 7.5 + 0.5))).to(tl.uint8)
    hi_q = tl.maximum(0.0, tl.minimum(15.0,
                       tl.floor(hi_fp / scale * 7.5 + 7.5 + 0.5))).to(tl.uint8)

    # Pack: low nibble ← first-half code, high nibble ← second-half code.
    packed = (lo_q & 0x0F) | ((hi_q & 0x0F) << 4)

    dst_base = blk * stride_cb + off * stride_cp + pid_h * stride_ch

    # Store packed nibbles.
    tl.store(cache_ptr + dst_base + half_offs, packed)

    # Store scale as float16 (2 bytes, little-endian).
    scale_u16 = scale.to(tl.float16).to(tl.uint16, bitcast=True)
    tl.store(cache_ptr + dst_base + HALF,
             (scale_u16 & 0xFF).to(tl.uint8))
    tl.store(cache_ptr + dst_base + HALF + 1,
             ((scale_u16 >> 8) & 0xFF).to(tl.uint8))


def turboquant_compress_kv(
    key_rot: torch.Tensor,    # [num_tokens, num_kv_heads, head_size]  BF16/FP16
    value_rot: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]  BF16/FP16
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, comp_head]  UINT8
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens]  INT64
) -> None:
    """Compress rotated K/V tensors into the uint8 KV cache in-place."""
    num_tokens, num_heads, head_size = key_rot.shape
    # Cache layout: [num_blocks, num_kv_heads, block_size, comp_head]
    block_size = key_cache.shape[2]   # block_size is now dim 2
    comp_head  = key_cache.shape[3]   # comp_head stays dim 3

    assert head_size % 2 == 0, "head_size must be even for TurboQuant 4-bit"
    assert comp_head == head_size // 2 + 2, (
        f"Expected comp_head={head_size // 2 + 2}, got {comp_head}")

    if num_tokens == 0:
        return

    grid = (num_tokens, num_heads)

    # Cache layout is now [num_blocks, num_kv_heads, block_size, comp_head].
    # stride(0)=block, stride(1)=kv_head, stride(2)=position, stride(3)=byte.
    _turboquant_pack_kernel[grid](
        kv_ptr=key_rot,
        cache_ptr=key_cache,
        slot_ptr=slot_mapping,
        num_heads=num_heads,
        block_size=block_size,
        stride_t=key_rot.stride(0),
        stride_h=key_rot.stride(1),
        stride_cb=key_cache.stride(0),
        stride_cp=key_cache.stride(2),   # position is now dim 2
        stride_ch=key_cache.stride(1),   # kv-head is now dim 1
        HEAD_SIZE=head_size,
        COMP_HEAD=comp_head,
        num_warps=4,
        num_stages=1,
    )

    _turboquant_pack_kernel[grid](
        kv_ptr=value_rot,
        cache_ptr=value_cache,
        slot_ptr=slot_mapping,
        num_heads=num_heads,
        block_size=block_size,
        stride_t=value_rot.stride(0),
        stride_h=value_rot.stride(1),
        stride_cb=value_cache.stride(0),
        stride_cp=value_cache.stride(2),   # position is now dim 2
        stride_ch=value_cache.stride(1),   # kv-head is now dim 1
        HEAD_SIZE=head_size,
        COMP_HEAD=comp_head,
        num_warps=4,
        num_stages=1,
    )


# ---------------------------------------------------------------------------
# Decompress kernel (read path)
# ---------------------------------------------------------------------------

@triton.jit
def _turboquant_unpack_kernel(
    cache_ptr,        # [num_src_blocks, block_size, num_heads, comp_head]  UINT8
    out_ptr,          # [num_dst_blocks, block_size, num_heads, head_size]  BF16/FP16
    block_ids_ptr,    # [num_dst_blocks]  INT64  which source blocks to unpack
    block_size,       # runtime int
    num_heads,        # runtime int
    stride_cb: tl.int64,   # cache_ptr strides
    stride_cp: tl.int64,
    stride_ch: tl.int64,
    stride_ob: tl.int64,   # out_ptr strides
    stride_op: tl.int64,
    stride_oh: tl.int64,
    HEAD_SIZE: tl.constexpr,
    COMP_HEAD: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Decompress one (dst_block_idx, page_offset, head) triple."""
    dst_blk  = tl.program_id(0)
    page_off = tl.program_id(1)
    head_idx = tl.program_id(2)

    src_blk = tl.load(block_ids_ptr + dst_blk).to(tl.int64)

    HALF: tl.constexpr = HEAD_SIZE // 2
    half_offs = tl.arange(0, HALF)

    src_base = src_blk * stride_cb + page_off * stride_cp + head_idx * stride_ch

    # Load packed nibbles.
    packed = tl.load(cache_ptr + src_base + half_offs).to(tl.uint8)
    lo_q = (packed & 0x0F).to(tl.float32)           # first-half codes
    hi_q = ((packed >> 4) & 0x0F).to(tl.float32)    # second-half codes

    # Reconstruct float16 scale from 2 little-endian bytes.
    b0 = tl.load(cache_ptr + src_base + HALF).to(tl.uint16)
    b1 = tl.load(cache_ptr + src_base + HALF + 1).to(tl.uint16)
    scale_bits = b0 | (b1 << 8)
    scale = scale_bits.to(tl.float16, bitcast=True).to(tl.float32)

    # Dequantise.
    lo_fp = (lo_q / 7.5 - 1.0) * scale
    hi_fp = (hi_q / 7.5 - 1.0) * scale

    dst_base = dst_blk * stride_ob + page_off * stride_op + head_idx * stride_oh

    tl.store(out_ptr + dst_base + half_offs,
             lo_fp.to(OUT_DTYPE))
    tl.store(out_ptr + dst_base + half_offs + HALF,
             hi_fp.to(OUT_DTYPE))


def turboquant_decompress_blocks(
    compressed_cache: torch.Tensor,  # [num_all_blocks, block_size, num_heads, comp_head] UINT8
    block_ids: torch.Tensor,         # [num_dst_blocks] INT64 — which blocks to decompress
    output: torch.Tensor,            # [num_dst_blocks, block_size, num_heads, head_size] BF16/FP16
) -> None:
    """Decompress a subset of blocks from the compressed KV cache."""
    num_dst_blocks = block_ids.shape[0]
    if num_dst_blocks == 0:
        return

    # Cache layout: [num_blocks, num_kv_heads, block_size, comp_head]
    num_heads   = compressed_cache.shape[1]   # kv_heads now dim 1
    block_size  = compressed_cache.shape[2]   # block_size now dim 2
    comp_head   = compressed_cache.shape[3]
    head_size   = output.shape[3]

    assert comp_head == head_size // 2 + 2

    OUT_DTYPE = tl.bfloat16 if output.dtype == torch.bfloat16 else tl.float16

    grid = (num_dst_blocks, block_size, num_heads)

    _turboquant_unpack_kernel[grid](
        cache_ptr=compressed_cache,
        out_ptr=output,
        block_ids_ptr=block_ids,
        block_size=block_size,
        num_heads=num_heads,
        stride_cb=compressed_cache.stride(0),
        stride_cp=compressed_cache.stride(2),   # position is dim 2
        stride_ch=compressed_cache.stride(1),   # kv-head is dim 1
        stride_ob=output.stride(0),
        stride_op=output.stride(1),
        stride_oh=output.stride(2),
        HEAD_SIZE=head_size,
        COMP_HEAD=comp_head,
        OUT_DTYPE=OUT_DTYPE,
        num_warps=4,
        num_stages=1,
    )


# ---------------------------------------------------------------------------
# Helper: generate the per-layer random rotation matrix
# ---------------------------------------------------------------------------

def make_turboquant_rotation(head_size: int,
                              dtype: torch.dtype,
                              device: torch.device,
                              seed: int = 0) -> torch.Tensor:
    """Return an orthogonal rotation matrix R ∈ ℝ^{D×D}.

    Generated via QR decomposition of a random Gaussian matrix so that
    applying R randomises the distribution of outliers across dimensions.
    The matrix is deterministic given `seed` and `head_size`.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    # CPU QR is not implemented for BF16, so always construct the rotation in
    # float32 on CPU and cast once the orthogonal basis is formed.
    G = torch.randn(
        head_size,
        head_size,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    Q, _ = torch.linalg.qr(G)        # orthogonal Q
    return Q.to(dtype=dtype, device=device)


def apply_rotation(
    x: torch.Tensor,     # [..., num_heads, head_size]
    R: torch.Tensor,     # [head_size, head_size]
) -> torch.Tensor:
    """Apply rotation matrix R to the last dimension of x via batched matmul."""
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])   # [N * num_heads, head_size]
    x_rot  = x_flat @ R                       # [N * num_heads, head_size]
    return x_rot.reshape(orig_shape)


def apply_rotation_inverse(
    x: torch.Tensor,     # [..., num_heads, head_size]
    R: torch.Tensor,     # [head_size, head_size]  (orthogonal: R^{-1} = R^T)
) -> torch.Tensor:
    """Apply R^T (= R^{-1}) to the last dimension of x."""
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    x_derot = x_flat @ R.T
    return x_derot.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Stage 2: 2-bit PolarQuant + 1-bit JL sketch   (turboquant_2bit_qjl)
# ---------------------------------------------------------------------------
# Storage per (token, kv-head):
#   [D/4 bytes : 2-bit codes, 4 per byte]
#   [D/8 bytes : 1-bit JL sketch, 8 per byte]
#   [2  bytes  : float16 scale]
#   Total = 3D/8 + 2 bytes  (for D=128: 50 bytes vs 256 BF16 → 5.12×)
#
# The JL sketch records sign(residual · p_i) for D/2 random unit vectors p_i.
# At decode, the sketch corrects the 2-bit quantisation error.
# ---------------------------------------------------------------------------

def turboquant_qjl_comp_head_size(head_size: int) -> int:
    """Compressed byte count per (head, token) for 2-bit + JL."""
    return head_size // 4 + head_size // 8 + 2


@triton.jit
def _tq_qjl_pack_kernel(
    kv_ptr,         # [num_tokens, num_heads, head_size]  BF16/FP16 (rotated)
    cache_ptr,      # [num_blocks, block_size, num_heads, comp_head]  UINT8
    sketch_seed_ptr,# [head_size // 2, head_size]  int8  — ±1 JL matrix (precomputed)
    slot_ptr,       # [num_tokens]  INT64
    num_heads,
    block_size,
    stride_t: tl.int64,
    stride_h: tl.int64,
    stride_cb: tl.int64,
    stride_cp: tl.int64,
    stride_ch: tl.int64,
    HEAD_SIZE: tl.constexpr,
    QSIZE: tl.constexpr,     # HEAD_SIZE // 4  (2-bit codes bytes)
    JSIZE: tl.constexpr,     # HEAD_SIZE // 8  (JL sketch bytes)
    COMP_HEAD: tl.constexpr, # QSIZE + JSIZE + 2
):
    """Pack rotated K/V as 2-bit codes + 1-bit JL residual sketch."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    slot = tl.load(slot_ptr + pid_t).to(tl.int64)
    if slot < 0:
        return

    blk = slot // block_size
    off = slot % block_size

    QUARTER: tl.constexpr = HEAD_SIZE // 4
    quarter_offs = tl.arange(0, QUARTER)
    src_base = pid_t * stride_t + pid_h * stride_h

    # Load all four quarters of the rotated head vector
    x0 = tl.load(kv_ptr + src_base + quarter_offs).to(tl.float32)
    x1 = tl.load(kv_ptr + src_base + quarter_offs + QUARTER).to(tl.float32)
    x2 = tl.load(kv_ptr + src_base + quarter_offs + 2 * QUARTER).to(tl.float32)
    x3 = tl.load(kv_ptr + src_base + quarter_offs + 3 * QUARTER).to(tl.float32)

    # Scale = max absolute value
    abs_max = tl.maximum(
        tl.maximum(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))),
        tl.maximum(tl.max(tl.abs(x2)), tl.max(tl.abs(x3))),
    )
    scale = tl.where(abs_max > 1e-8, abs_max, 1.0)

    # 2-bit encode: code ∈ {0,1,2,3}
    # Maps [-scale, +scale] → [0, 3] : code = round(x/scale*1.5 + 1.5)
    def enc2(v):
        return tl.maximum(0.0, tl.minimum(3.0,
               tl.floor(v / scale * 1.5 + 1.5 + 0.5))).to(tl.uint8)

    q0 = enc2(x0)
    q1 = enc2(x1)
    q2 = enc2(x2)
    q3 = enc2(x3)

    # Pack 4 codes per byte: bits [1:0]=q[0], [3:2]=q[1], [5:4]=q[2], [7:6]=q[3]
    packed = (q0 & 0x03) | ((q1 & 0x03) << 2) | ((q2 & 0x03) << 4) | ((q3 & 0x03) << 6)

    dst_base = blk * stride_cb + off * stride_cp + pid_h * stride_ch

    # Write 2-bit codes (QSIZE bytes)
    tl.store(cache_ptr + dst_base + quarter_offs, packed)

    # Decode to compute residual for JL sketch
    def dec2(code):
        return (code.to(tl.float32) / 1.5 - 1.0) * scale

    r0 = x0 - dec2(q0)
    r1 = x1 - dec2(q1)
    r2 = x2 - dec2(q2)
    r3 = x3 - dec2(q3)

    # JL sketch: 1 bit per dimension using a Rademacher (±1) projection
    # We use a fast hash-based ±1 sequence instead of storing the full matrix.
    # sketch[i] = 1 if dot(residual, p_i) > 0, else 0
    # p_i uses a Walsh-Hadamard-style ±1 pattern seeded by i.
    # Store 8 sketch bits per byte.
    HALF_HEAD: tl.constexpr = HEAD_SIZE // 2
    sketch_bytes = JSIZE

    # Simple implementation: compute sketch bits using XOR-based ±1 vectors
    # p_i[j] = +1 if popcount(i & j) is even, else -1
    # This gives a valid JL embedding with E[p_i · p_j] = 0 for i≠j
    jl_byte_offs = tl.arange(0, JSIZE)
    sketch_byte = tl.zeros([JSIZE], dtype=tl.uint8)

    # NOTE: Full JL implementation requires a loop over sketch dimensions.
    # For the initial PR, store zeros (placeholder) and implement full JL
    # in a follow-up once the 2-bit path is validated.
    # The 2-bit quantisation alone gives significant compression improvement.

    tl.store(cache_ptr + dst_base + QSIZE + jl_byte_offs, sketch_byte)

    # Write scale (float16, 2 bytes)
    scale_u16 = scale.to(tl.float16).to(tl.uint16, bitcast=True)
    tl.store(cache_ptr + dst_base + QSIZE + JSIZE,     (scale_u16 & 0xFF).to(tl.uint8))
    tl.store(cache_ptr + dst_base + QSIZE + JSIZE + 1, ((scale_u16 >> 8) & 0xFF).to(tl.uint8))
