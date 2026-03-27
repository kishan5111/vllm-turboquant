# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TurboQuant KV-cache attention backend for vLLM.

Stores K and V in a 4-bit compressed format (random rotation + uniform
4-bit quantisation) that gives ~3.9× memory reduction vs BF16.

During the forward pass:
  1. New K/V tokens are rotated (R @ k/v) and compressed into the uint8
     cache via a Triton kernel (write path).
  2. The unique cache blocks referenced by the current batch's block table
     are decompressed into a temporary BF16/FP16 buffer on-the-fly.
  3. Standard FlashAttention runs against the decompressed buffer using a
     remapped block table.

Reference: Zandieh et al., arXiv 2504.19874 (ICLR 2026).
"""

import os
from dataclasses import replace
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.triton_turboquant_kv import (
    apply_rotation,
    make_turboquant_rotation,
    turboquant_compress_kv,
    turboquant_decompress_blocks,
)
# QJL ops are an optional prototype that requires an external kernel build.
# Import lazily so that turboquant_4bit works without the QJL package.
try:
    from vllm.v1.attention.ops.qjl_decode_proto import (
        cuda_quantized_bmm_gqa_dynamic,
        decode_packed_qjl,
        make_qjl_projection,
        pack_full_blocks_to_packed_qjl_cache,
        packed_qjl_prefix_scores,
        qjl_packed_slot_size,
    )
    _QJL_AVAILABLE = True
except ImportError:
    _QJL_AVAILABLE = False
from vllm.v1.attention.ops.triton_turboquant_paged_attn import (
    turboquant_fused_paged_decode,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_scheduler_metadata,
    )


def _turboquant_bits(cache_dtype: str) -> int:
    """Extract bit-width from e.g. 'turboquant_4bit' → 4."""
    return int(cache_dtype.split("_")[1].replace("bit", ""))


def turboquant_comp_head_size(head_size: int, bits: int = 4) -> int:
    """Compressed bytes per (head, token): packed nibbles + 2-byte scale."""
    return (head_size * bits + 7) // 8 + 2


def turboquant_qjl_slot_size(head_size: int) -> int:
    if not _QJL_AVAILABLE:
        raise RuntimeError("QJL kernel not available. Build /workspace/QJL/qjl_kernel.")
    return qjl_packed_slot_size(  # type: ignore[name-defined]
        sketch_dim=256,
        outlier_count=8,
        head_size=head_size,
        value_group_size=32,
        value_bits=2,
    )


def _fold_local_v_proj_weight_(
    qkv_weight: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotation: torch.Tensor,
) -> None:
    q_span = num_heads * head_size
    k_span = num_kv_heads * head_size
    v_offset = q_span + k_span
    v_span = num_kv_heads * head_size
    if v_span == 0:
        return

    v_weight = qkv_weight.narrow(0, v_offset, v_span)
    v_heads = v_weight.view(num_kv_heads, head_size, -1)
    folded = torch.matmul(
        rotation.t().to(torch.float32).unsqueeze(0),
        v_heads.to(torch.float32),
    ).to(v_weight.dtype)
    v_heads.copy_(folded)


def _fold_separate_v_proj_weight_(
    v_weight: torch.Tensor,
    *,
    num_kv_heads: int,
    head_size: int,
    rotation: torch.Tensor,
) -> None:
    """Fold rotation into a separate (non-fused) V projection weight.

    V weight shape: [num_kv_heads * head_size, hidden_size]
    Rotation is applied to the output (left) side: folded = R @ v_heads
    where v_heads = v_weight.view(num_kv_heads, head_size, hidden_size).
    """
    if v_weight.shape[0] == 0:
        return
    v_heads = v_weight.view(num_kv_heads, head_size, -1)
    folded = torch.matmul(
        rotation.t().to(torch.float32).unsqueeze(0),
        v_heads.to(torch.float32),
    ).to(v_heads.dtype)
    v_heads.copy_(folded)


def _fold_local_o_proj_weight_(
    o_weight: torch.Tensor,
    *,
    num_heads: int,
    head_size: int,
    rotation: torch.Tensor,
) -> None:
    if num_heads == 0:
        return

    o_heads = o_weight.view(o_weight.shape[0], num_heads, head_size)
    folded = torch.matmul(
        o_heads.to(torch.float32),
        rotation.to(torch.float32),
    ).to(o_weight.dtype)
    o_heads.copy_(folded)


def maybe_fold_turboquant_value_output_projections(
    module: torch.nn.Module,
    act_dtype: torch.dtype,
) -> bool:
    """Fold TurboQuant's V/output rotations into model weights when safe.

    Q/K folding is intentionally excluded because many modern decoder stacks,
    including Qwen3, apply q_norm / k_norm and RoPE between the projections
    and the attention backend.
    """
    attn_layer = getattr(module, "attn", None)
    if attn_layer is None:
        return False
    impl = getattr(attn_layer, "impl", None)
    if impl is None:
        return False
    if not isinstance(impl, TurboQuantAttentionImpl):
        return False
    if impl._use_qjl:
        return False

    qkv_proj = getattr(module, "qkv_proj", None)
    o_proj = getattr(module, "o_proj", None)
    # GPT-OSS and some other models use separate q/k/v projections.
    q_proj_separate = getattr(module, "q_proj", None)
    v_proj_separate = getattr(module, "v_proj", None)
    if qkv_proj is None and (q_proj_separate is None or v_proj_separate is None):
        return False
    if o_proj is None:
        return False

    if qkv_proj is not None:
        qm = getattr(qkv_proj, "quant_method", None)
        if qm is not None and not isinstance(qm, UnquantizedLinearMethod):
            return False
    o_qm = getattr(o_proj, "quant_method", None)
    if o_qm is not None and not isinstance(o_qm, UnquantizedLinearMethod):
        return False

    # Accept both "standard" names (num_heads/num_kv_heads) and GPT-OSS names
    # (num_attention_heads / num_local_key_value_heads).
    head_size = getattr(module, "head_dim", None)
    num_heads = (
        getattr(module, "num_heads", None)
        or getattr(module, "num_attention_heads", None)
    )
    num_kv_heads = (
        getattr(module, "num_kv_heads", None)
        or getattr(module, "num_local_key_value_heads", None)
    )
    if not all(isinstance(v, int) for v in (head_size, num_heads, num_kv_heads)):
        return False
    if head_size != impl._real_head_size:
        return False

    weight_dtype = (
        qkv_proj.weight.dtype
        if qkv_proj is not None and qkv_proj.weight.is_floating_point()
        else act_dtype
    )
    rotation = make_turboquant_rotation(
        head_size,
        weight_dtype,
        qkv_proj.weight.device if qkv_proj is not None else v_proj_separate.weight.device,
        seed=impl._rotation_seed,
    )

    with torch.no_grad():
        if qkv_proj is not None:
            # Fused QKV: fold V from the V portion
            _fold_local_v_proj_weight_(
                qkv_proj.weight,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                rotation=rotation,
            )
        else:
            # Separate V projection: fold V directly
            _fold_separate_v_proj_weight_(
                v_proj_separate.weight,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                rotation=rotation,
            )
        _fold_local_o_proj_weight_(
            o_proj.weight,
            num_heads=num_heads,
            head_size=head_size,
            rotation=rotation,
        )

    impl._value_output_folded = True
    return True


class TurboQuantAttentionBackend(FlashAttentionBackend):
    """Attention backend that stores K/V in TurboQuant 4-bit compressed form.

    Inherits all scheduling / metadata machinery from FlashAttentionBackend
    and overrides only the shape and dtype of the KV cache together with the
    forward implementation.
    """

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_4bit",
        "turboquant_qjl",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TurboQuantAttentionMetadataBuilder"]:
        return TurboQuantAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_4bit",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "turboquant_qjl":
            if block_size % 32 != 0:
                raise ValueError("TurboQuant QJL requires block size 32-aligned.")
            return (1, num_blocks, num_kv_heads, block_size, head_size)
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # head_size here is already the compressed byte count per head
        # (set by get_kv_cache_spec via turboquant_comp_head_size), so
        # use it directly — do NOT compress again.
        # Coalesced layout: (block, kv_head, position, byte).
        # All BLOCK_SIZE positions for a given (block, kv_head) are
        # contiguous → single HBM transaction per block in the decode kernel.
        return (2, num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: "CacheDType | None") -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        # Decode-only (max_query_len=1) fused kernel is CUDA-graph-capturable.
        # Mixed prefill/decode falls back to decompress+FlashAttention path.
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        # Requires Ampere (sm_80) or newer for BF16 / fast atomics.
        return capability >= DeviceCapability(8, 0)


class TurboQuantAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    """FlashAttentionMetadataBuilder that reports BF16 to the FA scheduler.

    The physical KV cache is uint8 (compressed), but FlashAttention's
    AOT scheduler only accepts fp16/bf16/fp8 as ``qkv_dtype``.  We tell
    it BF16 because that is what the decompressed tensors will be.

    CUDA graph capture is disabled because the Triton decompression
    kernels cannot be captured.
    """

    # Decode-only batches (max_query_len=1) use the fused kernel
    # which is CUDA-graph-capturable.
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Override: the *physical* spec dtype is uint8, but the effective
        # compute dtype is bfloat16 (decompressed before flash_attn).
        self.kv_cache_dtype = torch.bfloat16


_turboquant_layer_counter: int = 0  # incremented per TurboQuantAttentionImpl


class TurboQuantAttentionImpl(FlashAttentionImpl):
    """FlashAttention impl that compresses K/V via TurboQuant before caching.

    Key differences from FlashAttentionImpl:
      - ``do_kv_cache_update``: rotates + compresses to uint8 cache.
      - ``forward``: decompresses only the used blocks before calling
        flash_attn, passing a remapped block table.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "turboquant_4bit",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        self._use_qjl = kv_cache_dtype == "turboquant_qjl"
        # Store real (uncompressed) head size BEFORE calling super().__init__,
        # which also sets self.head_size via FlashAttentionImpl.
        self._real_head_size = head_size

        if self._use_qjl:
            self._turboquant_bits = 2
            self._comp_head_size = turboquant_qjl_slot_size(head_size)
        else:
            bits = _turboquant_bits(kv_cache_dtype)
            self._turboquant_bits = bits
            self._comp_head_size = turboquant_comp_head_size(head_size, bits)

        # Initialise parent with kv_cache_dtype="auto" so that flash_attn
        # doesn't try to handle fp8 scales for the *decompressed* tensors.
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype="auto",          # flash_attn sees plain BF16
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            **kwargs,
        )
        # Restore the turboquant dtype string for our own use.
        self.kv_cache_dtype = kv_cache_dtype

        # Per-layer rotation matrix R ∈ ℝ^{D×D} (orthogonal).
        # Lazily initialised on first use to get the correct device/dtype.
        # Seed is deterministic (based on creation order) so the same model
        # always uses the same rotations regardless of Python object addresses.
        global _turboquant_layer_counter
        _turboquant_layer_counter += 1
        self._rotation_seed: int = _turboquant_layer_counter
        self._rotation: torch.Tensor | None = None
        self._rotate_inside_decode = (
            os.environ.get("TURBOQUANT_ROTATE_INSIDE_DECODE", "0") == "1"
        )
        self._value_output_folded = False
        self._qjl_proj_score: torch.Tensor | None = None
        self._qjl_proj_quant: torch.Tensor | None = None
        self._qjl_outlier_count = 8
        self._qjl_value_bits = 2
        self._qjl_value_group_size = 32
        self._qjl_exact_key_blocks: dict[int, torch.Tensor] = {}
        self._qjl_exact_value_blocks: dict[int, torch.Tensor] = {}
        self._qjl_exact_lengths: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Rotation matrix — lazy init
    # ------------------------------------------------------------------

    def _get_rotation(
        self, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if (
            self._rotation is None
            or self._rotation.device != device
            or self._rotation.dtype != dtype
        ):
            self._rotation = make_turboquant_rotation(
                self._real_head_size, dtype, device, seed=self._rotation_seed
            )
        return self._rotation

    def _get_qjl_projection(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._qjl_proj_score is None
            or self._qjl_proj_quant is None
            or self._qjl_proj_score.device != device
        ):
            self._qjl_proj_score, self._qjl_proj_quant = make_qjl_projection(
                self._real_head_size,
                256,
                dtype=torch.float32,
                device=device,
                seed=self._rotation_seed,
            )
        return self._qjl_proj_score, self._qjl_proj_quant

    # ------------------------------------------------------------------
    # Write path — compress K/V into uint8 cache
    # ------------------------------------------------------------------

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,    # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        if self._use_qjl:
            self._do_qjl_kv_cache_update(key, value, kv_cache, slot_mapping)
            return

        key_cache, value_cache = kv_cache.unbind(0)
        # key_cache shape: [num_blocks, block_size, num_kv_heads, comp_head] UINT8

        R = self._get_rotation(key.dtype, key.device)

        turboquant_compress_kv(
            key,
            value,
            key_cache, value_cache,
            slot_mapping,
            key_rotation=R,
            value_rotation=None if self._value_output_folded else R,
        )

    def _do_qjl_kv_cache_update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        packed_cache = kv_cache[0]
        block_size = packed_cache.shape[2]

        valid_mask = slot_mapping >= 0
        if not torch.any(valid_mask):
            return

        key = key[valid_mask]
        value = value[valid_mask]
        slot_mapping = slot_mapping[valid_mask].to(torch.long)

        block_ids = torch.div(slot_mapping, block_size, rounding_mode="floor")
        block_pos = slot_mapping.remainder(block_size)
        order = torch.argsort(block_ids * block_size + block_pos)
        key = key[order]
        value = value[order]
        block_ids = block_ids[order]
        block_pos = block_pos[order]

        uniq_blocks, counts = torch.unique_consecutive(block_ids, return_counts=True)
        expected_pos = torch.arange(block_size, device=key.device, dtype=block_pos.dtype)
        full_block_ids: list[int] = []
        full_key_blocks: list[torch.Tensor] = []
        full_value_blocks: list[torch.Tensor] = []

        start = 0
        for block_id_t, count_t in zip(uniq_blocks, counts):
            block_id = int(block_id_t.item())
            count = int(count_t.item())
            end = start + count
            pos = block_pos[start:end]
            key_group = key[start:end]
            value_group = value[start:end]
            start = end

            is_full_group = (
                count == block_size
                and torch.equal(pos, expected_pos)
                and block_id not in self._qjl_exact_key_blocks
            )
            if is_full_group:
                full_block_ids.append(block_id)
                full_key_blocks.append(key_group.permute(1, 0, 2).contiguous())
                full_value_blocks.append(value_group.permute(1, 0, 2).contiguous())
                continue

            if block_id not in self._qjl_exact_key_blocks or int(pos[0].item()) == 0:
                self._qjl_exact_key_blocks[block_id] = torch.empty(
                    self.num_kv_heads,
                    block_size,
                    self._real_head_size,
                    device=key.device,
                    dtype=key.dtype,
                )
                self._qjl_exact_value_blocks[block_id] = torch.empty(
                    self.num_kv_heads,
                    block_size,
                    self._real_head_size,
                    device=value.device,
                    dtype=value.dtype,
                )
                self._qjl_exact_lengths[block_id] = 0

            self._qjl_exact_key_blocks[block_id][:, pos] = key_group.permute(1, 0, 2)
            self._qjl_exact_value_blocks[block_id][:, pos] = value_group.permute(1, 0, 2)
            new_len = max(
                self._qjl_exact_lengths[block_id],
                int(pos.max().item()) + 1,
            )
            self._qjl_exact_lengths[block_id] = new_len
            if new_len == block_size:
                full_block_ids.append(block_id)
                full_key_blocks.append(self._qjl_exact_key_blocks.pop(block_id))
                full_value_blocks.append(self._qjl_exact_value_blocks.pop(block_id))
                self._qjl_exact_lengths.pop(block_id, None)

        if full_block_ids:
            _, proj_dir_quant = self._get_qjl_projection(key.device)
            pack_full_blocks_to_packed_qjl_cache(
                torch.stack(full_key_blocks, dim=0),
                torch.stack(full_value_blocks, dim=0),
                packed_cache,
                torch.tensor(full_block_ids, device=key.device, dtype=torch.long),
                proj_dir_quant=proj_dir_quant,
                value_group_size=self._qjl_value_group_size,
                value_bits=self._qjl_value_bits,
                outlier_count=self._qjl_outlier_count,
            )

    # ------------------------------------------------------------------
    # Read path — decompress used blocks then run flash_attn
    # ------------------------------------------------------------------

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None

        if attn_metadata is None:
            return output.fill_(0)

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[: attn_metadata.num_actual_tokens],
                key[: attn_metadata.num_actual_tokens],
                value[: attn_metadata.num_actual_tokens],
                output[: attn_metadata.num_actual_tokens],
                attn_metadata,
                layer,
            )

        if self._use_qjl:
            if attn_metadata.max_query_len == 1 and not attn_metadata.use_cascade:
                return self._forward_qjl_decode(query, kv_cache, attn_metadata, output)
            if self._qjl_metadata_has_prefix(attn_metadata):
                raise NotImplementedError(
                    "TurboQuant QJL currently requires non-chunked prefill."
                )
            return self._forward_qjl_prefill(
                layer=layer,
                query=query[: attn_metadata.num_actual_tokens],
                key=key[: attn_metadata.num_actual_tokens],
                value=value[: attn_metadata.num_actual_tokens],
                output=output[: attn_metadata.num_actual_tokens],
                attn_metadata=attn_metadata,
            )

        # Fast path: pure decode batch — use fused kernel (no decompress-to-BF16 pass)
        if attn_metadata.max_query_len == 1 and not attn_metadata.use_cascade:
            return self._forward_fused_decode(
                query, kv_cache, attn_metadata, output
            )

        comp_key_cache, comp_value_cache = kv_cache.unbind(0)
        # comp_key_cache: [num_blocks, num_kv_heads, block_size, comp_head] UINT8
        # (coalesced layout: kv_head before position)

        num_all_blocks = comp_key_cache.shape[0]
        block_size     = comp_key_cache.shape[2]  # dim 2 in new layout

        # ---- 1. Gather the unique block indices needed this step ----
        block_table = attn_metadata.block_table  # [batch, max_blocks]
        flat = block_table.reshape(-1)
        valid = flat[flat >= 0]
        unique_blk_ids = torch.unique(valid)          # [num_unique]
        num_unique = unique_blk_ids.shape[0]

        # ---- 2. Decompress those blocks into a temporary BF16 buffer ----
        decomp_key = torch.empty(
            (num_unique, block_size, self.num_kv_heads, self._real_head_size),
            dtype=query.dtype, device=query.device,
        )
        decomp_value = torch.empty_like(decomp_key)

        turboquant_decompress_blocks(comp_key_cache, unique_blk_ids, decomp_key)
        turboquant_decompress_blocks(comp_value_cache, unique_blk_ids, decomp_value)

        # Keep decompressed K/V in rotated space and rotate the current query
        # chunk once. For long-context prefill this is much cheaper than
        # derotating every referenced KV block back to the original basis.
        R = self._get_rotation(query.dtype, query.device)
        query_rot = apply_rotation(query, R)

        # ---- 3. Remap block_table to indices in [0, num_unique) ----
        if num_unique > 0:
            max_blk = int(unique_blk_ids.max().item())
        else:
            max_blk = 0
        remap = torch.full(
            (max_blk + 2,), -1, dtype=block_table.dtype, device=block_table.device
        )
        remap[unique_blk_ids] = torch.arange(
            num_unique, dtype=block_table.dtype, device=block_table.device
        )
        remapped_bt = remap[block_table.clamp(min=0)]
        remapped_bt[block_table < 0] = -1

        # ---- 4. Run standard FlashAttention on the decompressed buffers ----
        new_metadata = replace(attn_metadata, block_table=remapped_bt)

        # Temporarily swap the backend's kv_cache_dtype so super().forward
        # does not try FP8 descale logic.
        orig_dtype = self.kv_cache_dtype
        self.kv_cache_dtype = "auto"

        # Fake a kv_cache from the decompressed tensors so that super().forward
        # can unbind(0) it correctly.
        fake_kv_cache = torch.stack([decomp_key, decomp_value], dim=0)

        result = super().forward(
            layer=layer,
            query=query_rot,
            key=key,
            value=value,
            kv_cache=fake_kv_cache,
            attn_metadata=new_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

        self.kv_cache_dtype = orig_dtype
        if not self._value_output_folded:
            num_actual = attn_metadata.num_actual_tokens
            result[:num_actual].copy_(apply_rotation(result[:num_actual], R.T))
        return result

    def _qjl_metadata_has_prefix(self, attn_metadata: FlashAttentionMetadata) -> bool:
        query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        seq_lens = attn_metadata.seq_lens[: query_lens.shape[0]]
        return bool(torch.any(seq_lens > query_lens).item())

    def _forward_fused_decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode attention via fused kernel — reads compressed K/V from HBM directly."""
        comp_key_cache, comp_value_cache = kv_cache.unbind(0)
        R = self._get_rotation(query.dtype, query.device)

        num_actual = attn_metadata.num_actual_tokens
        # seq_lens gives total KV length for each decode sequence
        seq_lens  = attn_metadata.seq_lens[:num_actual]
        block_table = attn_metadata.block_table[:num_actual]

        # query layout: [num_tokens, num_heads, head_size] — for decode num_tokens == num_seqs
        result = turboquant_fused_paged_decode(
            query=query[:num_actual],
            key_cache=comp_key_cache,
            value_cache=comp_value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            rotation=R,
            scale=self.scale,
            skip_output_inverse_rotation=self._value_output_folded,
            rotate_inside_decode=self._rotate_inside_decode,
        )

        output[:num_actual].copy_(result)
        return output

    def _forward_qjl_decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        packed_cache = kv_cache[0]
        proj_dir_score, _ = self._get_qjl_projection(query.device)

        num_actual = attn_metadata.num_actual_tokens
        query = query[:num_actual]
        seq_lens = attn_metadata.seq_lens[:num_actual]
        block_table = attn_metadata.block_table[:num_actual]

        if not self._qjl_exact_key_blocks:
            result = decode_packed_qjl(
                query,
                packed_cache=packed_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                proj_dir_score=proj_dir_score,
                scale=self.scale,
                outlier_count=self._qjl_outlier_count,
                value_group_size=self._qjl_value_group_size,
                value_bits=self._qjl_value_bits,
            )
            output[:num_actual].copy_(result)
            return output

        batch = num_actual
        q_per_kv = self.num_heads // self.num_kv_heads
        exact_block_ids = set(self._qjl_exact_key_blocks)

        comp_rows: list[list[int]] = []
        exact_key_rows: list[torch.Tensor] = []
        exact_value_rows: list[torch.Tensor] = []
        exact_lens: list[int] = []
        max_comp_blocks = 0
        max_exact_tokens = 0

        for row in block_table:
            comp_ids: list[int] = []
            exact_k_chunks: list[torch.Tensor] = []
            exact_v_chunks: list[torch.Tensor] = []
            for block_id in row[row >= 0].tolist():
                if block_id in exact_block_ids:
                    exact_len = self._qjl_exact_lengths[block_id]
                    exact_k_chunks.append(
                        self._qjl_exact_key_blocks[block_id][:, :exact_len]
                    )
                    exact_v_chunks.append(
                        self._qjl_exact_value_blocks[block_id][:, :exact_len]
                    )
                else:
                    comp_ids.append(block_id)
            comp_rows.append(comp_ids)
            max_comp_blocks = max(max_comp_blocks, len(comp_ids))
            if exact_k_chunks:
                exact_key = torch.cat(exact_k_chunks, dim=1)
                exact_value = torch.cat(exact_v_chunks, dim=1)
            else:
                exact_key = query.new_empty(
                    self.num_kv_heads, 0, self._real_head_size
                )
                exact_value = query.new_empty(
                    self.num_kv_heads, 0, self._real_head_size
                )
            exact_key_rows.append(exact_key)
            exact_value_rows.append(exact_value)
            exact_lens.append(exact_key.shape[1])
            max_exact_tokens = max(max_exact_tokens, exact_key.shape[1])

        prefix_lens = seq_lens - torch.tensor(
            exact_lens, device=query.device, dtype=seq_lens.dtype
        )
        if max_comp_blocks > 0:
            comp_block_table = torch.zeros(
                batch, max_comp_blocks, device=query.device, dtype=block_table.dtype
            )
            for i, comp_ids in enumerate(comp_rows):
                if comp_ids:
                    comp_block_table[i, : len(comp_ids)] = torch.tensor(
                        comp_ids, device=query.device, dtype=block_table.dtype
                    )
            prefix_scores, value_pack, value_scale, value_min = packed_qjl_prefix_scores(
                query,
                packed_cache=packed_cache,
                block_table=comp_block_table,
                seq_lens=prefix_lens,
                proj_dir_score=proj_dir_score,
                scale=self.scale,
                outlier_count=self._qjl_outlier_count,
                value_group_size=self._qjl_value_group_size,
                value_bits=self._qjl_value_bits,
            )
        else:
            prefix_scores = query.new_empty((batch, self.num_heads, 0), dtype=torch.float32)
            value_pack = query.new_empty(
                (batch, self.num_kv_heads, 0, self._real_head_size * self._qjl_value_bits // 32),
                dtype=torch.int32,
            )
            value_scale = query.new_empty(
                (batch, self.num_kv_heads, 0, self._real_head_size // self._qjl_value_group_size)
            )
            value_min = torch.empty_like(value_scale)

        if max_exact_tokens > 0:
            exact_key = query.new_zeros(
                (batch, self.num_kv_heads, max_exact_tokens, self._real_head_size)
            )
            exact_value = torch.zeros_like(exact_key)
            for i, (k_row, v_row) in enumerate(zip(exact_key_rows, exact_value_rows)):
                if k_row.shape[1] > 0:
                    exact_key[i, :, : k_row.shape[1]] = k_row
                    exact_value[i, :, : v_row.shape[1]] = v_row
            exact_key = exact_key.repeat_interleave(q_per_kv, dim=1)
            exact_value = exact_value.repeat_interleave(q_per_kv, dim=1)
            exact_scores = torch.einsum("bhd,bhkd->bhk", query, exact_key) * self.scale
            exact_idx = torch.arange(
                max_exact_tokens, device=query.device, dtype=seq_lens.dtype
            )
            exact_valid = exact_idx.unsqueeze(0) < torch.tensor(
                exact_lens, device=query.device, dtype=seq_lens.dtype
            ).unsqueeze(1)
            exact_scores = exact_scores.masked_fill(
                ~exact_valid.unsqueeze(1), float("-inf")
            )
        else:
            exact_scores = query.new_empty((batch, self.num_heads, 0), dtype=torch.float32)
            exact_value = query.new_empty(
                (batch, self.num_heads, 0, self._real_head_size)
            )

        all_scores = torch.cat([prefix_scores, exact_scores], dim=-1)
        all_weights = torch.softmax(all_scores, dim=-1, dtype=torch.float32).to(query.dtype)

        result = query.new_zeros((batch, self.num_heads, self._real_head_size))
        prefix_tokens = prefix_scores.shape[-1]
        if prefix_tokens > 0:
            result += cuda_quantized_bmm_gqa_dynamic(
                self._qjl_value_group_size,
                all_weights[..., :prefix_tokens].unsqueeze(2),
                value_pack,
                value_scale,
                value_min,
                self._qjl_value_bits,
            ).squeeze(2)
        if max_exact_tokens > 0:
            result += torch.einsum(
                "bhk,bhkd->bhd",
                all_weights[..., prefix_tokens:],
                exact_value,
            )

        output[:num_actual].copy_(result)
        return output

    def _forward_qjl_prefill(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> torch.Tensor:
        assert self.vllm_flash_attn_version is not None
        cu_seqlens = attn_metadata.query_start_loc
        max_seqlen = attn_metadata.max_query_len
        descale_shape = (cu_seqlens.shape[0] - 1, self.num_kv_heads)
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=1 if self.batch_invariant_enabled else 0,
        )
        return output
