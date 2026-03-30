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

import atexit
import json
import os
import time
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
# QJL ops are an optional in-repo prototype. Import lazily so that
# turboquant_4bit works even if the experimental path is broken.
try:
    from vllm.v1.attention.ops.qjl_backend_proto import (
        cuda_quantized_bmm_gqa_dynamic,
        decode_packed_qjl,
        make_qjl_projection,
        pack_tokens_to_packed_qjl_cache,
        packed_qjl_prefix_scores_multi_query,
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


_PROFILE_PREFILL = os.environ.get("TURBOQUANT_PROFILE_PREFILL", "0") == "1"
_PROFILE_PREFILL_FLUSH_EVERY = max(
    1, int(os.environ.get("TURBOQUANT_PROFILE_PREFILL_FLUSH_EVERY", "1"))
)
_PREFILL_TIMINGS: dict[str, float] = {
    "calls": 0.0,
    "gather_unique_ms": 0.0,
    "decompress_ms": 0.0,
    "rotate_q_ms": 0.0,
    "remap_ms": 0.0,
    "stack_ms": 0.0,
    "flash_attn_ms": 0.0,
    "inverse_rotate_ms": 0.0,
    "num_unique_blocks": 0.0,
}


def _time_cuda_section(fn):
    if not _PROFILE_PREFILL:
        return fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1e3


def _record_prefill_timing(**kwargs: float) -> None:
    if not _PROFILE_PREFILL:
        return
    for key, value in kwargs.items():
        _PREFILL_TIMINGS[key] = _PREFILL_TIMINGS.get(key, 0.0) + value


def _dump_prefill_timings() -> None:
    if not _PROFILE_PREFILL or _PREFILL_TIMINGS["calls"] == 0:
        return
    calls = _PREFILL_TIMINGS["calls"]
    summary = {
        "calls": int(calls),
        "avg_unique_blocks": _PREFILL_TIMINGS["num_unique_blocks"] / calls,
    }
    for key in (
        "gather_unique_ms",
        "decompress_ms",
        "rotate_q_ms",
        "remap_ms",
        "stack_ms",
        "flash_attn_ms",
        "inverse_rotate_ms",
    ):
        summary[f"avg_{key}"] = _PREFILL_TIMINGS[key] / calls

    out_path = os.environ.get(
        "TURBOQUANT_PROFILE_PREFILL_PATH",
        f"/tmp/turboquant_prefill_timing_{os.getpid()}.json",
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"TurboQuant prefill timing summary written to {out_path}")


def _maybe_flush_prefill_timings() -> None:
    if not _PROFILE_PREFILL:
        return
    calls = int(_PREFILL_TIMINGS.get("calls", 0.0))
    if calls <= 0 or calls % _PROFILE_PREFILL_FLUSH_EVERY != 0:
        return
    _dump_prefill_timings()


atexit.register(_dump_prefill_timings)


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
        # TurboQuant stage 1 + 2:
        #   - 2-bit rotated key quantizer
        #   - 1-bit residual sketch with one bit per channel
        # The in-repo path no longer reserves dead outlier-index bytes.
        sketch_dim=head_size,
        outlier_count=0,
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
            # vLLM still allocates two cache planes for decoder attention.
            # The QJL path only uses kv_cache[0], but we keep the leading
            # dimension at 2 so the raw allocation matches the expected view.
            return (2, num_blocks, num_kv_heads, block_size, head_size)
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
        self._qjl_sketch_dim = head_size
        self._qjl_outlier_count = 0
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
                self._qjl_sketch_dim,
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
        _, proj_dir_quant = self._get_qjl_projection(key.device)
        rotation = self._get_rotation(key.dtype, key.device)
        assume_valid = (
            key.is_cuda
            and torch.cuda.is_current_stream_capturing()
        )
        pack_tokens_to_packed_qjl_cache(
            key,
            value,
            packed_cache,
            slot_mapping,
            proj_dir_quant=proj_dir_quant,
            rotation=rotation,
            value_group_size=self._qjl_value_group_size,
            value_bits=self._qjl_value_bits,
            outlier_count=self._qjl_outlier_count,
            assume_valid=assume_valid,
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
            if attn_metadata.max_query_len == 1:
                return self._forward_qjl_decode(query, kv_cache, attn_metadata, output)
            if self._qjl_metadata_has_prefix(attn_metadata):
                return self._forward_qjl_prefix_prefill(
                    query=query[: attn_metadata.num_actual_tokens],
                    key=key[: attn_metadata.num_actual_tokens],
                    value=value[: attn_metadata.num_actual_tokens],
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    output=output[: attn_metadata.num_actual_tokens],
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

        block_size     = comp_key_cache.shape[2]  # dim 2 in new layout

        # ---- 1. Gather the unique block indices needed this step ----
        num_reqs = attn_metadata.query_start_loc.shape[0] - 1
        block_table = attn_metadata.block_table[:num_reqs]  # [batch, max_blocks]
        seq_lens = attn_metadata.seq_lens[:num_reqs]
        if _PROFILE_PREFILL:
            def _gather_unique() -> tuple[torch.Tensor, torch.Tensor]:
                blocks_needed = torch.div(
                    seq_lens + (block_size - 1),
                    block_size,
                    rounding_mode="floor",
                )
                block_positions = torch.arange(
                    block_table.shape[1], device=block_table.device
                )
                used_mask = block_positions.unsqueeze(0) < blocks_needed.unsqueeze(1)
                valid = block_table[used_mask]
                valid = valid[valid >= 0]
                return used_mask, torch.unique(valid)

            (used_mask, unique_blk_ids), gather_ms = _time_cuda_section(_gather_unique)
        else:
            blocks_needed = torch.div(
                seq_lens + (block_size - 1),
                block_size,
                rounding_mode="floor",
            )
            block_positions = torch.arange(
                block_table.shape[1], device=block_table.device
            )
            used_mask = block_positions.unsqueeze(0) < blocks_needed.unsqueeze(1)
            valid = block_table[used_mask]
            valid = valid[valid >= 0]
            unique_blk_ids = torch.unique(valid)
            gather_ms = 0.0
        num_unique = unique_blk_ids.shape[0]

        # ---- 2. Decompress those blocks into a temporary BF16 buffer ----
        decomp_key = torch.empty(
            (num_unique, block_size, self.num_kv_heads, self._real_head_size),
            dtype=query.dtype, device=query.device,
        )
        decomp_value = torch.empty_like(decomp_key)

        if _PROFILE_PREFILL:
            def _do_decompress() -> None:
                turboquant_decompress_blocks(comp_key_cache, unique_blk_ids,
                                             decomp_key)
                turboquant_decompress_blocks(comp_value_cache, unique_blk_ids,
                                             decomp_value)
            _, decompress_ms = _time_cuda_section(_do_decompress)
        else:
            turboquant_decompress_blocks(comp_key_cache, unique_blk_ids, decomp_key)
            turboquant_decompress_blocks(comp_value_cache, unique_blk_ids, decomp_value)
            decompress_ms = 0.0

        # Keep decompressed K/V in rotated space and rotate the current query
        # chunk once. For long-context prefill this is much cheaper than
        # derotating every referenced KV block back to the original basis.
        num_actual = attn_metadata.num_actual_tokens
        R = self._get_rotation(query.dtype, query.device)
        if _PROFILE_PREFILL:
            query_rot, rotate_q_ms = _time_cuda_section(
                lambda: apply_rotation(query[:num_actual], R)
            )
        else:
            query_rot = apply_rotation(query[:num_actual], R)
            rotate_q_ms = 0.0

        # ---- 3. Remap block_table to indices in [0, num_unique) ----
        if _PROFILE_PREFILL:
            def _remap_blocks() -> torch.Tensor:
                remapped_bt = torch.full_like(block_table, -1)
                if num_unique > 0:
                    max_blk = int(unique_blk_ids.max().item())
                    remap = torch.full(
                        (max_blk + 2,),
                        -1,
                        dtype=block_table.dtype,
                        device=block_table.device,
                    )
                    remap[unique_blk_ids] = torch.arange(
                        num_unique, dtype=block_table.dtype, device=block_table.device
                    )
                    remapped_src = remap[block_table.clamp(min=0)]
                    remapped_bt[used_mask] = remapped_src[used_mask]
                return remapped_bt

            remapped_bt, remap_ms = _time_cuda_section(_remap_blocks)
        else:
            remapped_bt = torch.full_like(block_table, -1)
            if num_unique > 0:
                max_blk = int(unique_blk_ids.max().item())
                remap = torch.full(
                    (max_blk + 2,),
                    -1,
                    dtype=block_table.dtype,
                    device=block_table.device,
                )
                remap[unique_blk_ids] = torch.arange(
                    num_unique, dtype=block_table.dtype, device=block_table.device
                )
                remapped_src = remap[block_table.clamp(min=0)]
                remapped_bt[used_mask] = remapped_src[used_mask]
            remap_ms = 0.0

        # ---- 4. Run standard FlashAttention on the decompressed buffers ----
        new_metadata = replace(attn_metadata, block_table=remapped_bt)

        # Temporarily swap the backend's kv_cache_dtype so super().forward
        # does not try FP8 descale logic.
        orig_dtype = self.kv_cache_dtype
        self.kv_cache_dtype = "auto"

        # Fake a kv_cache from the decompressed tensors so that super().forward
        # can unbind(0) it correctly.
        if _PROFILE_PREFILL:
            fake_kv_cache, stack_ms = _time_cuda_section(
                lambda: torch.stack([decomp_key, decomp_value], dim=0)
            )
            result, flash_attn_ms = _time_cuda_section(
                lambda: super(TurboQuantAttentionImpl, self).forward(
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
            )
        else:
            fake_kv_cache = torch.stack([decomp_key, decomp_value], dim=0)
            stack_ms = 0.0
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
            flash_attn_ms = 0.0

        self.kv_cache_dtype = orig_dtype
        if not self._value_output_folded:
            if _PROFILE_PREFILL:
                rotated_out, inverse_rotate_ms = _time_cuda_section(
                    lambda: apply_rotation(result[:num_actual], R.T)
                )
                result[:num_actual].copy_(rotated_out)
            else:
                result[:num_actual].copy_(apply_rotation(result[:num_actual], R.T))
                inverse_rotate_ms = 0.0
        else:
            inverse_rotate_ms = 0.0
        _record_prefill_timing(
            calls=1.0,
            gather_unique_ms=gather_ms,
            decompress_ms=decompress_ms,
            rotate_q_ms=rotate_q_ms,
            remap_ms=remap_ms,
            stack_ms=stack_ms,
            flash_attn_ms=flash_attn_ms,
            inverse_rotate_ms=inverse_rotate_ms,
            num_unique_blocks=float(num_unique),
        )
        _maybe_flush_prefill_timings()
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
        rotation = self._get_rotation(query.dtype, query.device)

        num_actual = attn_metadata.num_actual_tokens
        query = query[:num_actual]
        seq_lens = attn_metadata.seq_lens[:num_actual]
        block_table = attn_metadata.block_table[:num_actual]

        result = decode_packed_qjl(
            query,
            packed_cache=packed_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            proj_dir_score=proj_dir_score,
            scale=self.scale,
            rotation=rotation,
            outlier_count=self._qjl_outlier_count,
            value_group_size=self._qjl_value_group_size,
            value_bits=self._qjl_value_bits,
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

    def _forward_qjl_prefix_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        packed_cache = kv_cache[0]
        proj_dir_score, _ = self._get_qjl_projection(query.device)
        rotation = self._get_rotation(query.dtype, query.device)
        q_per_kv = self.num_heads // self.num_kv_heads
        block_size = packed_cache.shape[2]

        query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        seq_lens = attn_metadata.seq_lens[: query_lens.shape[0]]
        prefix_lens = seq_lens - query_lens
        block_table = attn_metadata.block_table[: query_lens.shape[0]]

        for req_idx, q_len_tensor in enumerate(query_lens.tolist()):
            q_len = int(q_len_tensor)
            start = int(attn_metadata.query_start_loc[req_idx].item())
            stop = start + q_len

            q_chunk = query[start:stop]
            k_chunk = key[start:stop]
            v_chunk = value[start:stop]
            prefix_len = int(prefix_lens[req_idx].item())
            num_prefix_blocks = (prefix_len + block_size - 1) // block_size

            if prefix_len > 0 and num_prefix_blocks > 0:
                comp_block_table = block_table[req_idx, :num_prefix_blocks].to(
                    device=query.device,
                    dtype=block_table.dtype,
                )
                packed_scores, value_pack, value_scale, value_min = packed_qjl_prefix_scores_multi_query(
                    q_chunk,
                    packed_cache=packed_cache,
                    block_table=comp_block_table,
                    seq_len=prefix_len,
                    proj_dir_score=proj_dir_score,
                    scale=self.scale,
                    rotation=rotation,
                    outlier_count=self._qjl_outlier_count,
                    value_group_size=self._qjl_value_group_size,
                    value_bits=self._qjl_value_bits,
                )
                prefix_scores = packed_scores.to(torch.float32)
                prefix_value_pack = value_pack.unsqueeze(0)
                prefix_value_scale = value_scale.unsqueeze(0)
                prefix_value_min = value_min.unsqueeze(0)
            else:
                prefix_scores = query.new_empty((q_len, self.num_heads, 0), dtype=torch.float32)
                prefix_value_pack = query.new_empty(
                    (1, self.num_kv_heads, 0, self._real_head_size * self._qjl_value_bits // 32),
                    dtype=torch.int32,
                )
                prefix_value_scale = query.new_empty(
                    (1, self.num_kv_heads, 0, self._real_head_size // self._qjl_value_group_size)
                )
                prefix_value_min = torch.empty_like(prefix_value_scale)

            local_key = k_chunk.repeat_interleave(q_per_kv, dim=1)
            local_value = v_chunk.repeat_interleave(q_per_kv, dim=1)
            local_scores = torch.einsum(
                "qhd,khd->qhk",
                q_chunk.to(torch.float32),
                local_key.to(torch.float32),
            ) * self.scale
            causal_mask = torch.triu(
                torch.ones(q_len, q_len, device=query.device, dtype=torch.bool),
                diagonal=1,
            )
            local_scores = local_scores.masked_fill(causal_mask.unsqueeze(1), float("-inf"))

            all_scores = torch.cat([prefix_scores, local_scores], dim=-1)
            all_weights = torch.softmax(all_scores, dim=-1, dtype=torch.float32).to(query.dtype)

            req_out = query.new_zeros((q_len, self.num_heads, self._real_head_size))
            prefix_tokens = prefix_scores.shape[-1]

            if prefix_tokens > 0:
                req_out += cuda_quantized_bmm_gqa_dynamic(
                    self._qjl_value_group_size,
                    all_weights[:, :, :prefix_tokens].permute(1, 0, 2).unsqueeze(0),
                    prefix_value_pack,
                    prefix_value_scale,
                    prefix_value_min,
                    self._qjl_value_bits,
                ).squeeze(0).permute(1, 0, 2)

            req_out += torch.einsum(
                "qhk,khd->qhd",
                all_weights[:, :, prefix_tokens:],
                local_value,
            )
            output[start:stop].copy_(req_out)

        return output
