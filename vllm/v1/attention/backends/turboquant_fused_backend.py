# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TurboQuant Fused attention backend using 0xSero's FlatCache approach.

This backend provides end-to-end TurboQuant MSE+QJL compression with fully-fused
Triton kernels for decode attention. Unlike the existing turboquant_attn backend
which uses paged cache + decompression, this backend:

1. Captures KV into CompressedKVStore via shadow layer (like 0xSero)
2. Stores recent tokens in ring buffer for exact attention
3. Uses fused Triton kernels for computing attention scores directly from
   packed TurboQuant data without materializing full dequantized vectors

Key advantage:
  - PolarQuant (existing): 6 arithmetic ops per nibble for dequantization
  - MSE+QJL (this): 1 multiply per dimension for QJL contribution
  - Score computed directly from packed data via fused kernels

Usage:
  Use with vLLM's TurboQuant backend selection or via monkey-patching.
"""

from __future__ import annotations

import logging
import math
import os
import types
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.platforms.interface import DeviceCapability
from vllm.config.cache import CacheDType

from vllm.v1.attention.ops.turboquant_fused import (
    TurboQuantProd,
    CompressedKVStore,
    KVCaptureEngine,
    FlatCache,
    ValueQuantized,
    quantize_values,
    dequantize_values,
    generate_rotation_matrix,
    generate_qjl_matrix,
    get_codebook_tensors,
)
from vllm.v1.attention.ops.turboquant_fused.triton_kernels import (
    turboquant_fused_decode,
    turboquant_mse_score,
    turboquant_qjl_score,
)


logger = logging.getLogger("turboquant_fused_backend")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODE_OFF = "off"
MODE_CAPTURE_ONLY = "capture_only"
MODE_HYBRID = "hybrid"
MODE_FULL_TQ = "full_tq"

_GLOBAL_MODE = MODE_CAPTURE_ONLY


def set_mode(mode: str):
    global _GLOBAL_MODE
    _GLOBAL_MODE = mode
    logger.info(f"[TurboQuant Fused] Mode set to: {mode}")


def get_mode() -> str:
    return _GLOBAL_MODE


# ---------------------------------------------------------------------------
# Layer configuration and state
# ---------------------------------------------------------------------------

@dataclass
class LayerConfig:
    """Per-layer TurboQuant configuration."""
    head_dim: int
    num_kv_heads: int
    num_query_heads: int
    key_bits: int = 3
    value_bits: int = 2
    value_group_size: int = 32
    ring_capacity: int = 128
    layer_idx: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))


@dataclass
class LayerState:
    """Per-layer runtime state with FlatCache-based storage."""
    config: LayerConfig
    store: CompressedKVStore
    engine: KVCaptureEngine
    Pi: torch.Tensor  # Rotation matrix
    S: torch.Tensor   # QJL matrix
    centroids: torch.Tensor
    qjl_scale: float
    _log_count: int = 0

    @property
    def num_tokens(self) -> int:
        return self.store.num_tokens + self.engine.ring.size


def _create_layer_state(cfg: LayerConfig) -> LayerState:
    """Create a new layer state with all required tensors."""
    device = cfg.device
    head_dim = cfg.head_dim

    # Create compressed KV store
    store = CompressedKVStore(
        head_dim=head_dim,
        num_kv_heads=cfg.num_kv_heads,
        key_bits=cfg.key_bits,
        value_bits=cfg.value_bits,
        value_group_size=cfg.value_group_size,
        device=device,
        layer_idx=cfg.layer_idx,
    )

    # Create capture engine
    engine = KVCaptureEngine(
        store=store,
        ring_capacity=cfg.ring_capacity,
        device=device,
    )

    # Get rotation and QJL matrices
    Pi = generate_rotation_matrix(
        head_dim, device, torch.float32, seed=42 + cfg.layer_idx * 7
    )
    S = generate_qjl_matrix(
        head_dim, device, torch.float32, seed=12345 + cfg.layer_idx * 7
    )

    # Get codebook centroids (for MSE stage at bits-1)
    mse_bits = cfg.key_bits - 1
    centroids, _ = get_codebook_tensors(head_dim, mse_bits, device, torch.float32)

    # QJL scale constant
    qjl_scale = math.sqrt(math.pi / 2.0) / head_dim

    return LayerState(
        config=cfg,
        store=store,
        engine=engine,
        Pi=Pi,
        S=S,
        centroids=centroids,
        qjl_scale=qjl_scale,
    )


# ---------------------------------------------------------------------------
# Fused attention computation
# ---------------------------------------------------------------------------

def _compute_fused_attention(
    query: torch.Tensor,  # (num_tokens, num_query_heads, head_dim)
    store: CompressedKVStore,
    engine: KVCaptureEngine,
    Pi: torch.Tensor,
    S: torch.Tensor,
    centroids: torch.Tensor,
    qjl_scale: float,
    scale: float,
    gqa_ratio: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Compute attention using fused Triton kernels.

    This fuses the score computation directly from packed TurboQuant data.
    """
    flat = store.get_flat_cache()
    recent = engine.ring.peek()
    recent_k = recent[0] if recent else None
    recent_v = recent[1] if recent else None

    has_history = flat is not None and flat.num_tokens >= 16
    has_recent = recent_k is not None and recent_k.shape[0] > 0

    if not has_history and not has_recent:
        return torch.zeros_like(query)

    # Pre-compute rotated and sketched queries (one-time per decode step)
    # query: (T, Q, D) -> q_rot: (T*Q, D)
    T, Q, D = query.shape
    query_flat = query.reshape(-1, D)
    q_rot = torch.matmul(query_flat.float(), Pi.T)      # (T*Q, D)
    q_sketch = torch.matmul(query_flat.float(), S.T)    # (T*Q, D)

    # Expand query for GQA: replicate across kv heads
    # For now, assume GQA ratio and expand
    q_rot_expanded = q_rot.repeat_interleave(gqa_ratio, dim=0)  # (T*Q*H_kv, D)
    q_sketch_expanded = q_sketch.repeat_interleave(gqa_ratio, dim=0)

    if has_history and has_recent:
        # Concatenate compressed history with exact recent
        # This requires the fused kernel to handle both segments
        # For simplicity, concatenate in PyTorch for now
        k_all, v_all = _get_all_keys_values(flat, engine, num_kv_heads, head_dim)
        return _matmul_attention(query, k_all, v_all, scale, gqa_ratio, num_kv_heads)

    elif has_history:
        # History only - use fused Triton kernel (skip dequantization)
        # The kernel expects KV tensors in (BH=H_kv*G, N=H_kv*T, ...) layout.
        # For decode (T=1): MSE/QJL -> (H_kv, 1, packed_d), values -> (H_kv, 1, D)
        mse_packed = flat.prod_q.mse_indices
        qjl_signs = flat.prod_q.qjl_signs
        norms = flat.prod_q.norms
        res_norms = flat.prod_q.residual_norms
        value_q = flat.value_q

        # Unpack values
        v_data = _unpack_value_data(value_q)
        v_scales = value_q.scales
        v_zeros = value_q.zeros

        # Reshape for fused kernel:
        # mse_packed: (T, H_kv, packed_d) -> squeeze T -> (H_kv, T, packed_d)
        # For T=1: squeeze removes dim 0 (size 1), giving (H_kv, packed_d)
        # We need to add back the T dim for consistency: reshape to (H_kv, 1, packed_d)
        # qjl_signs: (T, H_kv, D//8) -> (H_kv, 1, D//8)
        # norms/res_norms: (T, H_kv) -> squeeze -> (H_kv,) -> unsqueeze -> (H_kv, 1)
        # v_data: (T, H_kv, D) -> (H_kv, 1, D)
        # v_scales/v_zeros: (T, H_kv, n_groups) -> (H_kv, 1, n_groups)
        BH = num_kv_heads * gqa_ratio  # T=1 so T*H_kv = H_kv

        # MSE/QJL: squeeze T dim then add back as 1 for uniform (H_kv, 1, ...) shape
        mse_packed = mse_packed.squeeze(1)  # (H_kv, packed_d)
        mse_packed = mse_packed.unsqueeze(1)  # (H_kv, 1, packed_d)
        qjl_signs = qjl_signs.squeeze(1).unsqueeze(1)  # (H_kv, 1, D//8)
        norms = norms.squeeze(1).unsqueeze(1)  # (H_kv, 1)
        res_norms = res_norms.squeeze(1).unsqueeze(1)  # (H_kv, 1)

        # Values: (T, H_kv, D) -> (H_kv, 1, D)
        v_data = v_data.squeeze(1).unsqueeze(1)  # (H_kv, 1, D)
        v_scales = v_scales.squeeze(1).unsqueeze(1)  # (H_kv, 1, n_groups)
        v_zeros = v_zeros.squeeze(1).unsqueeze(1)  # (H_kv, 1, n_groups)

        # Flatten query from (T, Q, D) to (BH, D) for kernel
        query_flat = query.reshape(-1, D)  # (BH, D)

        # Call the fused kernel with GQA-aware expansion inside
        result = turboquant_fused_decode(
            query=query_flat,
            quantized_key=flat.prod_q._replace(
                mse_indices=mse_packed,
                qjl_signs=qjl_signs,
                norms=norms,
                residual_norms=res_norms,
            ),
            value_quantized=value_q._replace(
                data=v_data,
                scales=v_scales,
                zeros=v_zeros,
            ),
            Pi=Pi,
            S=S,
            centroids=centroids,
            mse_bits=flat.prod_q.mse_bits,
            qjl_scale=qjl_scale,
            sm_scale=scale,
            group_size=32,
            gqa_ratio=gqa_ratio,
        )

        # Reshape back to (T, Q, D)
        return result.view(T, -1, D)

    else:  # recent only
        # Transpose from (T, H_kv, D) to (H_kv, T, D)
        recent_k = recent[0].transpose(0, 1)  # (H_kv, T, D)
        recent_v = recent[1].transpose(0, 1)
        return _matmul_attention(query, recent_k, recent_v, scale, gqa_ratio, num_kv_heads)


def _get_all_keys_values(
    flat: FlatCache,
    engine: KVCaptureEngine,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get all keys and values from compressed store + ring buffer."""
    # Dequantize compressed keys
    k_hist = engine.store.quantizer.dequantize(flat.prod_q)  # (H_kv, N_hist, D)
    v_hist = dequantize_values(flat.value_q, 32)

    # Get recent keys from ring buffer
    recent = engine.ring.peek()
    if recent is not None:
        recent_k, recent_v = recent
        recent_k = recent_k.transpose(0, 1)  # (H_kv, N_recent, D)
        recent_v = recent_v.transpose(0, 1)
        k_all = torch.cat([k_hist, recent_k], dim=1)
        v_all = torch.cat([v_hist, recent_v], dim=1)
    else:
        k_all = k_hist
        v_all = v_hist

    return k_all, v_all


def _unpack_value_data(value_q: ValueQuantized) -> torch.Tensor:
    """Unpack bit-packed value data."""
    bits = value_q.bits if len(value_q) > 3 else 2
    packed = value_q.data
    D = packed.shape[-1] * (4 if bits == 2 else 2)

    if bits == 2:
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        unpacked = torch.stack([v0, v1, v2, v3], dim=-1).reshape(*packed.shape[:-1], -1)
    else:
        v0 = packed & 0x0F
        v1 = (packed >> 4) & 0x0F
        unpacked = torch.stack([v0, v1], dim=-1).reshape(*packed.shape[:-1], -1)
    return unpacked


def _matmul_attention(
    query: torch.Tensor,
    kv_keys: torch.Tensor,
    kv_values: torch.Tensor,
    scale: float,
    gqa_ratio: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Standard GQA matmul attention using einsum (from 0xSero).

    query: (T, Q, D) where Q = num_kv_heads * gqa_ratio
    kv_keys: (num_kv_heads, N, D)
    kv_values: (num_kv_heads, N, D)

    Returns: (T, Q, D)
    """
    T, Q, D = query.shape
    H_kv = num_kv_heads
    G = gqa_ratio
    if Q != H_kv * G:
        raise ValueError(f"Incompatible GQA shapes: Q={Q}, H_kv={H_kv}, gqa_ratio={G}")

    # q: (T, Q, D) -> (H_kv, G, T, D)
    q = query.float().view(T, H_kv, G, D).permute(1, 2, 0, 3)
    # k: (H_kv, N, D) -> (H_kv, 1, N, D) broadcast over G
    k = kv_keys.float().unsqueeze(1)
    v = kv_values.float().unsqueeze(1)

    # scores: (H_kv, G, T, N)
    scores = torch.einsum("hgtd,hgnd->hgtn", q, k) * scale
    weights = F.softmax(scores, dim=-1)
    # out: (H_kv, G, T, D)
    out = torch.einsum("hgtn,hgnd->hgtd", weights, v)

    # Back to (T, Q, D)
    return out.permute(2, 0, 1, 3).reshape(T, Q, D).to(query.dtype)


# ---------------------------------------------------------------------------
# Patched forward functions
# ---------------------------------------------------------------------------

def _make_patched_forward(orig_fn, state: LayerState, no_alloc: bool = False):
    """Intercept forward to capture KV and use TQ decode."""

    def patched(
        self_impl,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        mode = _GLOBAL_MODE

        # Capture K/V when not in off mode
        if mode != MODE_OFF and attn_metadata is not None:
            num_tokens = getattr(attn_metadata, 'num_actual_tokens', key.shape[0])
            if num_tokens <= 1:
                state.engine.ingest_decode(key[:num_tokens], value[:num_tokens], num_tokens)
            else:
                state.engine.ingest_prefill(key[:num_tokens], value[:num_tokens], num_tokens)

        # Off mode: passthrough
        if mode == MODE_OFF:
            return orig_fn(
                self_impl, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        # Capture-only mode: no TQ decode
        if mode == MODE_CAPTURE_ONLY:
            return orig_fn(
                self_impl, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        # Profiling pass or prefill: use flash
        if attn_metadata is None:
            return orig_fn(
                self_impl, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        is_prefill = attn_metadata.max_query_len > 1
        if is_prefill:
            return orig_fn(
                self_impl, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        # --- Hybrid decode ---
        if mode == MODE_HYBRID:
            num_actual = getattr(attn_metadata, 'num_actual_tokens', query.shape[0])
            flat = state.store.get_flat_cache()
            has_history = flat is not None and flat.num_tokens >= 16
            recent = state.engine.ring.peek()
            has_recent = recent is not None and recent[0].shape[0] > 0

            # Only use fused path for single-token decode (num_actual == 1).
            # For multi-token sequences (prefill or multi-sequence batch),
            # the ring buffer concatenates all sequences without masking,
            # which is incompatible with standard cross-attention.
            if num_actual == 1 and (has_history or has_recent):
                q = query[:num_actual]
                if q.dim() == 2:
                    q = q.view(num_actual, state.config.num_query_heads, state.config.head_dim)

                recent_k = recent[0] if recent else None
                recent_v = recent[1] if recent else None

                gqa_ratio = state.config.num_query_heads // state.config.num_kv_heads
                head_dim = state.config.head_dim
                num_kv_heads = state.config.num_kv_heads
                scale = 1.0 / math.sqrt(head_dim)

                result = _compute_fused_attention(
                    query=q,
                    store=state.store,
                    engine=state.engine,
                    Pi=state.Pi,
                    S=state.S,
                    centroids=state.centroids,
                    qjl_scale=state.qjl_scale,
                    scale=scale,
                    gqa_ratio=gqa_ratio,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )

                result_flat = result.reshape(
                    num_actual, state.config.num_query_heads * head_dim
                ).to(query.dtype)

                if output is not None:
                    out_slice = output[:num_actual]
                    if out_slice.dim() == 3:
                        out_slice.copy_(result.to(out_slice.dtype))
                    else:
                        out_slice.copy_(result_flat.to(out_slice.dtype))
                    return output
                if query.dim() == 3:
                    return result.to(query.dtype)
                return result_flat

        return orig_fn(
            self_impl, layer, query, key, value, kv_cache,
            attn_metadata, output, output_scale, output_block_scale,
        )

    return patched


def _make_patched_kv_update(orig_fn, state: LayerState, no_alloc: bool = False):
    """Intercept KV cache updates to capture into TQ store."""

    def patched(
        self_impl, layer, key, value, kv_cache, slot_mapping
    ):
        if not no_alloc:
            orig_fn(self_impl, layer, key, value, kv_cache, slot_mapping)

        mode = _GLOBAL_MODE
        if mode == MODE_OFF:
            return

        num_tokens = slot_mapping.shape[0]
        if num_tokens <= 1:
            state.engine.ingest_decode(key, value, num_tokens)
        else:
            state.engine.ingest_prefill(key, value, num_tokens)

    return patched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def install_hooks(
    model_runner,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    ring_capacity: int = 128,
    initial_layers_count: int = 4,
    initial_layers_key_bits: int | None = None,
    mode: str = MODE_CAPTURE_ONLY,
    no_alloc: bool = False,
) -> dict[str, LayerState]:
    """Install TurboQuant Fused hooks on all attention layers.

    Returns: dict mapping layer_name -> LayerState
    """
    global _GLOBAL_MODE
    _GLOBAL_MODE = mode

    if initial_layers_key_bits is None:
        initial_layers_key_bits = min(key_bits + 1, 4)

    static_ctx = model_runner.compilation_config.static_forward_context
    device = model_runner.device

    layer_states: dict[str, LayerState] = {}
    layer_idx = 0

    for layer_name, attn_module in static_ctx.items():
        if not hasattr(attn_module, "impl"):
            continue

        impl = attn_module.impl

        # Check if this is a flash attention implementation
        has_forward = hasattr(impl, "forward")
        has_kv_update = hasattr(impl, "do_kv_cache_update")

        if not has_forward:
            continue

        num_kv_heads = getattr(impl, "num_kv_heads", None)
        if num_kv_heads is None:
            continue

        # Determine head_dim
        head_dim = getattr(impl, "head_size", None)
        if head_dim is None:
            head_dim = getattr(impl, "kv_lora_rank", None)
        if head_dim is None:
            continue
        head_dim = int(head_dim)

        num_query_heads = getattr(impl, "num_heads", None)
        if num_query_heads is None:
            num_query_heads = getattr(attn_module, "num_heads", num_kv_heads)
        num_query_heads = int(num_query_heads)

        bits = initial_layers_key_bits if layer_idx < initial_layers_count else key_bits

        cfg = LayerConfig(
            head_dim=head_dim,
            num_kv_heads=int(num_kv_heads),
            num_query_heads=num_query_heads,
            key_bits=bits,
            value_bits=value_bits,
            value_group_size=min(value_group_size, head_dim),
            ring_capacity=ring_capacity,
            layer_idx=layer_idx,
            device=device,
        )

        state = _create_layer_state(cfg)
        layer_states[layer_name] = state

        # Patch forward
        patched_forward = _make_patched_forward(
            impl.forward.__func__, state, no_alloc=no_alloc
        )
        impl.forward = types.MethodType(
            lambda self, *a, _p=patched_forward, **kw: _p(self, *a, **kw), impl
        )

        # Patch KV update if available
        if has_kv_update:
            patched_update = _make_patched_kv_update(
                impl.do_kv_cache_update.__func__, state, no_alloc=no_alloc
            )
            impl.do_kv_cache_update = types.MethodType(
                lambda self, *a, _p=patched_update, **kw: _p(self, *a, **kw), impl
            )

        impl._tq_fused_state = state
        layer_idx += 1

    model_runner._tq_fused_layer_states = layer_states
    model_runner._tq_fused_no_alloc = no_alloc

    logger.info(
        f"[TurboQuant Fused] Hooks installed on {len(layer_states)} layers "
        f"(mode={mode}, no_alloc={no_alloc})"
    )
    return layer_states


def get_stats(model_runner) -> dict:
    """Return summary statistics for all TQ layer states."""
    layer_states = getattr(model_runner, "_tq_fused_layer_states", None)
    if not layer_states:
        return {}

    stats = {}
    total_compressed = 0
    total_buffered = 0
    total_memory = 0

    for name, state in layer_states.items():
        compressed = state.store.num_tokens
        buffered = state.engine.ring.size
        mem = state.store.memory_bytes()
        total_compressed += compressed
        total_buffered += buffered
        total_memory += mem

    stats["num_layers"] = len(layer_states)
    stats["total_compressed_tokens"] = total_compressed
    stats["total_buffered_tokens"] = total_buffered
    stats["total_memory_bytes"] = total_memory
    stats["mode"] = _GLOBAL_MODE
    return stats


def reset(model_runner):
    """Reset all layer states."""
    layer_states = getattr(model_runner, "_tq_fused_layer_states", None)
    if layer_states:
        for state in layer_states.values():
            state.engine.reset()


# ---------------------------------------------------------------------------
# Integration helper — wires TurboQuant Fused into vLLM engine initialization
# ---------------------------------------------------------------------------

def enable_turboquant_fused(
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    ring_capacity: int = 128,
    initial_layers_count: int = 4,
    initial_layers_key_bits: int | None = None,
    mode: str = MODE_CAPTURE_ONLY,
    no_alloc: bool = False,
):
    """Patch Executor to automatically install TurboQuant Fused hooks.

    Call this BEFORE creating vllm.LLM().

    This patches UniProcExecutor (and other executors) so that TurboQuant Fused
    hooks are installed on all attention layers during engine initialization —
    after the model is loaded but before KV cache is allocated (when no_alloc=True).

    Args:
        key_bits: Bits for key quantization (default 3)
        value_bits: Bits for value quantization (default 2)
        value_group_size: Group size for value quantization (default 32)
        ring_capacity: Ring buffer capacity for exact recent tokens (default 128)
        initial_layers_count: Number of initial layers with higher key_bits
        initial_layers_key_bits: Override key_bits for initial layers
        mode: TurboQuant mode — off | capture_only | hybrid | full_tq
        no_alloc: If True, skip allocating regular KV cache for TQ layers
                  (saves memory but requires backend support)
    """
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.worker.gpu_worker import Worker

    if hasattr(Executor, "_tq_fused_patched"):
        return

    orig_init_from_config = Executor.initialize_from_config

    def patched_init_from_config(self, kv_cache_configs):
        # Call original first — this allocates KV cache
        orig_init_from_config(self, kv_cache_configs)

        # Install TQ hooks on all workers after KV cache is set up
        def _install_on_worker(worker):
            if worker.model_runner is None:
                return {}
            from vllm.v1.attention.backends.turboquant_fused_backend import install_hooks
            states = install_hooks(
                worker.model_runner,
                key_bits=key_bits,
                value_bits=value_bits,
                value_group_size=value_group_size,
                ring_capacity=ring_capacity,
                initial_layers_count=initial_layers_count,
                initial_layers_key_bits=initial_layers_key_bits,
                mode=mode,
                no_alloc=no_alloc,
            )
            return {
                "hooks": len(states),
                "flash_layers": sum(
                    1 for s in states.values()
                    if hasattr(s, "supports_hybrid") and s.supports_hybrid
                ),
            }

        try:
            results = self.collective_rpc(_install_on_worker)
            total_hooks = sum(r.get("hooks", 0) for r in results if r)
            total_flash = sum(r.get("flash_layers", 0) for r in results if r)
            print(
                f"[TurboQuant Fused] Installed {total_hooks} hooks "
                f"({total_flash} flash layers) — mode={mode}, no_alloc={no_alloc}",
                flush=True,
            )
        except Exception as e:
            print(
                f"[TurboQuant Fused] collective_rpc install failed: {e}", flush=True
            )

    Executor.initialize_from_config = patched_init_from_config
    Executor._tq_fused_patched = True

    # Also patch Worker.load_model to install hooks right after model loads
    # (needed for cases where initialize_from_config isn't called)
    orig_worker_load = Worker.load_model

    def patched_worker_load(self_worker, *, load_dummy_weights=False):
        orig_worker_load(self_worker, load_dummy_weights=load_dummy_weights)
        # Install hooks after model is loaded
        if self_worker.model_runner is not None:
            already_hooked = hasattr(
                self_worker.model_runner, "_tq_fused_layer_states"
            )
            if not already_hooked:
                try:
                    from vllm.v1.attention.backends.turboquant_fused_backend import install_hooks
                    install_hooks(
                        self_worker.model_runner,
                        key_bits=key_bits,
                        value_bits=value_bits,
                        value_group_size=value_group_size,
                        ring_capacity=ring_capacity,
                        initial_layers_count=initial_layers_count,
                        initial_layers_key_bits=initial_layers_key_bits,
                        mode=MODE_CAPTURE_ONLY,  # capture-only during load
                        no_alloc=False,
                    )
                    print(
                        f"[TurboQuant Fused] Hooks installed via load_model patch",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[TurboQuant Fused] load_model install failed: {e}",
                        flush=True,
                    )

    Worker.load_model = patched_worker_load

    print(
        f"[TurboQuant Fused] Patched Executor for auto TQ hook installation "
        f"(mode={mode}, no_alloc={no_alloc})",
        flush=True,
    )