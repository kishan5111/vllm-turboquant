# TurboQuant Fused Backend — GPT-OSS 20B Integration

**Repo**: `kishan5111/vllm-turboquant`
**Branch**: `turboquant-fused` (flatcache-based)
**Working model**: `/workspace/models/gpt-oss-20b`
**Target**: H100 80GB, AIMO3 long-context MoE serving
**Updated**: 2026-04-01

## Goal

Integrate TurboQuant MSE+QJL KV-cache compression (FlatCache approach from 0xSero)
with vLLM v1 attention backends for end-to-end decode throughput improvement.
The key advantage: fused Triton kernels compute attention scores directly from
packed quantized data without materializing full-precision KV.

## Architecture

```
Attention Forward (Hybrid Mode)
├── Capture path: KV → ring buffer → compressed store (TQ MSE+QJL)
└── Decode path:
    ├── Ring buffer (exact recent tokens)
    ├── Compressed store (TQ MSE+QJL packed)
    └── Fused score compute from packed data (via Triton kernels)
```

**vs Existing TurboQuant Backend**: Uses FlatCache ring buffer + compressed store
instead of vLLM paged cache + decompression. Directly computes scores from
packed `prod_q` data.

## Benchmark Results

**Model**: GPT-OSS 20B (64 Q-heads, 8 KV-heads, GQA ratio=8)
**Setup**: 8 prompts, 128 max_tokens, chunked prefill

| Config | Mode | Throughput | Notes |
|--------|------|----------:|-------|
| BF16 KV | eager + FlashAttn | ~207 tok/s | baseline |
| FP8 KV | eager + FlashAttn | ~186 tok/s | 10% slower than bf16 |
| FP8 KV | cudagraph + FlashInfer | **~1258 tok/s** | 6× from graph capture |
| TurboQuant (3K/2V) | cudagraph + PyT fallback | **~1205 tok/s** | ~4% behind FP8 |
| TurboQuant (3K/2V) | eager + PyT fallback | ~189 tok/s | same as eager FP8 |

**Key findings**:
- TQ + cudagraph = **~1205 tok/s** vs FP8 + cudagraph = **~1258 tok/s**
- Gap is **~4%** — TQ uses PyTorch `_matmul_attention` fallback; FP8 uses FlashInfer
- TQ fused kernel (Triton) is wired but only used in "history only" decode (ring buffer empty) — rare during cudagraph warmup
- During cudagraph replay, the "history+recent" path always taken → PyTorch fallback
- Output quality verified identical to baseline

## Bug Fixes Applied (2026-04-01)

### 1. Wrong `num_query_heads` Source
**File**: `vllm/v1/attention/backends/turboquant_fused_backend.py` (line ~536)

GPT-OSS has 64 query heads and 8 KV heads. Code was reading `num_query_heads=8`
from `attn_module.num_heads` (wrapper) instead of `impl.num_heads`
(FlashAttentionImpl which has the correct 64).

```python
# Before: read from wrapper
num_query_heads = getattr(attn_module, "num_heads", None)

# After: read from impl
num_query_heads = getattr(impl, "num_heads", None)
```

### 2. Missing Transpose in "Recent Only" Branch
**File**: `vllm/v1/attention/backends/turboquant_fused_backend.py` (line ~252)

Ring buffer stores `(T, H_kv, D)` but `_matmul_attention` expects `(H_kv, T, D)`.
The transpose was applied in the "history+recent" path via `_get_all_keys_values`
but NOT in the "recent only" branch.

```python
# Before: recent_k passed without transpose
return _matmul_attention(query, recent_k, recent_v, ...)

# After: transpose to (H_kv, T, D)
recent_k = recent[0].transpose(0, 1)
recent_v = recent[1].transpose(0, 1)
return _matmul_attention(query, recent_k, recent_v, ...)
```

### 3. `torch.tensor()` Breaks CUDA Graph Capture
**File**: `vllm/v1/attention/ops/turboquant_fused/quantizer.py` (lines ~220, ~228)

`_pack_qjl_signs` and `_unpack_qjl_signs` created `torch.tensor([1,2,4,8,...])`
on CUDA during capture. This caused `cudaErrorStreamCaptureUnsupported`.

```python
# Before: per-call tensor allocation (breaks cudagraph)
powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=x.device, dtype=torch.uint8)

# After: pre-allocated buffer registered in __init__
self.register_buffer("_powers", torch.tensor([1,2,4,8,16,32,64,128], device=self.device, dtype=torch.uint8))
```

## How to Run

```bash
# Integration test
HF_HOME=/workspace/.hf_home .venv/bin/python tests/v1/test_turboquant_fused.py

# Benchmark cudagraph modes
HF_HOME=/workspace/.hf_home .venv/bin/python benchmarks/benchmark_turboquant_fused.py

# Benchmark eager modes (baseline comparison)
HF_HOME=/workspace/.hf_home .venv/bin/python benchmarks/benchmark_turboquant_fused_eager.py
```

## Key Files

- `vllm/v1/attention/backends/turboquant_fused_backend.py` — backend with hooks
- `vllm/v1/attention/ops/turboquant_fused/` — TQ ops (quantizer, store, capture, rotation)
- `vllm/v1/attention/ops/turboquant_fused/triton_kernels.py` — wrapper for Triton kernels
- `vllm/v1/attention/ops/triton_turboquant_fused_decode.py` — fused Triton kernels

## What's Working

- ✅ Hook installation on all 24 attention layers (GQA-aware)
- ✅ KV capture into ring buffer + compressed store
- ✅ Hybrid decode path (TQ fused path for single-token decode)
- ✅ CUDA graph compatibility (cudagraph capture succeeds)
- ✅ Output quality matches baseline (verified with identical outputs)
- ✅ Eager mode throughput comparable to FP8 eager

## What's NOT Working Yet

- ❌ **Triton fused kernels in common path** — fused kernel is wired for "history only" case
  but during cudagraph warmup the ring buffer always has recent tokens → "history+recent"
  path is always taken → uses PyTorch `_matmul_attention` fallback
  - Kernel GQA support is implemented (repeat_interleave expansion)
  - Would need "history+recent" kernel extension to close the ~4% gap
- ❌ **FlashInfer integration** — FlashInfer is used for FP8 KV baseline; TQ doesn't use it
- ❌ **0 flash layers detected** — `LayerConfig.backend_kind` not set, so `install_hooks`
  doesn't count any "flash layers"

## Next Steps to Beat FP8 KV (~1263 tok/s → higher)

1. **Extend Triton kernels for GQA** — current kernels assume MHA; need GQA-aware version
2. **Wire Triton kernels as compute path** — replace `_matmul_attention` fallback with fused kernel
3. **Skip dequantization** — compute scores from packed `prod_q` directly, saving memory bandwidth
4. **Verify with FlashInfer** — FlashInfer used for FP8 KV; TQ needs its own FlashInfer-style kernel
