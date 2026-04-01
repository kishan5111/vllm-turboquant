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

### Single-Request Throughput (GPT-OSS 20B)

**Model**: GPT-OSS 20B (64 Q-heads, 8 KV-heads, GQA ratio=8)
**Setup**: 8 prompts, 128 max_tokens, chunked prefill

| Config | Mode | Throughput | Notes |
|--------|------|----------:|-------|
| BF16 KV | eager + FlashAttn | ~207 tok/s | baseline |
| FP8 KV | eager + FlashAttn | ~186 tok/s | 10% slower than bf16 |
| FP8 KV | cudagraph + FlashInfer | **~1258 tok/s** | 6× from graph capture |
| TurboQuant (3K/2V) | cudagraph + Triton kernel | **~1211 tok/s** | ~4% behind FP8 |
| TurboQuant (3K/2V) | eager + PyT fallback | ~189 tok/s | same as eager FP8 |

### Concurrency Benchmark (GPT-OSS 20B, max_model_len=65536)

**Setup**: gpu_memory_utilization=0.90, 64-512 concurrent requests

| Concurrency | FP8 tok/s | TurboQuant tok/s | Winner |
|-------------|-----------|------------------|--------|
| 8 | 819.7 | 654.9 | FP8 |
| 16 | 1701.0 | 1348.1 | FP8 |
| 32 | 2466.5 | 2469.4 | ~Tie |
| 64 | 4581.2 | 4373.8 | FP8 |
| 128 | 7167.1 | 6478.5 | FP8 |
| 256 | 10040.1 | 7068.6 | **FP8 (+30%)** |
| 512 | 8608.4 | 8696.8 | **TurboQuant!** |

**Key findings**:
- At low concurrency (8-64), FP8 maintains 5-21% throughput advantage
- At very high concurrency (512), TurboQuant slightly edges out FP8
- At 256 concurrency (AIMO3 batch_size), FP8 is **30% faster**
- Root cause: FlashInfer's CUDA kernels are far more optimized than Triton's fused kernel
- Output quality verified identical to baseline

## Bug Fixes Applied (2026-04-01)

### 1. Wrong `num_query_heads` Source
**File**: `vllm/v1/attention/backends/turboquant_fused_backend.py` (line ~536)

GPT-OSS has 64 query heads and 8 KV heads. Code was reading `num_query_heads=8`
from `attn_module.num_heads` (wrapper) instead of `impl.num_heads`
(FlashAttentionImpl which has the correct 64).

### 2. Missing Transpose in "Recent Only" Branch
**File**: `vllm/v1/attention/backends/turboquant_fused_backend.py` (line ~252)

Ring buffer stores `(T, H_kv, D)` but `_matmul_attention` expects `(H_kv, T, D)`.
The transpose was applied in the "history+recent" path via `_get_all_keys_values`
but NOT in the "recent only" branch.

### 3. `torch.tensor()` Breaks CUDA Graph Capture
**File**: `vllm/v1/attention/ops/turboquant_fused/quantizer.py` (lines ~220, ~228)

`_pack_qjl_signs` and `_unpack_qjl_signs` created `torch.tensor([1,2,4,8,...])`
on CUDA during capture. This caused `cudaErrorStreamCaptureUnsupported`.

### 4. Critical: `squeeze(1).unsqueeze(1)` Corrupting Tensor Shapes
**File**: `vllm/v1/attention/backends/turboquant_fused_backend.py`

The pattern `mse_packed.squeeze(1).unsqueeze(1).contiguous()` on shape `(T=1, H_kv=8, packed_d)`:
- `squeeze(1)` removes dim 1 (size 8!) giving `(T=1, packed_d)` — **H_kv dimension lost!**
- `unsqueeze(1)` adds it back as `(T=1, 1, packed_d)` — **garbage data**
- `repeat_interleave(G, dim=0)` on garbage produces garbage

**Fix:**
```python
# Before (BROKEN):
mse_packed = mse_packed.squeeze(1).unsqueeze(1).contiguous()

# After (CORRECT):
mse_packed = mse_packed.reshape(H_kv, T, -1).contiguous()  # for history+recent
# or
mse_packed = mse_packed.permute(1, 0, 2).contiguous()  # for history only
```

**Commit**: `d6587138d` — fix: correct tensor reshape in TurboQuant fused attention backend

## How to Run

```bash
# Integration test
HF_HOME=/workspace/.hf_home .venv/bin/python tests/v1/test_turboquant_fused.py

# Benchmark cudagraph modes (FP8 vs TQ comparison)
HF_HOME=/workspace/.hf_home .venv/bin/python benchmarks/benchmark_turboquant_fused_cudagraph.py

# Concurrency benchmark (256/512 concurrent requests)
HF_HOME=/workspace/.hf_home .venv/bin/python benchmarks/benchmark_max_concurrency.py

# Profile kernel bottlenecks
HF_HOME=/workspace/.hf_home .venv/bin/python benchmarks/profile_kernel.py
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
- ✅ Triton fused kernel with history+recent support
- ✅ Autotuning configured for BLOCK_N parameter
- ✅ SDPA optimization in PyTorch fallback path

## What's NOT Working Yet

- ❌ **FlashInfer-level CUDA kernel** — Triton's fused kernel is ~4% slower than FlashInfer
- ❌ **0 flash layers detected** — `LayerConfig.backend_kind` not set

## Next Steps

1. **Benchmark GPT-OSS 120B** — test FP8 vs TQ at high concurrency (256 workers)
   - 120B has 6x higher KV cache pressure than 20B
   - TQ's denser KV might outperform FP8 on the larger model
2. **CUDA kernel** — attempt FlashInfer-style kernel if 120B shows TQ advantage
3. **For AIMO3 with 120B**: Test both FP8 and TQ at batch_size=256, compare throughput
