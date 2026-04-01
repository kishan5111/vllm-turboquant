# TurboQuant Fused Backend — Action Plan

## Executive Summary

**Current State** (2026-04-01):
- TurboQuant Fused (3-bit key, 2-bit value) integrated with vLLM v1
- **1205 tok/s** with cudagraph + PyTorch fallback attention
- **1258 tok/s** FP8 KV baseline with cudagraph + FlashInfer
- Gap: **~4%** behind FP8
- Fused kernel (Triton) is GQA-aware and wired but rarely exercised during cudagraph

**Root Causes Fixed**:
1. ✅ Wrong `num_query_heads` — was reading from wrapper instead of impl
2. ✅ Missing transpose in "recent only" decode branch
3. ✅ `torch.tensor()` allocation in `_pack_qjl_signs` breaking cudagraph

## Changes Applied

### 1. `num_query_heads` Fix
Reading from `attn_module.num_heads` returned 8 (wrapper), but impl has 64.
GPT-OSS: 64 Q-heads, 8 KV-heads, GQA ratio=8.

### 2. Transpose Fix in "Recent Only" Branch
Ring buffer returns `(T, H_kv, D)` but `_matmul_attention` expects `(H_kv, T, D)`.
Only the "history+recent" path had the transpose via `_get_all_keys_values`.

### 3. CUDA Graph Compatibility Fix
`torch.tensor([1,2,4,8,16,32,64,128], device=x.device)` in hot path
breaks cudagraph capture. Pre-allocated as `register_buffer` instead.

## Benchmark Results

**GPT-OSS 20B, 8 prompts, 128 max_tokens**:

| Config | Mode | Throughput | Gap to FP8 cudagraph |
|--------|------|----------:|----------------------|
| BF16 KV | eager | ~207 tok/s | +baseline |
| FP8 KV | eager | ~186 tok/s | -10% |
| FP8 KV | cudagraph | **~1263 tok/s** | — |
| TQ 3K/2V | eager | ~189 tok/s | similar to FP8 eager |
| TQ 3K/2V | cudagraph | **~1206 tok/s** | **-4%** |

**Key insight**: Gap with cudagraph is only 4%, not the 10-20% we feared.
The PyTorch fallback attention is comparable to FlashAttention on this workload.

## Next Steps (Priority Order)

### P0 — Extend fused kernel for "history+recent" path (closes the 4% gap)

**What was done**: GQA support added to Triton kernel (`gqa_ratio` param + `repeat_interleave` expansion).
Kernel wired in `_compute_fused_attention` for "history only" case.

**Why not fully working**: During cudagraph warmup, ring buffer always gets tokens after first
decode step, so the "history+recent" path is always taken in benchmark. The fused kernel
only runs in the rare "history only" case.

**Fix needed**:
1. Extend fused kernel to handle combined history+recent tokens (concatenate compressed
   history with ring buffer float data, or add a second kernel stage)
2. Or: Optimize `_matmul_attention` PyTorch fallback to be FlashInfer-competitive
3. Benchmark to verify we reach ~1258 tok/s

### P1 — 0 Flash Layers Detection

`install_hooks` reports "0 flash layers" because `LayerConfig.backend_kind`
is never set. This means the flash attention path is not being used as a fallback.
Currently all attention goes through TQ patched path.

**Fix**: Set `backend_kind` in `LayerConfig` or detect it another way.

### P2 — FlashInfer Integration (long-term)

FlashInfer is vLLM's optimized decode attention backend for FP8.
TQ would benefit from FlashInfer-style kernels that compute directly from
packed quantized data — similar to how FlashInfer computes from FP8.

## Success Criteria

**Minimum**: TQ cudagraph throughput ≥ FP8 cudagraph throughput (1263 tok/s)
- Wire Triton kernels to close the 4% gap
- Verify output quality remains identical

**Stretch**: TQ cudagraph throughput > FP8 cudagraph throughput
- Fused compute from packed data avoids dequantization memory bandwidth
- Should be measurable once kernels are GQA-aware

## Key Files to Modify

| File | Change |
|------|--------|
| `vllm/v1/attention/ops/triton_turboquant_fused_decode.py` | Add GQA support to fused kernel |
| `vllm/v1/attention/backends/turboquant_fused_backend.py` | Wire Triton kernel as compute path |
| `vllm/v1/attention/backends/turboquant_fused_backend.py` | Set `backend_kind` for flash layer detection |
