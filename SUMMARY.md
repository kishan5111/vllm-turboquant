# TurboQuant GPT-OSS Decode Optimization

**Repo**: `kishan5111/vllm-turboquant`
**Branch**: `main`
**Working model**: `openai/gpt-oss-20b`
**Primary target**: H100 80GB, AIMO3 long-context MoE serving
**Updated**: 2026-03-31

## Goal

Optimize decode throughput for AIMO3-style workload (200 token input → 65k token output)
so that TurboQuant 4-bit's 2x memory capacity advantage translates to better
aggregate serving throughput than FP8.

## Current Benchmark Results (AIMO3-style, 8192 decode tokens)

| Config | tok/s | Notes |
|---|---:|---|
| FP8 | **921** | FlashInfer backend, piecewise cudagraph |
| TQ4bit + piecewise | 477 | decompress+FlashAttention fallback |
| TQ4bit + full_decode_only | **643** | fused Triton kernel, best TQ |

**Gap: TQ4bit is ~30% slower than FP8 on per-request decode tok/s.**

## Why the Gap Exists

The 30% decode gap is **fundamental arithmetic overhead**, not tunable:

FP8 dequantization (FlashInfer):
```
dequant_k = fp8_k * scale  // 1 op per element
```

TQ4bit dequantization (Triton kernel, lines 237-238):
```
k_lo = (k_packed & 0x0F).to(fp16) * ks_scale - ks  // 4 ops per element
k_hi = ((k_packed >> 4) & 0x0F).to(fp16) * ks_scale - ks
```

TQ does 4× more arithmetic per K/V element, inside nested loops over
BLOCK_SIZE positions × all context blocks. This overhead is paid on every
decode step regardless of CUDA graph mode or kv_splits.

## What Was Ruled Out

- **kv_splits tuning**: Changes only how work distributes across SMs,
  not the per-element cost. No split value closes the gap.
- **cudagraph_mode=piecewise vs full_decode_only**: full_decode_only is
  ~35% better but doesn't close the arithmetic gap.
- **Concurrency sweeps**: vLLM handles batching via continuous batching;
  manual concurrency sweeps don't improve per-request decode speed.

## Capacity vs Decode Speed Tradeoff

TQ4bit's advantage is **memory capacity**, not per-request speed:

| | FP8 | TQ4bit |
|---|---|---|
| Decode tok/s | 921 | 643 |
| KV cache capacity | 1× | ~2× |
| Concurrent requests fit | 1× | ~2× |
| Aggregate tok/s (full util) | 921 | ~1286 (2× × 643) |

At full GPU memory utilization with many concurrent AIMO3 requests,
TQ4bit's 2× capacity could theoretically yield ~40% more aggregate throughput
than FP8 — but this depends on continuous batching saturation.

## What Still Needs to Be Built

1. **FlashInfer integration for TQ4bit** — Only path to truly closing the
   decode gap. FlashInfer has H100-native FP8 decode kernels; TQ4bit would
   need its own FlashInfer-style CUDA kernel. ~1-2 weeks for someone familiar
   with FlashInfer internals.

2. **INT8 WMMA tensor core path** — Replace FP16 dequantization with H100
   INT8 WMMA: load nibbles as INT8, matmul INT8→INT32, convert to FP32 after
   accumulation. Cuts dequant arithmetic in half. Medium difficulty, needs CUDA.

3. **Fused prefill kernel for fresh prefills** — Current fused prefill kernel
   (`turboquant_context_attention_fwd`) only fires when `has_cached_prefix_ctx=True`,
   i.e., multi-turn only. For fresh single-turn prefills it falls back to
   decompress+FlashAttention. The fused path needs to trigger for fresh prefills too.

## Key Files

- `vllm/v1/attention/backends/turboquant_attn.py` — backend routing, fused decode path
- `vllm/v1/attention/ops/triton_turboquant_paged_attn.py` — fused decode Triton kernel
- `vllm/v1/attention/ops/triton_turboquant_prefill.py` — fused prefill Triton kernel (inactive for fresh prefills)
- `vllm/v1/attention/ops/triton_turboquant_kv.py` — K/V compression helpers

## Bottom Line

TQ4bit's decode gap to FP8 is arithmetic-bound, not tunable via kernel params.
The path to closing it requires either FlashInfer integration or INT8 WMMA.
Until then, TQ4bit's value proposition is 2× memory capacity, which may or
may not translate to aggregate throughput advantage depending on AIMO3 workload
concurrency patterns.

If AIMO3 competition allows high concurrency (many simultaneous long outputs),
the capacity win likely wins. If it's single-request latency-focused, FP8 wins.
