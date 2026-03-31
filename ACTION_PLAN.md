# AIMO-3 Throughput Optimization - Action Plan

## Executive Summary

**Current State** (2026-03-31):
- TQ4bit at 65k context: **27.8 tok/s** (up from 20.4, **+36% improvement**)
- TQ4bit now at **74% of FP8** aggregate throughput (was 55%)
- Need: 12-18x concurrency with equivalent throughput

**Root Causes Identified and Fixed**:
1. ✅ Inverted num_stages heuristic - disabled pipelining at high concurrency
2. ✅ num_warps too low - 4 warps insufficient for GQA_RATIO=8
3. ✅ kv_splits target too aggressive - 4224 programs at 64 requests caused resource contention
4. ✅ **NEW: USE_BT_PREFETCH threshold** - was 64, caused dependent loads at 65k context

**Remaining Gap**: 26% behind FP8 due to fundamental decompression arithmetic overhead.

## Changes Applied

### 1. Inverted num_stages Heuristic (FIXED)
**Before**:
```python
num_stages = 1 if blocks_per_split >= 128 else 2 if blocks_per_split >= 64 else 3
```

**After**:
```python
num_stages = 4 if blocks_per_split >= 256 else 3 if blocks_per_split >= 128 else 2
```

### 2. Increased num_warps (FIXED)
**Before**: 4 warps
**After**: 8 warps

### 3. Reduced kv_splits Target (FIXED)
**Before**: `target = _SM_COUNT * 32` → 4224 programs
**After**: `target = _SM_COUNT * 16` → 2112 programs

### 4. USE_BT_PREFETCH Threshold (FIXED)
**Before**: `use_bt_prefetch = (bps_pow2 <= 64)` ← **disabled at 65k context!**
**After**: `use_bt_prefetch = (bps_pow2 <= 256)`

At 65k context with 2 sequences, `blocks_per_split=128` exceeded the old threshold,
causing dependent loads that prevented memory pipelining.

## Test Results

**TQ4bit at 65k context**:
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Aggregate tok/s | 20.4 | 27.8 | **+36%** |
| TTFT | 5.4s | 3.4s | **-37%** |

**TQ4bit vs FP8** (65k context, 2 requests):
| Metric | TQ4bit | FP8 | Ratio |
|--------|--------|-----|-------|
| Aggregate tok/s | 27.8 | 37.4 | 0.74 |
| Decode tok/s | 56.5 | 59.4 | 0.95 |
| TTFT | 3.4s | 2.3s | 1.48 |

## Next Steps

### Short-term (kernel tuning):
1. Profile remaining bottleneck - likely decompression arithmetic
2. Consider INT8 WMMA path to halve dequant arithmetic

### Medium-term:
1. FlashInfer integration for TQ4bit - only path to truly closing the gap
2. Test on 120B model - different capacity/speed tradeoff

### Long-term:
1. Multi-GPU with Expert Parallelism for MoE models
2. Disaggregated Prefill/Decode (FP8 for prefill, TQ4bit for decode)

## Success Criteria

**Minimum viable**: 12x concurrency @ equivalent aggregate throughput to current 6x FP8

**Target**: 18x concurrency @ 80% of current per-request speed
- This gives 2.4x more total output tokens per second
- Sufficient to win AIMO-3 competition

## Key Insight

The "99.8% overhead" was misleading - it compared actual time to pure memory transfer time, ignoring attention compute cost. The real issue was **zero pipelining** causing memory stalls while compute sat idle.

After fixes, TQ4bit is now at 74% of FP8 aggregate throughput. The remaining 26% gap is fundamental decompression arithmetic overhead that requires either FlashInfer integration or INT8 WMMA to close.
