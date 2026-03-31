# AIMO-3 Throughput Optimization - Action Plan

## Executive Summary

**Current State**:
- GPT-OSS 120B @ 65K context: 6x concurrency
- Need: 12-18x concurrency (2-3x improvement)
- TurboQuant 4-bit: 1.94x better memory, but 0.62x speed = **INSUFFICIENT (8.8x effective)**

**Root Cause Analysis**:
1. TurboQuant kernel ran with **num_stages=1** (zero pipelining) at all tested concurrency levels
2. At 64 requests: blocks_per_split=512 triggered the inverted heuristic → no pipelining → 99% memory stalls
3. Aggregate throughput is FLAT for both FP8 and TQ4bit (not scaling with concurrency)
4. Memory bandwidth utilization: 0.7% (NOT memory bound - compute/scheduling bottleneck)

## Changes Applied

Fixed 3 critical bugs in `vllm/v1/attention/ops/triton_turboquant_paged_attn.py`:

### 1. Inverted num_stages Heuristic (line 537-546)
**Before**:
```python
num_stages = 1 if blocks_per_split >= 128 else 2 if blocks_per_split >= 64 else 3
```
This disabled pipelining at high concurrency (when you need it most).

**After**:
```python
num_stages = 4 if blocks_per_split >= 256 else 3 if blocks_per_split >= 128 else 2
```
Enables max pipelining for large chunks to hide memory latency.

### 2. Increased num_warps (line 525-528)
**Before**: 4 warps (default)
**After**: 8 warps

With GQA_RATIO=8, each kernel processes 8 Q heads. More warps = better occupancy and latency hiding on H100.

### 3. Reduced kv_splits Target (line 405)
**Before**: `target = _SM_COUNT * 32` → 4224 programs at 64 requests
**After**: `target = _SM_COUNT * 16` → 2112 programs at 64 requests

Reduces register pressure and resource contention.

## Test Now

Run the benchmark to validate fixes:

```bash
cd /Users/kishanvavdara/Github/turboquant-vllm/turboquant-vllm-kishan/vllm-turboquant

# Test 1: Single sweep at 65K with fixes
python scripts/run_bench.py \
  --model openai/gpt-oss-20b \
  --kv-dtype turboquant_4bit \
  --max-model-len 69632 \
  --gpu-util 0.93 \
  --cudagraph-mode full_decode_only \
  --no-chunked-prefill \
  --workloads ctx65k

# Compare against previous results in results/65k_fp8_vs_tq4bit_concurrency_20260331.json
```

**What to check**:
- Does TQ4bit throughput hold steady or improve at higher concurrency?
- Previous: 28.6 → 24.2 tok/s (degraded). Should now maintain or increase.

## Profile (GPU Required)

Need profiler data to identify remaining bottleneck:

```bash
# NVIDIA Nsight Compute
ncu --set full -o profile_tq4bit.ncu-rep \
  python scripts/run_bench.py \
    --model openai/gpt-oss-20b \
    --kv-dtype turboquant_4bit \
    --max-model-len 69632 \
    --workloads ctx65k

# Look for:
# 1. SM occupancy (should be >50%)
# 2. Memory throughput (should be >50% of peak)
# 3. Warp stall reasons
# 4. Register spills
# 5. Achieved vs theoretical compute throughput
```

## Next Steps Based on Results

### If TQ4bit Now Scales Well (throughput stable/increasing with concurrency):
1. Test at higher concurrency (128, 256 requests) to find crossover point
2. Deploy for AIMO-3

### If TQ4bit Still Degrades:
Profiler will show the bottleneck. Common issues:
- **Register spills**: Reduce num_warps back to 4 or optimize kernel to use fewer registers
- **Low compute throughput**: Decompression arithmetic is the bottleneck (need CUDA rewrite or INT8 WMMA path)
- **Low memory throughput**: Memory access pattern issues (bank conflicts, uncoalesced loads)

### If Both FP8 and TQ4bit Don't Scale:
vLLM v1 scheduler issue - decode operations aren't being batched efficiently. This is outside the kernel.

## Alternative Paths

If kernel optimization doesn't hit target:

### Option A: Test on 120B Model
All current benchmarks use 20B. The 120B model has:
- 6x more parameters → less memory for KV cache
- Crossover point where TQ4bit wins might be at LOWER concurrency

### Option B: Multi-GPU with Expert Parallelism
For MoE models, shard experts across 2x H100:
- Each GPU handles different experts
- Can scale to 18-24x concurrency total

### Option C: Disaggregated Prefill/Decode
Split workload:
- Prefill GPU: FP8, optimized for throughput
- Decode GPU: TQ4bit, optimized for capacity

## Success Criteria

**Minimum viable**: 12x concurrency @ equivalent aggregate throughput to current 6x FP8

**Target**: 18x concurrency @ 80% of current per-request speed
- This gives 2.4x more total output tokens per second
- Sufficient to win AIMO-3 competition

## Key Insight

The "99.8% overhead" was misleading - it compared actual time to pure memory transfer time, ignoring attention compute cost. The real issue was **zero pipelining** causing memory stalls while compute sat idle.

Fixes applied should significantly improve throughput if decompression/attention math isn't the new bottleneck. Profiling will tell us definitively.
