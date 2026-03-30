# TurboQuant GPT-OSS Summary

**Repo**: `kishan5111/vllm-turboquant`
**Branch**: `main`
**Working model**: `openai/gpt-oss-20b`
**Primary target**: H100 80GB, long-context MoE serving
**Current date**: 2026-03-30

## End Goal

We are not trying to build "QJL but smaller."

The real end goal is a production-usable TurboQuant KV-cache path for GPT-OSS-20B where:

1. Stage A stores keys with a high-quality rotated vector quantizer.
2. Stage B adds a 1-bit residual sketch so inner products stay much less biased than plain low-bit quantization.
3. Values stay aggressively compressed.
4. The backend is fast enough under vLLM CUDA graph capture to matter for real serving, not just microbenchmarks.

For GPT-OSS-20B specifically, the serving target is:

- materially better KV-cache compression than FP8
- enough long-context capacity gain to admit more concurrent requests
- decode and mixed prefill/decode throughput that is competitive enough for the capacity gain to translate into higher aggregate serving throughput

Practical performance target:

- published TurboQuant claim: 4-bit TurboQuant achieves up to `8x` attention-logit speedup over `32-bit` unquantized keys on H100, per the Google Research blog and TurboQuant results
- our GPT-OSS serving target: materially better KV-cache compression than FP8, then enough end-to-end long-context throughput that the larger admitted concurrency yields about `~1.5x-2.0x` aggregate serving throughput vs FP8

Important nuance:

- this does **not** require per-request decode tok/s to beat FP8 everywhere
- it **does** require TurboQuant attention overhead to be low enough that the KV-capacity win is not erased by slower attention

## Current Status

### What is implemented

The repo now has a working in-repo TurboQuant-stage1-plus-stage2 prototype path under `kv_cache_dtype='turboquant_qjl'`:

- stage A: rotated 2-bit key quantization
- stage B: packed 1-bit residual sketch correction
- values: 2-bit grouped quantization
- decode: CUDA-graph-safe packed-cache decode path
- mixed prefix-prefill: working, but still partly eager/PyTorch

This path no longer depends on the external `qjl_decode_proto` package.

### What is true today

The current implementation already wins on KV-cache capacity at long context, and decode throughput has improved substantially, but it is still not the final TurboQuant target.

What is still missing:

- a fully fused Triton/CUDA stage-B path for shared-prefix prefill
- a Triton/CUDA write path for the stage-B residual sketch pack itself
- a cleaner migration from the current `qjl_backend_proto` adapter into a true TurboQuant-specific ops stack
- end-to-end evidence that aggregate long-context serving throughput beats FP8 once concurrency is increased

## Latest GPT-OSS-20B Results

All numbers below are from the local smoke harness:

`/workspace/benchmark_fp8_vs_turboquant_qjl_smoke.py`

### TurboQuant stage1+2 prototype (`turboquant_qjl`)

| Shape | Throughput | KV cache size | Notes |
|---|---:|---:|---|
| `256 / batch 8 / max_tokens 16` | `203.4 tok/s` | `2,864,512 tokens` | `FULL_DECODE_ONLY` capture |
| `2048 / batch 2 / max_tokens 16` | `67.9 tok/s` | `2,857,184 tokens` | `FULL_DECODE_ONLY` capture |
| `8192 / batch 2 / max_tokens 16` | `42.6 tok/s` | `2,813,536 tokens` | `FULL_DECODE_ONLY` capture |

### FP8 reference

| Shape | Throughput | KV cache size |
|---|---:|---:|
| `8192 / batch 2 / max_tokens 16` | `193.8 tok/s` | `2,287,776 tokens` |

### Interpretation

- TurboQuant stage1+2 is now clearly ahead of FP8 on KV-cache capacity for long context.
- The decode scorer rewrite and fused packed stage1+2 score kernel produced real gains.
- TurboQuant is still substantially slower than FP8 on per-request decode throughput.
- The project is now in the "capacity win is real, attention cost still too high" phase.

## What Was Built

### 1. vLLM backend integration

#### `vllm/v1/attention/backends/turboquant_attn.py`

This is the central wiring point.

It now:

- routes `turboquant_4bit` and `turboquant_qjl`
- owns the lazy rotation matrix and QJL/TurboQuant projection state
- updates packed KV cache on the write path
- dispatches between:
  - fused 4-bit TurboQuant decode
  - packed stage1+2 decode
  - packed-prefix prefill path
  - regular prefill path
- keeps the QJL/TurboQuant path CUDA-graph-capture-safe

For `turboquant_qjl`, this file currently defines the effective runtime contract:

- `sketch_dim = head_size`
- `outlier_count = 0`
- value quantization stays 2-bit grouped

### 2. KV-cache allocator wiring

#### `vllm/model_executor/layers/attention/attention.py`

This file was patched so vLLM allocates KV cache with the compressed slot size, not the raw head size.

Without this, the backend shape would be logically correct but the allocator would hand it the wrong amount of memory.

### 3. 4-bit TurboQuant path

#### `vllm/v1/attention/ops/triton_turboquant_kv.py`

This file contains the original TurboQuant 4-bit machinery:

- 4-bit pack kernel
- 4-bit unpack kernel
- rotation helpers
- early stage-2 experimental section

Today, the 4-bit path here is real and used.

The stage-2 section here is **not** yet the live path for GPT-OSS. It still contains placeholder logic for the JL sketch and needs to be finished if we want the final implementation to live fully here instead of in the proto adapter.

#### `vllm/v1/attention/ops/triton_turboquant_paged_attn.py`

This is the fused paged decode kernel for the 4-bit TurboQuant backend.

It already contains:

- GQA-fused split-KV decode
- efficient AoS compressed cache reads
- vectorized shared-KV-head processing
- CUDA-graph-friendly decode behavior

This file is important because it shows the performance bar the stage1+2 path ultimately needs to meet.

### 4. Stage1+2 packed-cache adapter

#### `vllm/v1/attention/ops/qjl_backend_proto.py`

This is the current live implementation for `turboquant_qjl`.

It provides:

- slot layout definition
- packed-cache views
- stage-A 2-bit key pack/unpack
- stage-B residual sketch pack/unpack helpers
- token and block packers into the backend cache layout
- packed decode score path
- packed shared-prefix score path
- packed value decode / weighted sum path

This file is not just "temporary glue" anymore. It currently holds the working stage1+2 algorithmic path for GPT-OSS.

### 5. Low-level QJL / packed-score kernels

#### `vllm/v1/attention/ops/qjl_kernels.py`

This file now carries the low-level pieces we actually needed:

- fixed Triton bit-packing kernel
- QJL helper kernels and projections
- value quantization helpers
- fused packed stage1+2 Triton score kernel for decode

Most important recent addition:

- a fused Triton kernel that computes stage-A coarse score plus stage-B residual correction directly from packed bytes, without expanding to full boolean residual tensors

That kernel is the main reason recent decode throughput moved from:

- `183.3 -> 203.4 tok/s` at `256 / batch 8 / 16`
- `62.0 -> 67.9 tok/s` at `2048 / batch 2 / 16`
- `34.1 -> 42.6 tok/s` at `8192 / batch 2 / 16`

## Wiring Map

If you want to understand "which file is wired where", this is the shortest correct map.

### Runtime path for `kv_cache_dtype='turboquant_qjl'`

1. Model attention layer asks for KV-cache spec.
2. `vllm/model_executor/layers/attention/attention.py`
   computes compressed slot size using `turboquant_qjl_slot_size(...)`.
3. vLLM allocates the packed uint8 cache.
4. `vllm/v1/attention/backends/turboquant_attn.py`
   owns the backend implementation and chooses the QJL/TurboQuant stage1+2 path.
5. On KV write:
   `turboquant_attn.py -> qjl_backend_proto.pack_tokens_to_packed_qjl_cache(...)`
6. On decode score:
   `turboquant_attn.py -> qjl_backend_proto.decode_packed_qjl(...)`
7. Inside packed decode score:
   `qjl_backend_proto.py -> qjl_kernels.qjl_score_stage12_triton(...)`
8. On packed value decode:
   `qjl_backend_proto.py -> cuda_quantized_bmm_gqa_dynamic(...)`

### Runtime path for `kv_cache_dtype='turboquant_4bit'`

1. `attention.py` computes 4-bit compressed head size.
2. `turboquant_attn.py` selects the 4-bit backend path.
3. KV write uses:
   `triton_turboquant_kv.turboquant_compress_kv(...)`
4. Decode uses:
   `triton_turboquant_paged_attn.turboquant_fused_paged_decode(...)`
5. Mixed prefill/decode uses decompress-to-temp then standard attention.

## What Still Needs To Be Built

These are the next tasks that matter if the goal is the real TurboQuant outcome, not just another incremental local speedup.

### 1. Fused shared-prefix stage1+2 scorer

Current problem:

- decode now uses the fused packed Triton scorer
- shared-prefix prefill still uses the grouped eager/PyTorch scorer

Why it matters:

- long-context GPT-OSS serving is not decode-only
- TTFT and mixed prompt steps still matter a lot
- until this path is fused too, the end-to-end long-context gap to FP8 stays too large

### 2. Move stage-B write packing out of eager Torch

Current problem:

- the residual sketch write path in `qjl_backend_proto.py` still does:
  - residual reconstruction
  - matmul against projection
  - sign pack
  in eager Torch

Why it matters:

- this is not yet the final "true TurboQuant ops path"
- it keeps the write path more expensive and more adapter-shaped than it should be

### 3. Finish the true stage-2 ops path in `triton_turboquant_kv.py`

Current problem:

- the stage-2 section in `triton_turboquant_kv.py` still has placeholder JL-sketch logic

Why it matters:

- the final architecture should live under TurboQuant-native ops, not depend forever on `qjl_backend_proto.py`
- once the stage-2 pack path exists here, the backend can become cleaner and more maintainable

### 4. Fused weighted-sum path for packed stage1+2

Current problem:

- the score path is better now
- the packed value decode path is still not on a strong fused kernel path for all cases

Why it matters:

- once score gets cheaper, the next bottleneck becomes packed value reconstruction and accumulation

### 5. Real serving-scale benchmark, not just per-request smoke

We still need the benchmark that really answers the project question:

- at fixed H100 memory
- fixed long context
- increasing admitted requests
- does TurboQuant overtake FP8 on aggregate output tok/s because it fits more work?

That is the business end of the project.

## Recommended Next Execution Order

1. Fuse `packed_qjl_prefix_scores_multi_query(...)` with a Triton kernel analogous to the decode scorer.
2. Benchmark again at `8k`, `16k`, `32k`, and `65k`.
3. Move residual sketch pack/write into TurboQuant-native Triton ops.
4. Re-benchmark write-heavy and mixed prefill workloads.
5. Run concurrency sweeps to measure true aggregate serving throughput vs FP8.

## Bottom Line

What we have today is already meaningful:

- real KV-capacity gain over FP8
- real CUDA-graph-safe packed stage1+2 decode path
- real throughput progress on GPT-OSS-20B
- no external dependency on the author prototype

What we do **not** have yet:

- a finished production TurboQuant stage1+2 implementation
- enough end-to-end long-context throughput to claim victory over FP8

The project is now past the "does it work?" stage.

The current phase is:

**make the remaining packed stage1+2 attention paths as fused and native as the decode scorer, then prove the capacity win turns into serving throughput on GPT-OSS-20B.**
