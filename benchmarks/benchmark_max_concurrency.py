#!/usr/bin/env python3
"""Max concurrency test: How many requests can each method handle in parallel?

TurboQuant's heavier KV compression (3K/2V) should allow more concurrent
requests at the same GPU memory utilization vs FP8.
"""

import os
os.environ["HF_HOME"] = "/workspace/.hf_home"
os.environ["HF_HUB_OFFLINE"] = "0"

import vllm
from vllm import SamplingParams

MODEL = "/workspace/models/gpt-oss-20b"
MAX_TOKENS = 64
GPU_MEM = 0.90  # Same memory for both

PROMPTS_SHORT = ["The capital of France is"] * 8
PROMPTS_MED = ["def fibonacci(n):"] * 8

def find_max_concurrency(name, llm, prompts, max_tokens, gpu_mem_label):
    """Find max concurrent requests before OOM or severe degradation."""
    print(f"\n--- {name} ---")
    SP = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)

    # Test different concurrency levels
    for n in [8, 16, 32, 64, 128, 256, 512]:
        p = prompts * (n // len(prompts) + 1)
        p = p[:n]
        try:
            start = time.perf_counter()
            outputs = llm.generate(p, SP)
            elapsed = time.perf_counter() - start
            tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            tps = tokens / elapsed
            print(f"  {n:4d} requests: {tokens:5d} tokens, {tps:7.1f} tok/s, {elapsed:.2f}s")
        except Exception as e:
            print(f"  {n:4d} requests: FAILED — {e}")
            break

import time

# FP8 KV — find max concurrency
print("Loading FP8 KV...")
llm_fp8 = vllm.LLM(
    model=MODEL, trust_remote_code=True, tensor_parallel_size=1,
    gpu_memory_utilization=GPU_MEM, max_model_len=4096,
    enforce_eager=False, disable_custom_all_reduce=True,
    kv_cache_dtype="fp8",
)
find_max_concurrency(f"FP8 KV (gpu_mem={GPU_MEM})", llm_fp8, PROMPTS_SHORT, MAX_TOKENS, "FP8")
del llm_fp8

# TurboQuant — find max concurrency
from vllm.v1.attention.backends.turboquant_fused_backend import enable_turboquant_fused
enable_turboquant_fused(
    key_bits=3, value_bits=2, value_group_size=32,
    ring_capacity=128, initial_layers_count=4,
    mode="hybrid", no_alloc=False,
)
print("\nLoading TurboQuant...")
llm_tq = vllm.LLM(
    model=MODEL, trust_remote_code=True, tensor_parallel_size=1,
    gpu_memory_utilization=GPU_MEM, max_model_len=4096,
    enforce_eager=False, disable_custom_all_reduce=True,
)
find_max_concurrency(f"TQ (gpu_mem={GPU_MEM})", llm_tq, PROMPTS_SHORT, MAX_TOKENS, "TQ")
del llm_tq

print("\nNote: TQ compression ratio vs FP8:")
print(f"  FP8:  8-bit key + 8-bit value per element")
print(f"  TQ:   3-bit key + 2-bit value per element (~61% smaller KV)")
print(f"  TQ should fit ~2.5x more tokens in same GPU memory")