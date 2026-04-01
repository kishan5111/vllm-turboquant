#!/usr/bin/env python3
"""Max concurrency test: How many concurrent requests can each method handle?

With same gpu_memory_utilization, TQ's denser KV (3K/2V vs FP8 8-bit)
should fit more tokens in KV cache → handle more concurrent requests.
"""

import os, time
os.environ["HF_HOME"] = "/workspace/.hf_home"
os.environ["HF_HUB_OFFLINE"] = "0"

import vllm
from vllm import SamplingParams

MODEL = "/workspace/models/gpt-oss-20b"
MAX_TOKENS = 128
MAX_MODEL_LEN = 65536
GPU_MEM = 0.90  # Same memory for both

PROMPTS = [
    "The capital of France is",
    "The quick brown fox jumps over",
    "def fibonacci(n):",
    "SELECT * FROM users WHERE",
    "import torch\n\nx = torch.randn",
    "class Attention(nn.Module):",
    "The answer to life the universe and everything is",
    "In a galaxy far far away",
]
SP = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS)

def test_concurrency(name, llm, num_requests):
    p = PROMPTS * (num_requests // len(PROMPTS) + 1)
    p = p[:num_requests]
    start = time.perf_counter()
    outputs = llm.generate(p, SP)
    elapsed = time.perf_counter() - start
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / elapsed
    return total_tokens, elapsed, tps

# FP8 KV
print("Loading FP8 KV (gpu_memory_utilization=0.90, max_model_len=65536)...")
llm_fp8 = vllm.LLM(
    model=MODEL, trust_remote_code=True, tensor_parallel_size=1,
    gpu_memory_utilization=GPU_MEM, max_model_len=MAX_MODEL_LEN,
    enforce_eager=False, disable_custom_all_reduce=True,
    kv_cache_dtype="fp8",
)

# Warmup
print("FP8 warmup...")
_ = llm_fp8.generate(PROMPTS[:8], SP)

print("\n--- FP8 KV ---")
for n in [8, 16, 32, 64, 128, 256]:
    try:
        tokens, elapsed, tps = test_concurrency("FP8", llm_fp8, n)
        print(f"  {n:4d} requests: {tokens:5d} tokens, {tps:8.1f} tok/s, {elapsed:.2f}s")
    except Exception as e:
        print(f"  {n:4d} requests: FAILED — {str(e)[:80]}")
        break
del llm_fp8

# TurboQuant
from vllm.v1.attention.backends.turboquant_fused_backend import enable_turboquant_fused
enable_turboquant_fused(
    key_bits=3, value_bits=2, value_group_size=32,
    ring_capacity=128, initial_layers_count=4,
    mode="hybrid", no_alloc=False,
)
print("\nLoading TurboQuant (gpu_memory_utilization=0.90, max_model_len=65536)...")
llm_tq = vllm.LLM(
    model=MODEL, trust_remote_code=True, tensor_parallel_size=1,
    gpu_memory_utilization=GPU_MEM, max_model_len=MAX_MODEL_LEN,
    enforce_eager=False, disable_custom_all_reduce=True,
)

# Warmup
print("TQ warmup...")
_ = llm_tq.generate(PROMPTS[:8], SP)

print("\n--- TurboQuant ---")
for n in [8, 16, 32, 64, 128, 256]:
    try:
        tokens, elapsed, tps = test_concurrency("TQ", llm_tq, n)
        print(f"  {n:4d} requests: {tokens:5d} tokens, {tps:8.1f} tok/s, {elapsed:.2f}s")
    except Exception as e:
        print(f"  {n:4d} requests: FAILED — {str(e)[:80]}")
        break
del llm_tq

print(f"\nNote: Same GPU memory (gpu_mem={GPU_MEM}) for both. TQ uses 3K/2V compression (~0.625 bytes/el) vs FP8 8-bit (~1 byte/el).")