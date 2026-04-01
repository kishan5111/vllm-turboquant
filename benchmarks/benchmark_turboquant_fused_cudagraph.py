#!/usr/bin/env python3
"""Benchmark: FP8 KV + cudagraph vs TurboQuant + cudagraph"""

import os, time
os.environ["HF_HOME"] = "/workspace/.hf_home"
os.environ["HF_HUB_OFFLINE"] = "0"

import vllm
from vllm import SamplingParams

MODEL = "/workspace/models/gpt-oss-20b"
MAX_TOKENS = 128
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

def run(name, llm):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    _ = llm.generate(PROMPTS[:2], SP)  # warmup + cudagraph capture
    results = []
    for i in range(3):
        start = time.perf_counter()
        outputs = llm.generate(PROMPTS, SP)
        elapsed = time.perf_counter() - start
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = tokens / elapsed
        results.append(tps)
        print(f"  Run {i+1}: {elapsed:.2f}s, {tokens} tokens, {tps:.1f} tok/s")
    avg = sum(results) / len(results)
    print(f"  Average: {avg:.1f} tok/s")
    return avg

# FP8 KV + cudagraph (FlashInfer)
print("Loading FP8 KV baseline...")
llm_fp8 = vllm.LLM(
    model=MODEL, trust_remote_code=True, tensor_parallel_size=1,
    gpu_memory_utilization=0.85, max_model_len=4096,
    enforce_eager=False, disable_custom_all_reduce=True,
    kv_cache_dtype="fp8",
)
fp8_tps = run("FP8 KV + cudagraph (FlashInfer)", llm_fp8)
del llm_fp8

# TurboQuant + cudagraph
from vllm.v1.attention.backends.turboquant_fused_backend import enable_turboquant_fused
enable_turboquant_fused(
    key_bits=3, value_bits=2, value_group_size=32,
    ring_capacity=128, initial_layers_count=4,
    mode="hybrid", no_alloc=False,
)
print("\nLoading TurboQuant...")
llm_tq = vllm.LLM(
    model=MODEL, trust_remote_code=True, tensor_parallel_size=1,
    gpu_memory_utilization=0.85, max_model_len=4096,
    enforce_eager=False, disable_custom_all_reduce=True,
)
tq_tps = run("TurboQuant + cudagraph", llm_tq)
del llm_tq

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  FP8 KV + cudagraph:  {fp8_tps:.1f} tok/s")
print(f"  TurboQuant + cudagraph: {tq_tps:.1f} tok/s")
print(f"  Delta:               {tq_tps - fp8_tps:+.1f} tok/s ({tq_tps/fp8_tps:.2f}x)")
print(f"{'='*60}")