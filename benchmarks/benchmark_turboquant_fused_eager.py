#!/usr/bin/env python3
"""Proper benchmark: both baseline and TurboQuant in eager mode (fair comparison)."""

import os, time
os.environ["HF_HOME"] = "/workspace/.hf_home"
os.environ["HF_HUB_OFFLINE"] = "0"

import vllm
from vllm import SamplingParams

MODEL_PATH = "/workspace/models/gpt-oss-20b"
MAX_TOKENS = 128

prompts = [
    "The capital of France is",
    "The quick brown fox jumps over",
    "def fibonacci(n):",
    "SELECT * FROM users WHERE",
    "import torch\n\nx = torch.randn",
    "class Attention(nn.Module):",
    "The answer to life the universe and everything is",
    "In a galaxy far far away",
]

def run(name, llm, desc):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, min_tokens=1)

    # Warmup
    print("  Warming up...")
    _ = llm.generate(prompts[:2], sampling_params)

    print("  Running 3 rounds...")
    results = []
    for i in range(3):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = tokens / elapsed
        results.append(tps)
        print(f"    Round {i+1}: {elapsed:.2f}s, {tokens} tokens, {tps:.1f} tok/s")

    avg = sum(results) / len(results)
    print(f"  Average: {avg:.1f} tok/s")
    return avg

# ===== BASELINE (eager mode) =====
llm_baseline = vllm.LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    enable_prefix_caching=False,
    disable_custom_all_reduce=True,
    enforce_eager=True,  # Same as TQ
)
baseline_tps = run("BASELINE (bf16 KV, eager, FlashAttention)", llm_baseline, "baseline")
del llm_baseline

# ===== TURBOQUANT HYBRID (eager mode) =====
from vllm.v1.attention.backends.turboquant_fused_backend import enable_turboquant_fused
enable_turboquant_fused(
    key_bits=3, value_bits=2, value_group_size=32,
    ring_capacity=128, initial_layers_count=4,
    mode="hybrid", no_alloc=False,
)

llm_turboquant = vllm.LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    enable_prefix_caching=False,
    disable_custom_all_reduce=True,
    enforce_eager=True,  # Same as baseline
)
turboquant_tps = run("TURBOQUANT (3-bit key, 2-bit value, eager)", llm_turboquant, "turboquant")
del llm_turboquant

# ===== SUMMARY =====
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  Baseline:       {baseline_tps:.1f} tok/s")
print(f"  TurboQuant:     {turboquant_tps:.1f} tok/s")
print(f"  Delta:          {turboquant_tps - baseline_tps:+.1f} tok/s ({turboquant_tps/baseline_tps:.2f}x)")
print(f"{'='*60}")
print(f"\nBoth used enforce_eager=True. KV dtype: baseline=bf16, TQ=3-bit key + 2-bit value")