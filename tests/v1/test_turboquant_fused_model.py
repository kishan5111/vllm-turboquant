#!/usr/bin/env python3
"""
Test TurboQuant Fused backend with real GPT-OSS 20B model.

Usage:
    HF_HOME=/workspace/.hf_home .venv/bin/python test_turboquant_fused_model.py
"""

import os
import sys

# Set HF_HOME before any imports
os.environ["HF_HOME"] = "/workspace/.hf_home"
os.environ["HF_HUB_OFFLINE"] = "0"

# Enable TurboQuant Fused integration BEFORE creating LLM
from vllm.v1.attention.backends.turboquant_fused_backend import (
    enable_turboquant_fused,
    MODE_CAPTURE_ONLY,
)

# Install hooks in hybrid mode (TU captures KV and uses fused decode)
enable_turboquant_fused(
    key_bits=3,
    value_bits=2,
    value_group_size=32,
    ring_capacity=128,
    initial_layers_count=4,
    mode="hybrid",  # Use hybrid for actual TU decode
    no_alloc=False,
)

import vllm
from vllm import SamplingParams

print("=" * 80)
print("  TurboQuant Fused + GPT-OSS 20B Integration Test")
print("=" * 80)

# Load model
print("\nLoading GPT-OSS 20B model...")
model_path = "/workspace/models/gpt-oss-20b"

llm = vllm.LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    enable_prefix_caching=False,
    disable_custom_all_reduce=True,
    enforce_eager=True,  # Avoid CUDA graph capture issues with TQ hooks
)

print("Model loaded successfully!")

# Test decode
print("\nRunning decode test...")
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=32,
    min_tokens=1,
)

prompts = [
    "The capital of France is",
    "The quick brown fox",
]

# Run generation
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"\nPrompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")

print("\n" + "=" * 80)
print("  TurboQuant Fused Stats")
print("=" * 80)

# Try to get stats from model runner
try:
    model_runner = llm.llm_engine.executor.driver_worker.model_runner
    from vllm.v1.attention.backends.turboquant_fused_backend import get_stats
    stats = get_stats(model_runner)
    if stats:
        print(f"  Mode: {stats.get('mode', 'N/A')}")
        print(f"  Num layers: {stats.get('num_layers', 0)}")
        print(f"  Total compressed tokens: {stats.get('total_compressed_tokens', 0)}")
        print(f"  Total buffered tokens: {stats.get('total_buffered_tokens', 0)}")
        print(f"  Total memory: {stats.get('total_memory_bytes', 0) / 1e6:.1f} MB")
    else:
        print("  No stats available")
except Exception as e:
    print(f"  Could not retrieve stats: {e}")

print("\nTest completed!")