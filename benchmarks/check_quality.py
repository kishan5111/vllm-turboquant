#!/usr/bin/env python3
"""Verify each method is deterministic (temp=0) — run same prompt 3 times."""

import os
os.environ["HF_HOME"] = "/workspace/.hf_home"
os.environ["HF_HUB_OFFLINE"] = "0"

import vllm
from vllm import SamplingParams

MODEL = "/workspace/models/gpt-oss-20b"
PROMPT = "The capital of France is"
SP = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)

# FP8 KV — 3 runs
print("Loading FP8 KV...")
llm_fp8 = vllm.LLM(
    model=MODEL, trust_remote_code=True, tensor_parallel_size=1,
    gpu_memory_utilization=0.85, max_model_len=4096,
    enforce_eager=False, disable_custom_all_reduce=True,
    kv_cache_dtype="fp8",
)
print("FP8 KV runs:")
for r in range(3):
    outs = llm_fp8.generate([PROMPT], SP)
    print(f"  Run {r+1}: {outs[0].outputs[0].text!r}")
del llm_fp8

# TurboQuant — 3 runs
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
print("TurboQuant runs:")
for r in range(3):
    outs = llm_tq.generate([PROMPT], SP)
    print(f"  Run {r+1}: {outs[0].outputs[0].text!r}")
del llm_tq

print("\nBoth methods should produce identical output across runs (temp=0).")
print("Different outputs between FP8 and TQ are expected — different quantization.")