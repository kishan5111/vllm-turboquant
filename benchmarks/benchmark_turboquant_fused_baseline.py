#!/usr/bin/env python3
"""Baseline benchmark - no TurboQuant."""

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

llm = vllm.LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    enable_prefix_caching=False,
    disable_custom_all_reduce=True,
    enforce_eager=True,
)
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS, min_tokens=1)

# Warmup
_ = llm.generate(prompts[:2], sampling_params)

print("Running 3 benchmark rounds...")
for run in range(3):
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  Run {run+1}: {elapsed:.2f}s, {tokens} tokens, {tokens/elapsed:.1f} tok/s")

print("Baseline benchmark complete!")