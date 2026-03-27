#!/usr/bin/env python3
"""
End-to-end throughput benchmark: FP8 vs TurboQuant KV cache.

Runs offline inference (no server) for direct comparison.
Measures: output tok/s, TTFT, per-request latency, peak VRAM.

Workloads:
  - high_context    : input=8192  output=64
  - high_throughput : input=2048  output=128
  - concurrency_16  : input=1024  output=64  concurrency=16
  - concurrency_64  : input=1024  output=64  concurrency=64
  - concurrency_128 : input=1024  output=64  concurrency=128

Outputs: results/fp8_baseline.json or results/tq_baseline.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")


WORKLOADS = {
    "high_context": {"input_len": 8192, "output_len": 64, "num_requests": 8},
    "high_throughput": {"input_len": 2048, "output_len": 128, "num_requests": 32},
    "concurrency_16": {"input_len": 1024, "output_len": 64, "num_requests": 16},
    "concurrency_64": {"input_len": 1024, "output_len": 64, "num_requests": 64},
    "concurrency_128": {"input_len": 1024, "output_len": 64, "num_requests": 128},
}


def _make_prompts(tokenizer, input_len: int, n: int) -> list[str]:
    """Generate n prompts that each tokenize to approximately input_len tokens."""
    # Use a repeated token pattern to get precise length
    tok_id = tokenizer.encode("the", add_special_tokens=False)[0]
    tokens = [tok_id] * input_len
    prompt = tokenizer.decode(tokens)
    return [prompt] * n


def _run_workload(llm, tokenizer, name: str, input_len: int, output_len: int,
                  num_requests: int, kv_dtype: str) -> dict:
    from vllm import SamplingParams

    prompts = _make_prompts(tokenizer, input_len, num_requests)
    sp = SamplingParams(max_tokens=output_len, temperature=0, ignore_eos=True)

    # Warmup (1 request)
    warmup_p = _make_prompts(tokenizer, min(input_len, 512), 1)
    llm.generate(warmup_p, SamplingParams(max_tokens=8, temperature=0))
    torch.cuda.synchronize()

    peak_vram_before = torch.cuda.max_memory_allocated() / 1024**3
    torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    total_time = t_end - t_start
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_input_tokens  = sum(len(o.prompt_token_ids or []) for o in outputs)

    # Per-request latencies
    req_latencies = []
    for o in outputs:
        if o.metrics and o.metrics.finished_time and o.metrics.first_scheduled_time:
            req_latencies.append(o.metrics.finished_time - o.metrics.first_scheduled_time)

    p99_latency = sorted(req_latencies)[int(len(req_latencies) * 0.99)] if req_latencies else 0

    result = {
        "workload": name,
        "backend": kv_dtype,
        "num_requests": num_requests,
        "input_len": input_len,
        "output_len": output_len,
        "total_time_s": round(total_time, 3),
        "total_output_tokens": total_output_tokens,
        "output_tok_per_s": round(total_output_tokens / total_time, 1),
        "request_per_s": round(num_requests / total_time, 2),
        "p99_latency_s": round(p99_latency, 3),
        "peak_vram_gb": round(peak_vram, 2),
        "effective_kv_bytes_per_token": _kv_bytes_per_token(kv_dtype),
    }
    print(f"  [{name:20s}] {result['output_tok_per_s']:8.1f} tok/s  "
          f"req/s={result['request_per_s']:.2f}  "
          f"p99={result['p99_latency_s']:.3f}s  "
          f"vram={result['peak_vram_gb']:.1f}GB")
    return result


def _kv_bytes_per_token(kv_dtype: str) -> float:
    """Bytes per token per KV head pair for the given dtype."""
    HEAD_SIZE = 128
    NUM_KV_HEADS = 8
    if kv_dtype.startswith("turboquant"):
        return (HEAD_SIZE // 2 + 2) * 2 * NUM_KV_HEADS  # 66 * 2 * 8 = 1056
    elif kv_dtype.startswith("fp8"):
        return HEAD_SIZE * 2 * NUM_KV_HEADS              # 128 * 2 * 8 = 2048
    else:
        return HEAD_SIZE * 2 * 2 * NUM_KV_HEADS          # bf16: 256 * 2 * 8 = 4096


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--kv-dtype", default="fp8", choices=["fp8", "turboquant_4bit", "auto"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOADS.keys()))
    args = parser.parse_args()

    if args.output is None:
        tag = args.kv_dtype.replace("_", "-")
        args.output = f"results/{tag}_baseline.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Backend  : {args.kv_dtype}")
    print(f"Model    : {args.model}")
    print(f"Max len  : {args.max_model_len}")
    print(f"GPU util : {args.gpu_util}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        kv_cache_dtype=args.kv_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=False,
    )

    results = []
    for wl_name in args.workloads:
        if wl_name not in WORKLOADS:
            print(f"Unknown workload: {wl_name}, skipping")
            continue
        wl = WORKLOADS[wl_name]
        # Skip workloads that exceed max_model_len
        if wl["input_len"] + wl["output_len"] > args.max_model_len:
            print(f"  [{wl_name:20s}] SKIPPED (input+output > max_model_len)")
            continue
        try:
            r = _run_workload(
                llm, tokenizer, wl_name,
                wl["input_len"], wl["output_len"], wl["num_requests"],
                args.kv_dtype,
            )
            results.append(r)
        except Exception as e:
            print(f"  [{wl_name:20s}] ERROR: {e}")

    del llm
    torch.cuda.empty_cache()

    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
