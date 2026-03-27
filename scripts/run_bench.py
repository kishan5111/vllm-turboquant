#!/usr/bin/env python3
"""
End-to-end throughput benchmark: FP8 vs TurboQuant KV cache.

Runs offline inference (no server) for direct comparison.
Measures: output tok/s, TTFT, per-request latency, peak VRAM.

Workloads:
  - ctx8k            : input=8192   output=64  num_requests=8
  - ctx16k           : input=16384  output=64  num_requests=4
  - ctx32k           : input=32768  output=64  num_requests=4
  - ctx65k           : input=65536  output=64  num_requests=2
  - high_context     : input=8192   output=64
  - high_throughput  : input=2048   output=128
  - concurrency_16   : input=1024   output=64  concurrency=16
  - concurrency_64   : input=1024   output=64  concurrency=64
  - concurrency_128  : input=1024   output=64  concurrency=128

Outputs: results/<model>_<kv_dtype>_baseline.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


WORKLOADS = {
    "ctx8k": {"input_len": 8192, "output_len": 64, "num_requests": 8},
    "ctx16k": {"input_len": 16384, "output_len": 64, "num_requests": 4},
    "ctx32k": {"input_len": 32768, "output_len": 64, "num_requests": 4},
    "ctx65k": {"input_len": 65536, "output_len": 64, "num_requests": 2},
    "high_context": {"input_len": 8192, "output_len": 64, "num_requests": 8},
    "high_throughput": {"input_len": 2048, "output_len": 128, "num_requests": 32},
    "concurrency_16": {"input_len": 1024, "output_len": 64, "num_requests": 16},
    "concurrency_64": {"input_len": 1024, "output_len": 64, "num_requests": 64},
    "concurrency_128": {"input_len": 1024, "output_len": 64, "num_requests": 128},
}


def _slugify_model_name(model: str) -> str:
    return model.split("/")[-1].replace(".", "-").replace("_", "-").lower()


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
    ttfts = []
    prefill_times = []
    decode_times = []
    tpot_ms = []
    for o in outputs:
        metrics = o.metrics
        if metrics is None:
            continue
        if (
            getattr(metrics, "finished_time", None)
            and getattr(metrics, "first_scheduled_time", None)
        ):
            req_latencies.append(
                metrics.finished_time - metrics.first_scheduled_time
            )
        ttft = getattr(metrics, "first_token_latency", None)
        if ttft is not None:
            ttfts.append(ttft)
        scheduled_ts = getattr(metrics, "scheduled_ts", None)
        first_token_ts = getattr(metrics, "first_token_ts", None)
        last_token_ts = getattr(metrics, "last_token_ts", None)
        num_gen_tokens = getattr(metrics, "num_generation_tokens", 0) or 0
        if scheduled_ts is not None and first_token_ts is not None:
            prefill_times.append(max(0.0, first_token_ts - scheduled_ts))
        if first_token_ts is not None and last_token_ts is not None:
            decode_time = max(0.0, last_token_ts - first_token_ts)
            decode_times.append(decode_time)
            if num_gen_tokens > 1 and decode_time > 0:
                tpot_ms.append((decode_time / (num_gen_tokens - 1)) * 1e3)

    p99_latency = sorted(req_latencies)[int(len(req_latencies) * 0.99)] if req_latencies else 0
    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
    mean_prefill = sum(prefill_times) / len(prefill_times) if prefill_times else 0
    mean_decode = sum(decode_times) / len(decode_times) if decode_times else 0
    mean_tpot_ms = sum(tpot_ms) / len(tpot_ms) if tpot_ms else 0
    total_decode_time = sum(decode_times)
    total_decode_tokens = sum(
        max(0, len(o.outputs[0].token_ids) - 1)
        for o in outputs
    )

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
        "mean_ttft_s": round(mean_ttft, 3),
        "mean_prefill_s": round(mean_prefill, 3),
        "mean_decode_s": round(mean_decode, 3),
        "mean_tpot_ms": round(mean_tpot_ms, 3),
        "post_first_token_tok_per_s": round(
            total_decode_tokens / total_decode_time, 1
        ) if total_decode_time > 0 else 0,
        "peak_vram_gb": round(peak_vram, 2),
        "effective_kv_bytes_per_token": _kv_bytes_per_token(kv_dtype),
    }
    print(f"  [{name:20s}] {result['output_tok_per_s']:8.1f} tok/s  "
          f"req/s={result['request_per_s']:.2f}  "
          f"ttft={result['mean_ttft_s']:.3f}s  "
          f"decode={result['post_first_token_tok_per_s']:.1f} tok/s  "
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


def _profile_turboquant_paths(model: str, workloads: list[str]) -> list[dict]:
    from scripts.microbench_kv_path import (
        profile_turboquant_paths,
        topology_from_model,
    )

    ctx_lens = sorted({WORKLOADS[name]["input_len"] for name in workloads})
    num_seqs = sorted({WORKLOADS[name]["num_requests"] for name in workloads})
    topology = topology_from_model(model)
    return profile_turboquant_paths(ctx_lens, num_seqs, topology)


def _resolve_cudagraph_mode(requested_mode: str, kv_dtype: str) -> str:
    """Pick a safe default graph mode for the selected KV backend."""
    if requested_mode != "auto":
        return requested_mode

    # TurboQuant's mixed PIECEWISE graphs are healthy, but FULL decode graphs
    # are currently expensive to capture because the custom Triton decode path
    # specializes heavily on exact decode batch shapes.
    if kv_dtype.startswith("turboquant"):
        return "piecewise"

    return "full_decode_only"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--kv-dtype", default="fp8", choices=["fp8", "turboquant_4bit", "auto"])
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--no-chunked-prefill", action="store_true", default=False)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--workloads", nargs="+", default=["ctx8k", "ctx16k"])
    parser.add_argument("--profile-turboquant-paths", action="store_true", default=False)
    parser.add_argument(
        "--cudagraph-mode",
        default="auto",
        choices=[
            "auto",
            "none",
            "piecewise",
            "full",
            "full_decode_only",
            "full_and_piecewise",
        ],
    )
    args = parser.parse_args()
    resolved_cudagraph_mode = _resolve_cudagraph_mode(
        args.cudagraph_mode, args.kv_dtype
    )

    if args.output is None:
        tag = args.kv_dtype.replace("_", "-")
        model_tag = _slugify_model_name(args.model)
        args.output = f"results/{model_tag}_{tag}_baseline.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Backend  : {args.kv_dtype}")
    print(f"Attention: {args.attention_backend or 'auto'}")
    print(f"Model    : {args.model}")
    print(f"Max len  : {args.max_model_len}")
    print(f"GPU util : {args.gpu_util}")
    print(f"Chunked  : {not args.no_chunked_prefill}")
    print(f"CUDAGraph: {resolved_cudagraph_mode} (requested: {args.cudagraph_mode})")
    print(f"{'='*60}")

    import os
    if args.attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        kv_cache_dtype=args.kv_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        disable_log_stats=False,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=not args.no_chunked_prefill,
        compilation_config={"cudagraph_mode": resolved_cudagraph_mode},
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

    profile_rows: list[dict] = []
    if args.profile_turboquant_paths:
        print("\nProfiling TurboQuant KV paths...")
        try:
            profile_rows = _profile_turboquant_paths(args.model, args.workloads)
            for row in profile_rows:
                if row["backend"] == "compression_ratio":
                    continue
                print(f"  [{row['backend']:24s}] ctx={row['ctx_len']:5d} seqs={row['num_seqs']:3d} "
                      f"{row['latency_ms']:8.3f} ms")
        except Exception as e:
            print(f"  [profile_turboquant_paths] ERROR: {e}")

    del llm
    torch.cuda.empty_cache()

    with open(args.output, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "resolved_cudagraph_mode": resolved_cudagraph_mode,
                "results": results,
                "profile": profile_rows,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
