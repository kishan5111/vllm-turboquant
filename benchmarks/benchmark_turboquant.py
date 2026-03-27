# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark TurboQuant KV-cache vs FP8 and BF16 baselines on Qwen3-8B.

Measures:
  - GPU memory used by KV cache (MB)
  - Time-to-first-token / prefill latency (ms)
  - Decoding throughput (tokens/s)
  - End-to-end latency for a batch of requests (ms)

Usage:
  python benchmarks/benchmark_turboquant.py
  python benchmarks/benchmark_turboquant.py --model Qwen/Qwen3-8B --output results.json
"""

import argparse
import json
import os
import time
from typing import Any

import torch
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="TurboQuant vs FP8 benchmark")
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--output", default=None, help="Save JSON results to this path")
parser.add_argument(
    "--dtypes",
    nargs="+",
    default=["turboquant_qjl", "turboquant_4bit", "fp8_e4m3", "auto"],
    help="KV cache dtypes to benchmark",
)
parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16])
parser.add_argument(
    "--prompt-len", type=int, default=512, help="Input prompt length in tokens"
)
parser.add_argument(
    "--gen-len", type=int, default=256, help="Output tokens to generate"
)
parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
parser.add_argument("--iters", type=int, default=5, help="Benchmark iterations")
parser.add_argument(
    "--gpu-mem-util", type=float, default=0.90, help="GPU memory utilization"
)
parser.add_argument("--block-size", type=int, default=None)
parser.add_argument("--enable-chunked-prefill", action="store_true")
parser.add_argument("--enforce-eager", action="store_true")
parser.add_argument("--max-num-batched-tokens", type=int, default=None)
args = parser.parse_args()

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
TOKENIZER = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

# ---------------------------------------------------------------------------
# Imports (after arg parse so --help works without GPU)
# ---------------------------------------------------------------------------

from vllm import LLM, SamplingParams  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prompt_token_ids(length: int) -> list[int]:
    """Create a synthetic prompt with exactly `length` tokens."""
    seed_ids = TOKENIZER.encode(
        "TurboQuant benchmark prompt. The quick brown fox jumps over the lazy dog.\n",
        add_special_tokens=False,
    )
    if not seed_ids:
        raise RuntimeError("Tokenizer produced an empty prompt seed.")
    repeats = (length + len(seed_ids) - 1) // len(seed_ids)
    prompt_ids = (seed_ids * repeats)[:length]
    assert len(prompt_ids) == length
    return prompt_ids


def _gpu_mem_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def _peak_gpu_mem_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def _create_llm(dtype: str, prompt_len: int, gen_len: int) -> tuple[LLM, float]:
    """Create one engine per dtype and reuse it across batch sizes."""
    engine_kwargs: dict[str, Any] = dict(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=prompt_len + gen_len + 64,
    )
    if dtype != "auto":
        engine_kwargs["kv_cache_dtype"] = dtype
    if args.block_size is not None:
        engine_kwargs["block_size"] = args.block_size
    engine_kwargs["enable_chunked_prefill"] = args.enable_chunked_prefill
    if args.enforce_eager or dtype == "turboquant_qjl":
        engine_kwargs["enforce_eager"] = True
    if dtype == "turboquant_qjl":
        engine_kwargs["enable_prefix_caching"] = False
    if args.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    torch.cuda.reset_peak_memory_stats()
    mem_before = _gpu_mem_mb()
    llm = LLM(**engine_kwargs)
    mem_after_load = _peak_gpu_mem_mb()
    return llm, mem_after_load - mem_before


def measure_prefill_latency(
    dtype: str,
    prompt_len: int,
    iters: int,
) -> tuple[float, float]:
    """Measure TTFT on a dedicated engine so batch E2E runs stay uncontaminated."""
    llm, kv_mem_overhead_mb = _create_llm(
        dtype=dtype,
        prompt_len=prompt_len,
        gen_len=1,
    )
    prefill_params = SamplingParams(temperature=0.0, max_tokens=1)
    prompt_token_ids = _make_prompt_token_ids(prompt_len)
    prefill_prompt = [{"prompt_token_ids": prompt_token_ids}]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        llm.generate(prefill_prompt, prefill_params, use_tqdm=False)
    torch.cuda.synchronize()
    t_prefill_ms = (time.perf_counter() - t0) / iters * 1000

    del llm
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return t_prefill_ms, kv_mem_overhead_mb


def run_benchmark(
    llm: LLM,
    dtype: str,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    warmup: int,
    iters: int,
    prefill_latency_ms: float,
    kv_mem_overhead_mb: float,
) -> dict[str, Any]:
    """Run one benchmark configuration and return a result dict."""
    print(f"\n{'='*60}")
    print(f"  dtype={dtype}  batch={batch_size}  prompt={prompt_len}  gen={gen_len}")
    print(f"{'='*60}")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=gen_len,
        ignore_eos=True,
    )

    prompt_token_ids = _make_prompt_token_ids(prompt_len)
    prompts = [
        {"prompt_token_ids": prompt_token_ids}
        for _ in range(batch_size)
    ]

    # Warmup
    print(f"  Warming up ({warmup} iters)...")
    for _ in range(warmup):
        llm.generate(prompts, sampling_params, use_tqdm=False)

    # ---- E2E latency + throughput ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    t_e2e_ms = elapsed / iters * 1000

    # Actual tokens generated
    total_gen = sum(
        len(o.outputs[0].token_ids) for o in outputs
    )
    throughput_tps = total_gen * iters / max(elapsed, 1e-9)

    peak_mb = _peak_gpu_mem_mb()

    result = {
        "dtype": dtype,
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "prefill_latency_ms": round(prefill_latency_ms, 2),
        "e2e_latency_ms": round(t_e2e_ms, 2),
        "decode_throughput_tps": round(throughput_tps, 1),
        "peak_gpu_mem_mb": round(peak_mb, 1),
        "kv_mem_overhead_mb": round(kv_mem_overhead_mb, 1),
    }

    print(f"  Prefill latency : {result['prefill_latency_ms']:.1f} ms")
    print(f"  E2E latency     : {result['e2e_latency_ms']:.1f} ms")
    print(f"  Throughput      : {result['decode_throughput_tps']:.1f} tok/s")
    print(f"  Peak GPU mem    : {result['peak_gpu_mem_mb']:.0f} MB")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_results: list[dict] = []

    # Run each dtype once, then sweep batch sizes without reloading the model.
    for dtype in args.dtypes:
        try:
            prefill_latency_ms, kv_mem_overhead_mb = measure_prefill_latency(
                dtype=dtype,
                prompt_len=args.prompt_len,
                iters=args.iters,
            )
        except Exception as exc:
            print(f"  ERROR measuring prefill for dtype={dtype}: {exc}")
            for failed_bs in args.batch_sizes:
                all_results.append({
                    "dtype": dtype,
                    "batch_size": failed_bs,
                    "error": str(exc),
                })
            continue

        llm = None
        for bs in args.batch_sizes:
            if llm is None:
                try:
                    llm, _ = _create_llm(
                        dtype=dtype,
                        prompt_len=args.prompt_len,
                        gen_len=args.gen_len,
                    )
                except Exception as exc:
                    print(f"  ERROR creating engine for dtype={dtype}: {exc}")
                    for failed_bs in args.batch_sizes:
                        all_results.append({
                            "dtype": dtype,
                            "batch_size": failed_bs,
                            "error": str(exc),
                        })
                    break
            try:
                r = run_benchmark(
                    llm=llm,
                    dtype=dtype,
                    batch_size=bs,
                    prompt_len=args.prompt_len,
                    gen_len=args.gen_len,
                    warmup=args.warmup,
                    iters=args.iters,
                    prefill_latency_ms=prefill_latency_ms,
                    kv_mem_overhead_mb=kv_mem_overhead_mb,
                )
                all_results.append(r)
            except Exception as exc:
                print(f"  ERROR for dtype={dtype} batch={bs}: {exc}")
                all_results.append({
                    "dtype": dtype,
                    "batch_size": bs,
                    "error": str(exc),
                })
        if llm is not None:
            del llm
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("BENCHMARK SUMMARY  —  Qwen3-8B  |  prompt=%d  gen=%d" % (
        args.prompt_len, args.gen_len))
    print("=" * 90)
    header = (
        f"{'dtype':<20} {'batch':>5} {'prefill(ms)':>12} "
        f"{'e2e(ms)':>10} {'tok/s':>10} {'peak_mem(MB)':>13}"
    )
    print(header)
    print("-" * 90)

    # Group by dtype for relative comparison
    fp8_row: dict[int, dict] = {}
    tq_row:  dict[int, dict] = {}
    for r in all_results:
        if "error" in r:
            continue
        bs = r["batch_size"]
        if r["dtype"] == "fp8_e4m3":
            fp8_row[bs] = r
        elif r["dtype"] == "turboquant_4bit":
            tq_row[bs] = r

    for r in all_results:
        if "error" in r:
            print(f"  {'ERROR':<20} {r['batch_size']:>5}  {r.get('error','')}")
            continue
        print(
            f"  {r['dtype']:<20} {r['batch_size']:>5} "
            f"{r['prefill_latency_ms']:>12.1f} "
            f"{r['e2e_latency_ms']:>10.1f} "
            f"{r['decode_throughput_tps']:>10.1f} "
            f"{r['peak_gpu_mem_mb']:>13.0f}"
        )

    # Relative comparison: TurboQuant vs FP8
    if fp8_row and tq_row:
        print("\n--- TurboQuant vs FP8 relative ---")
        for bs in sorted(set(fp8_row) & set(tq_row)):
            fp = fp8_row[bs]
            tq = tq_row[bs]
            mem_ratio   = fp["peak_gpu_mem_mb"] / max(tq["peak_gpu_mem_mb"], 1)
            lat_ratio   = fp["e2e_latency_ms"]  / max(tq["e2e_latency_ms"], 1)
            tput_ratio  = tq["decode_throughput_tps"] / max(fp["decode_throughput_tps"], 1)
            print(
                f"  batch={bs}:  mem {mem_ratio:.2f}x  "
                f"e2e-lat {lat_ratio:.2f}x  throughput {tput_ratio:.2f}x  "
                f"(>1 = TurboQuant better)"
            )

    # Save JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
