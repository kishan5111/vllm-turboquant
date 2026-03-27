#!/usr/bin/env python3
"""
Profile GPT-OSS-20B workloads with vLLM's worker-side torch profiler.

This is intended for attribution, not headline throughput. It helps answer
whether wall time is still concentrated in attention/KV work or has shifted to
MoE routing / expert execution.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_bench import (  # noqa: E402
    WORKLOADS,
    _make_prompts,
    _resolve_cudagraph_mode,
    _slugify_model_name,
)


TIME_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(us|ms|s|ns)\s*$")


def _parse_time_to_us(value: str) -> float | None:
    match = TIME_RE.match(value)
    if not match:
        return None
    num = float(match.group(1))
    unit = match.group(2)
    if unit == "s":
        return num * 1_000_000
    if unit == "ms":
        return num * 1_000
    if unit == "us":
        return num
    if unit == "ns":
        return num / 1_000
    return None


def _split_table_line(line: str) -> list[str]:
    if "|" in line:
        return [part.strip() for part in line.split("|")[1:-1]]
    return [part.strip() for part in re.split(r"\s{2,}", line.strip()) if part.strip()]


def _parse_profiler_table(table_path: Path) -> list[dict]:
    rows: list[dict] = []
    lines = table_path.read_text().splitlines()
    header = None
    time_col = None

    for line in lines:
        stripped = line.strip()
        if not stripped or set(stripped) == {"-"}:
            continue
        parts = _split_table_line(line)
        if not parts:
            continue
        if parts[0] == "Name":
            header = parts
            for idx, name in enumerate(header):
                if name in ("Self CUDA", "Self CUDA total"):
                    time_col = idx
                    break
            continue
        if header is None or time_col is None:
            continue
        if len(parts) != len(header):
            continue
        time_us = _parse_time_to_us(parts[time_col])
        if time_us is None:
            continue
        row = dict(zip(header, parts))
        row["self_cuda_us"] = time_us
        rows.append(row)

    return rows


def _categorize_op(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("execute_context_") or name in {
        "Command Buffer Full",
        "Buffer Flush",
        "Activity Buffer Request",
        "Lazy Function Loading",
    }:
        return "wrapper_overhead"
    if any(
        token in lowered
        for token in (
            "moe",
            "expert",
            "topk",
            "gating",
            "grouped_gemm",
            "prepareandfinalize",
            "prepare_finalize",
            "fused_experts",
            "swiglu",
            "_matmul_ogs",
            "nvjet_tst",
            "router_gemm",
        )
    ):
        return "moe_ffn_routing"
    if any(
        token in lowered
        for token in (
            "attn",
            "attention",
            "flash_attn",
            "flashinfer",
            "paged",
            "kv_cache",
            "reshape_and_cache",
            "turboquant",
            "decode",
            "prefill",
            "cache",
            "rotat",
            "_tq_",
            "fa3",
            "unified_attention",
        )
    ):
        return "attention_kv"
    if any(token in lowered for token in ("rmsnorm", "layernorm", "norm")):
        return "norm"
    if any(token in lowered for token in ("embedding", "lm_head", "logits")):
        return "embedding_logits"
    if any(token in lowered for token in ("mm", "gemm", "matmul", "linear")):
        return "other_gemm"
    return "other"


def _summarize_rows(rows: list[dict], top_k: int) -> dict:
    filtered_rows = [
        row for row in rows if _categorize_op(row["Name"]) != "wrapper_overhead"
    ]
    total_us = sum(row["self_cuda_us"] for row in filtered_rows)
    by_category: dict[str, float] = {}
    for row in filtered_rows:
        category = _categorize_op(row["Name"])
        by_category[category] = by_category.get(category, 0.0) + row["self_cuda_us"]

    top_rows = sorted(filtered_rows, key=lambda row: row["self_cuda_us"], reverse=True)[:top_k]
    top_rows_json = [
        {
            "name": row["Name"],
            "self_cuda_us": round(row["self_cuda_us"], 3),
            "share_pct": round(
                100.0 * row["self_cuda_us"] / total_us, 2
            )
            if total_us
            else 0.0,
        }
        for row in top_rows
    ]
    categories_json = [
        {
            "category": category,
            "self_cuda_us": round(time_us, 3),
            "share_pct": round(100.0 * time_us / total_us, 2) if total_us else 0.0,
        }
        for category, time_us in sorted(
            by_category.items(), key=lambda item: item[1], reverse=True
        )
    ]
    return {
        "total_self_cuda_us": round(total_us, 3),
        "categories": categories_json,
        "top_ops": top_rows_json,
    }


def _print_summary(summary: dict) -> None:
    print("\nCategory summary by self CUDA time:")
    for item in summary["categories"]:
        print(
            f"  {item['category']:18s} {item['self_cuda_us']:12.1f} us"
            f"  ({item['share_pct']:5.1f}%)"
        )

    print("\nTop ops by self CUDA time:")
    for item in summary["top_ops"]:
        print(
            f"  {item['name'][:72]:72s} {item['self_cuda_us']:12.1f} us"
            f"  ({item['share_pct']:5.1f}%)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--kv-dtype", default="fp8")
    parser.add_argument("--workload", default="ctx8k", choices=sorted(WORKLOADS))
    parser.add_argument("--output-dir", default="results/profiles")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--cudagraph-mode", default="auto")
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--enable-layerwise-nvtx", action="store_true", default=False)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    workload = WORKLOADS[args.workload]
    resolved_cudagraph_mode = _resolve_cudagraph_mode(
        args.cudagraph_mode, args.kv_dtype
    )
    profile_tag = (
        f"{_slugify_model_name(args.model)}_{args.kv_dtype}_{args.workload}"
        f"{'_eager' if args.enforce_eager else ''}"
    )
    profile_dir = Path(args.output_dir) / profile_tag
    profile_dir.mkdir(parents=True, exist_ok=True)

    print(f"Profile dir : {profile_dir}")
    print(f"Backend     : {args.kv_dtype}")
    print(f"Workload    : {args.workload}")
    print(f"CUDAGraph   : {resolved_cudagraph_mode}")
    print(f"Eager       : {args.enforce_eager}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = _make_prompts(tokenizer, workload["input_len"], workload["num_requests"])
    warmup_prompts = _make_prompts(tokenizer, min(workload["input_len"], 512), 1)
    sampling_params = SamplingParams(
        max_tokens=workload["output_len"], temperature=0, ignore_eos=True
    )

    llm = LLM(
        model=args.model,
        kv_cache_dtype=args.kv_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=True,
        compilation_config={"cudagraph_mode": resolved_cudagraph_mode},
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir.resolve()),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_memory": False,
            "warmup_iterations": 0,
            "wait_iterations": 0,
            "active_iterations": 5,
            "ignore_frontend": True,
        },
        enable_layerwise_nvtx_tracing=args.enable_layerwise_nvtx,
    )

    llm.generate(warmup_prompts, SamplingParams(max_tokens=8, temperature=0))
    torch.cuda.synchronize()

    llm.start_profile(profile_prefix=profile_tag)
    llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    llm.stop_profile()

    profiler_table = profile_dir / "profiler_out_0.txt"
    if not profiler_table.exists():
        raise FileNotFoundError(
            f"Expected profiler output not found at {profiler_table}"
        )

    rows = _parse_profiler_table(profiler_table)
    summary = _summarize_rows(rows, args.top_k)
    summary.update(
        {
            "model": args.model,
            "kv_dtype": args.kv_dtype,
            "workload": args.workload,
            "resolved_cudagraph_mode": resolved_cudagraph_mode,
            "enforce_eager": args.enforce_eager,
            "profile_dir": str(profile_dir.resolve()),
            "profiler_table": str(profiler_table.resolve()),
        }
    )

    summary_path = profile_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary)
    print(f"\nSummary JSON: {summary_path}")
    print(f"Profiler table: {profiler_table}")


if __name__ == "__main__":
    main()
