#!/usr/bin/env python3
"""
Sweep aggregate throughput at fixed long contexts across request counts.

This is the direct proxy for serving-scale wins:
  - hold prompt/output length fixed
  - increase admitted request count
  - compare aggregate output tok/s for FP8 vs TurboQuant
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_bench import _resolve_cudagraph_mode, _run_workload  # noqa: E402


DEFAULT_SWEEPS = {
    32768: [2, 4, 8, 12, 16],
    65536: [1, 2, 4, 6, 8],
}


def _slugify_model_name(model: str) -> str:
    return model.split("/")[-1].replace(".", "-").replace("_", "-").lower()


def _parse_ctx_spec(spec: str) -> tuple[int, list[int]]:
    ctx_str, reqs_str = spec.split(":", maxsplit=1)
    ctx_len = int(ctx_str)
    req_counts = [int(part) for part in reqs_str.split(",") if part]
    return ctx_len, req_counts


def _kv_block_size(kv_dtype: str) -> int | None:
    if kv_dtype == "turboquant_qjl":
        return 32
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--kv-dtypes",
        nargs="+",
        default=["fp8", "turboquant_qjl"],
        choices=["fp8", "turboquant_4bit", "turboquant_qjl"],
    )
    parser.add_argument("--max-model-len", type=int, default=69632)
    parser.add_argument("--gpu-util", type=float, default=0.93)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument(
        "--ctx-spec",
        action="append",
        default=[],
        help="context_len:req1,req2,... e.g. 32768:2,4,8,12,16",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        model_tag = _slugify_model_name(args.model)
        args.output = f"results/{model_tag}_longctx_concurrency_sweep.json"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.ctx_spec:
        sweep_map = dict(_parse_ctx_spec(spec) for spec in args.ctx_spec)
    else:
        sweep_map = DEFAULT_SWEEPS

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_results: list[dict] = []

    for kv_dtype in args.kv_dtypes:
        resolved_cudagraph_mode = _resolve_cudagraph_mode("auto", kv_dtype)
        enable_chunked_prefill = kv_dtype != "turboquant_qjl"
        if kv_dtype == "turboquant_qjl":
            print("Using non-chunked prefill for turboquant_qjl (stability workaround).")
        print(f"\n{'='*60}")
        print(f"Backend  : {kv_dtype}")
        print(f"Model    : {args.model}")
        print(f"Max len  : {args.max_model_len}")
        print(f"GPU util : {args.gpu_util}")
        print(f"CUDAGraph: {resolved_cudagraph_mode}")
        print(f"{'='*60}")

        llm_kwargs = dict(
            model=args.model,
            kv_cache_dtype=kv_dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_util,
            disable_log_stats=False,
            enable_prefix_caching=False,
            enforce_eager=False,
            enable_chunked_prefill=enable_chunked_prefill,
            compilation_config={"cudagraph_mode": resolved_cudagraph_mode},
        )
        block_size = _kv_block_size(kv_dtype)
        if block_size is not None:
            llm_kwargs["block_size"] = block_size

        llm = LLM(**llm_kwargs)

        backend_rows: list[dict] = []
        for ctx_len, req_counts in sweep_map.items():
            for num_requests in req_counts:
                wl_name = f"ctx{ctx_len//1024}k_req{num_requests}"
                if ctx_len + args.output_len > args.max_model_len:
                    print(f"  [{wl_name:20s}] SKIPPED (input+output > max_model_len)")
                    continue
                try:
                    row = _run_workload(
                        llm,
                        tokenizer,
                        wl_name,
                        ctx_len,
                        args.output_len,
                        num_requests,
                        kv_dtype,
                    )
                    row["context_len"] = ctx_len
                    backend_rows.append(row)
                except Exception as exc:
                    print(f"  [{wl_name:20s}] ERROR: {exc}")
                    backend_rows.append(
                        {
                            "workload": wl_name,
                            "backend": kv_dtype,
                            "context_len": ctx_len,
                            "num_requests": num_requests,
                            "error": str(exc),
                        }
                    )
        all_results.extend(backend_rows)
        del llm

    with open(args.output, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "sweeps": sweep_map,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved -> {args.output}")


if __name__ == "__main__":
    main()
