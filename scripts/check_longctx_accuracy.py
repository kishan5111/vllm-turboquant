#!/usr/bin/env python3
"""
Compare greedy token agreement between FP8 and TurboQuant on long-context prompts.
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

from scripts.run_bench import _resolve_cudagraph_mode  # noqa: E402


def _make_prompt(tokenizer, input_len: int) -> str:
    filler = (
        "TurboQuant compresses the KV cache while preserving useful attention "
        "behavior over long contexts. "
    )
    closing = (
        "After reading the repeated background, continue with one short factual "
        "sentence about efficient long-context inference."
    )
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    closing_ids = tokenizer.encode(closing, add_special_tokens=False)
    usable = max(0, input_len - len(closing_ids))
    repeated = (filler_ids * ((usable + len(filler_ids) - 1) // len(filler_ids)))[:usable]
    prompt_ids = repeated + closing_ids
    return tokenizer.decode(prompt_ids)


def _run_backend(model: str, kv_dtype: str, prompt: str, max_model_len: int,
                 gpu_util: float, max_tokens: int) -> tuple[list[int], str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        kv_cache_dtype=kv_dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        disable_log_stats=False,
        enable_prefix_caching=False,
        enforce_eager=False,
        enable_chunked_prefill=True,
        compilation_config={"cudagraph_mode": _resolve_cudagraph_mode("auto", kv_dtype)},
    )
    out = llm.generate(
        [prompt],
        SamplingParams(max_tokens=max_tokens, temperature=0, ignore_eos=True),
    )[0]
    token_ids = list(out.outputs[0].token_ids)
    text = out.outputs[0].text
    del llm
    return token_ids, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768, 65536])
    parser.add_argument("--max-model-len", type=int, default=69632)
    parser.add_argument("--gpu-util", type=float, default=0.93)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output", default="results/gpt-oss-20b_longctx_accuracy.json")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows: list[dict] = []

    for ctx_len in args.contexts:
        prompt = _make_prompt(tokenizer, ctx_len)
        print(f"\nChecking ctx={ctx_len}...")
        fp8_ids, fp8_text = _run_backend(
            args.model, "fp8", prompt, args.max_model_len, args.gpu_util, args.max_tokens
        )
        tq_ids, tq_text = _run_backend(
            args.model, "turboquant_4bit", prompt, args.max_model_len, args.gpu_util, args.max_tokens
        )
        prefix_match = 0
        for a, b in zip(fp8_ids, tq_ids):
            if a != b:
                break
            prefix_match += 1
        row = {
            "context_len": ctx_len,
            "max_tokens": args.max_tokens,
            "fp8_token_ids": fp8_ids,
            "turboquant_token_ids": tq_ids,
            "prefix_match_tokens": prefix_match,
            "all_match": fp8_ids == tq_ids,
            "fp8_text": fp8_text,
            "turboquant_text": tq_text,
        }
        rows.append(row)
        print(
            f"  prefix_match={prefix_match}/{args.max_tokens} "
            f"all_match={row['all_match']}"
        )

    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "results": rows}, f, indent=2)
    print(f"\nResults saved -> {args.output}")


if __name__ == "__main__":
    main()
