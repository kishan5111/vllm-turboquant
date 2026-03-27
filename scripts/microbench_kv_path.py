#!/usr/bin/env python3
"""
Microbenchmark: hot KV write / decode attention path only.

Measures:
  - TurboQuant KV compress (write path)
  - TurboQuant KV decompress (read path)
  - TurboQuant fused paged decode attention

Sweeps:
  - context_len: 8192, 16384, 32768, 65536
  - num_seqs:    4, 8, 16, 64, 128, 256

Outputs: results/microbench.csv
"""
import argparse
import csv
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoConfig

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class ModelTopology:
    model: str
    head_size: int
    num_heads: int
    num_kv_heads: int
    block_size: int = 16

    @property
    def gqa_ratio(self) -> int:
        return self.num_heads // self.num_kv_heads


def topology_from_model(
    model: str,
    head_size: int | None = None,
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
    block_size: int = 16,
) -> ModelTopology:
    cfg = AutoConfig.from_pretrained(model)
    resolved_num_heads = num_heads or getattr(cfg, "num_attention_heads")
    resolved_num_kv_heads = num_kv_heads or getattr(
        cfg,
        "num_key_value_heads",
        resolved_num_heads,
    )
    resolved_head_size = head_size or getattr(
        cfg,
        "head_dim",
        getattr(cfg, "hidden_size") // resolved_num_heads,
    )
    return ModelTopology(
        model=model,
        head_size=resolved_head_size,
        num_heads=resolved_num_heads,
        num_kv_heads=resolved_num_kv_heads,
        block_size=block_size,
    )


def _warmup(fn, n=3):
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


def _bench(fn, n=20):
    """Return median wall-time in ms over n iterations."""
    _warmup(fn)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


# ------------------------------------------------------------------
# Helpers: allocate paged cache tensors
# ------------------------------------------------------------------

def _make_tq_cache(num_blocks: int, topology: ModelTopology):
    from vllm.v1.attention.backends.turboquant_attn import turboquant_comp_head_size
    comp = turboquant_comp_head_size(topology.head_size, 4)
    # shape: (2, num_blocks, num_kv_heads, block_size, comp_head)
    k = torch.zeros(
        num_blocks,
        topology.num_kv_heads,
        topology.block_size,
        comp,
        dtype=torch.uint8,
        device=DEVICE,
    )
    v = torch.zeros_like(k)
    return k, v


def _make_fp8_cache(num_blocks: int, topology: ModelTopology):
    # standard vLLM FP8 layout: (num_blocks, block_size, num_kv_heads, head_size)
    k = torch.zeros(
        num_blocks,
        topology.block_size,
        topology.num_kv_heads,
        topology.head_size,
        dtype=torch.uint8,
        device=DEVICE,
    )
    v = torch.zeros_like(k)
    return k, v


# ------------------------------------------------------------------
# Benchmark 1: TurboQuant KV compress (write path)
# ------------------------------------------------------------------

def bench_tq_compress(
    ctx_len: int,
    num_seqs: int,
    topology: ModelTopology,
    use_fused_rotation: bool = False,
) -> dict:
    from vllm.v1.attention.ops.triton_turboquant_kv import (
        make_turboquant_rotation, turboquant_compress_kv,
    )
    from vllm.v1.attention.backends.turboquant_attn import turboquant_comp_head_size
    comp = turboquant_comp_head_size(topology.head_size, 4)

    # Simulate ctx_len new tokens per sequence (prefill scenario)
    num_tokens = num_seqs * ctx_len
    num_blocks = (num_tokens + topology.block_size - 1) // topology.block_size + 16

    key = torch.randn(
        num_tokens,
        topology.num_kv_heads,
        topology.head_size,
        dtype=DTYPE,
        device=DEVICE,
    )
    val = torch.randn_like(key)
    R = make_turboquant_rotation(topology.head_size, DTYPE, DEVICE, seed=1)

    k_cache = torch.zeros(
        num_blocks,
        topology.num_kv_heads,
        topology.block_size,
        comp,
        dtype=torch.uint8,
        device=DEVICE,
    )
    v_cache = torch.zeros_like(k_cache)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=DEVICE)

    def fn():
        if use_fused_rotation:
            turboquant_compress_kv(
                key,
                val,
                k_cache,
                v_cache,
                slot_mapping,
                key_rotation=R,
                value_rotation=R,
            )
        else:
            turboquant_compress_kv(key @ R, val @ R, k_cache, v_cache, slot_mapping)

    ms = _bench(fn)
    tok_per_s = num_tokens / (ms / 1e3)
    ns_per_tok = ms * 1e6 / num_tokens
    return {"backend": "tq_compress_fused" if use_fused_rotation else "tq_compress",
            "ctx_len": ctx_len, "num_seqs": num_seqs,
            "metric": "ns_per_token", "value": round(ns_per_tok, 2),
            "tok_per_s": round(tok_per_s, 0), "latency_ms": round(ms, 3)}


# ------------------------------------------------------------------
# Benchmark 2: TurboQuant decompress (read path)
# ------------------------------------------------------------------

def bench_tq_decompress(ctx_len: int, num_seqs: int, topology: ModelTopology) -> dict:
    from vllm.v1.attention.ops.triton_turboquant_kv import (
        turboquant_decompress_blocks,
    )
    from vllm.v1.attention.backends.turboquant_attn import turboquant_comp_head_size
    comp = turboquant_comp_head_size(topology.head_size, 4)

    num_tokens = num_seqs * ctx_len
    num_blocks = (num_tokens + topology.block_size - 1) // topology.block_size
    # Fill cache with random compressed data
    k_cache = torch.randint(
        0, 256,
        (num_blocks, topology.num_kv_heads, topology.block_size, comp),
        dtype=torch.uint8,
        device=DEVICE,
    )
    v_cache = torch.randint_like(k_cache, 0, 256)

    block_ids = torch.arange(num_blocks, dtype=torch.long, device=DEVICE)
    out_k = torch.empty(
        num_blocks,
        topology.block_size,
        topology.num_kv_heads,
        topology.head_size,
        dtype=DTYPE,
        device=DEVICE,
    )
    out_v = torch.empty_like(out_k)

    def fn():
        turboquant_decompress_blocks(k_cache, block_ids, out_k)
        turboquant_decompress_blocks(v_cache, block_ids, out_v)

    ms = _bench(fn)
    total_tokens = num_blocks * topology.block_size
    ns_per_tok = ms * 1e6 / total_tokens
    return {"backend": "tq_decompress", "ctx_len": ctx_len, "num_seqs": num_seqs,
            "metric": "ns_per_token", "value": round(ns_per_tok, 2),
            "tok_per_s": round(total_tokens / (ms / 1e3), 0), "latency_ms": round(ms, 3)}


# ------------------------------------------------------------------
# Benchmark 3: TurboQuant fused paged decode
# ------------------------------------------------------------------

def bench_tq_fused_decode(
    ctx_len: int,
    num_seqs: int,
    topology: ModelTopology,
    rotate_inside_decode: bool = False,
    kv_splits: int | None = None,
    split_num_warps: int | None = None,
    split_num_stages: int | None = None,
    merge_num_warps: int | None = None,
) -> dict:
    from vllm.v1.attention.ops.triton_turboquant_kv import make_turboquant_rotation
    from vllm.v1.attention.ops.triton_turboquant_paged_attn import turboquant_fused_paged_decode
    from vllm.v1.attention.backends.turboquant_attn import turboquant_comp_head_size
    comp = turboquant_comp_head_size(topology.head_size, 4)

    num_blocks_per_seq = (ctx_len + topology.block_size - 1) // topology.block_size
    total_blocks = num_seqs * num_blocks_per_seq + 4

    k_cache = torch.randint(0, 256,
        (total_blocks, topology.num_kv_heads, topology.block_size, comp),
        dtype=torch.uint8,
        device=DEVICE)
    v_cache = torch.randint_like(k_cache, 0, 256)

    query = torch.randn(
        num_seqs,
        topology.num_heads,
        topology.head_size,
        dtype=DTYPE,
        device=DEVICE,
    )
    R = make_turboquant_rotation(topology.head_size, DTYPE, DEVICE, seed=1)

    # Build block table: each seq gets its own blocks
    block_table = torch.zeros(num_seqs, num_blocks_per_seq, dtype=torch.int32, device=DEVICE)
    for i in range(num_seqs):
        block_table[i] = torch.arange(i * num_blocks_per_seq,
                                      (i + 1) * num_blocks_per_seq,
                                      dtype=torch.int32, device=DEVICE)

    seq_lens = torch.full((num_seqs,), ctx_len, dtype=torch.int32, device=DEVICE)
    scale = topology.head_size ** -0.5

    def fn():
        turboquant_fused_paged_decode(
            query=query,
            key_cache=k_cache,
            value_cache=v_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            rotation=R,
            scale=scale,
            kv_splits=kv_splits,
            skip_output_inverse_rotation=True,
            rotate_inside_decode=rotate_inside_decode,
            split_num_warps=split_num_warps,
            split_num_stages=split_num_stages,
            merge_num_warps=merge_num_warps,
        )

    ms = _bench(fn)
    ns_per_tok = ms * 1e6 / num_seqs   # per output token (one decode step = one token per seq)
    backend = "tq_fused_decode_rot_in_kernel" if rotate_inside_decode else "tq_fused_decode"
    if any(v is not None for v in (kv_splits, split_num_warps, split_num_stages, merge_num_warps)):
        backend += "_tuned"
    return {"backend": backend,
            "ctx_len": ctx_len, "num_seqs": num_seqs,
            "metric": "ns_per_token", "value": round(ns_per_tok, 2),
            "tok_per_s": round(num_seqs / (ms / 1e3), 0), "latency_ms": round(ms, 3),
            "kv_splits": kv_splits,
            "split_num_warps": split_num_warps,
            "split_num_stages": split_num_stages,
            "merge_num_warps": merge_num_warps}


# ------------------------------------------------------------------
# Benchmark 4: compression ratio (static, no timing)
# ------------------------------------------------------------------

def compression_ratio(topology: ModelTopology) -> dict:
    from vllm.v1.attention.backends.turboquant_attn import turboquant_comp_head_size
    comp = turboquant_comp_head_size(topology.head_size, 4)
    bf16_bytes = topology.head_size * 2
    fp8_bytes  = topology.head_size
    tq_bytes   = comp            # 66
    return {
        "backend": "compression_ratio",
        "ctx_len": 0, "num_seqs": 0, "metric": "ratio_vs_bf16",
        "value": round(bf16_bytes / tq_bytes, 3),
        "tok_per_s": 0,
        "latency_ms": 0,
        "ratio_vs_fp8": round(fp8_bytes / tq_bytes, 3),
        "bytes_per_token_kv_head": tq_bytes,
    }


def profile_turboquant_paths(
    ctx_lens: list[int],
    num_seqs: list[int],
    topology: ModelTopology,
    include_fused_paths: bool = True,
) -> list[dict]:
    rows: list[dict] = [compression_ratio(topology) | asdict(topology)]
    for ctx_len in ctx_lens:
        for seq_count in num_seqs:
            est_blocks = (seq_count * ctx_len // topology.block_size) + seq_count
            est_bytes = (
                est_blocks
                * topology.block_size
                * topology.num_kv_heads
                * (topology.head_size // 2 + 2)
                * 2
            )
            if est_bytes > 60 * 1024**3:
                continue

            for bench_fn, kwargs in (
                (bench_tq_compress, {"use_fused_rotation": False}),
                (bench_tq_compress, {"use_fused_rotation": True}),
                (bench_tq_decompress, {}),
            ):
                row = bench_fn(ctx_len, seq_count, topology, **kwargs)
                rows.append(row | asdict(topology))

            if include_fused_paths:
                for rotate_inside_decode in (False, True):
                    row = bench_tq_fused_decode(
                        ctx_len,
                        seq_count,
                        topology,
                        rotate_inside_decode=rotate_inside_decode,
                    )
                    rows.append(row | asdict(topology))
            torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/microbench.csv")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--ctx-lens", nargs="+", type=int, default=[8192, 16384, 32768, 65536])
    parser.add_argument("--num-seqs", nargs="+", type=int, default=[4, 8, 16, 64, 128, 256])
    parser.add_argument("--head-size", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    topology = topology_from_model(
        args.model,
        head_size=args.head_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        block_size=args.block_size,
    )
    rows = []

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {topology.model}")
    print(f"Topology: heads={topology.num_heads} kv_heads={topology.num_kv_heads} "
          f"head_size={topology.head_size} block_size={topology.block_size}")
    print(f"Context lens: {args.ctx_lens}")
    print(f"Num seqs: {args.num_seqs}")
    print()

    # Static compression ratio (no sweep needed)
    r = compression_ratio(topology) | asdict(topology)
    rows.append(r)
    print(f"Compression ratio vs BF16: {r['value']:.3f}x  vs FP8: {r['ratio_vs_fp8']:.3f}x  ({r['bytes_per_token_kv_head']} bytes/token/kv-head)")
    print()

    for ctx_len in args.ctx_lens:
        for num_seqs in args.num_seqs:
            # Skip combinations that exceed GPU memory
            est_blocks = (num_seqs * ctx_len // topology.block_size) + num_seqs
            est_bytes = (
                est_blocks
                * topology.block_size
                * topology.num_kv_heads
                * (topology.head_size // 2 + 2)
                * 2
            )
            if est_bytes > 60 * 1024**3:  # 60 GB limit
                print(f"  SKIP ctx={ctx_len} seqs={num_seqs} (would need ~{est_bytes/1e9:.0f}GB)")
                continue

            print(f"  ctx={ctx_len:6d} seqs={num_seqs:3d}", end="  ", flush=True)

            try:
                r = bench_tq_compress(ctx_len, num_seqs, topology, use_fused_rotation=False) | asdict(topology)
                rows.append(r)
                print(f"compress={r['latency_ms']:.1f}ms ({r['tok_per_s']/1e6:.1f}M tok/s)", end="  ")
            except Exception as e:
                print(f"compress=ERR({e})", end="  ")

            try:
                r = bench_tq_decompress(ctx_len, num_seqs, topology) | asdict(topology)
                rows.append(r)
                print(f"decomp={r['latency_ms']:.1f}ms", end="  ")
            except Exception as e:
                print(f"decomp=ERR({e})", end="  ")

            try:
                r = bench_tq_fused_decode(ctx_len, num_seqs, topology, rotate_inside_decode=False) | asdict(topology)
                rows.append(r)
                print(f"fused_decode={r['latency_ms']:.2f}ms ({r['tok_per_s']:.0f} tok/s)", end="")
            except Exception as e:
                print(f"fused_decode=ERR({e})", end="")

            print()
            torch.cuda.empty_cache()

    if rows:
        fields = sorted({k for r in rows for k in r.keys()})
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved → {args.output}")
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
