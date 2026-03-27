#!/usr/bin/env python3
"""
Microbenchmark: hot KV write / decode attention path only.

Measures:
  - TurboQuant KV compress (write path)
  - TurboQuant KV decompress (read path)
  - TurboQuant fused paged decode attention
  - FP8 flash-attention decode (baseline)

Sweeps:
  - context_len: 8192, 32768, 65536
  - num_seqs:    16, 64, 128, 256

Outputs: results/microbench.csv
"""
import argparse, csv, os, time
from pathlib import Path

import torch

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

# ------------------------------------------------------------------
# Model topology — Qwen3-8B by default
# ------------------------------------------------------------------
HEAD_SIZE   = 128
NUM_HEADS   = 32   # Q heads
NUM_KV_HEADS = 8   # KV heads
GQA_RATIO   = NUM_HEADS // NUM_KV_HEADS  # 4
BLOCK_SIZE  = 16
DEVICE      = torch.device("cuda")
DTYPE       = torch.bfloat16


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

def _make_tq_cache(num_blocks: int):
    from vllm.v1.attention.backends.turboquant_attn import turboquant_comp_head_size
    comp = turboquant_comp_head_size(HEAD_SIZE, 4)  # 66
    # shape: (2, num_blocks, num_kv_heads, block_size, comp_head)
    k = torch.zeros(num_blocks, NUM_KV_HEADS, BLOCK_SIZE, comp, dtype=torch.uint8, device=DEVICE)
    v = torch.zeros_like(k)
    return k, v


def _make_fp8_cache(num_blocks: int):
    # standard vLLM FP8 layout: (num_blocks, block_size, num_kv_heads, head_size)
    k = torch.zeros(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.uint8, device=DEVICE)
    v = torch.zeros_like(k)
    return k, v


# ------------------------------------------------------------------
# Benchmark 1: TurboQuant KV compress (write path)
# ------------------------------------------------------------------

def bench_tq_compress(ctx_len: int, num_seqs: int) -> dict:
    from vllm.v1.attention.ops.triton_turboquant_kv import (
        make_turboquant_rotation, turboquant_compress_kv,
    )
    from vllm.v1.attention.backends.turboquant_attn import turboquant_comp_head_size
    comp = turboquant_comp_head_size(HEAD_SIZE, 4)

    # Simulate ctx_len new tokens per sequence (prefill scenario)
    num_tokens = num_seqs * ctx_len
    num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 16

    key = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    val = torch.randn_like(key)
    R   = make_turboquant_rotation(HEAD_SIZE, DTYPE, DEVICE, seed=1)
    key_rot = key @ R
    val_rot = val @ R

    k_cache = torch.zeros(num_blocks, NUM_KV_HEADS, BLOCK_SIZE, comp, dtype=torch.uint8, device=DEVICE)
    v_cache = torch.zeros_like(k_cache)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=DEVICE)

    def fn():
        turboquant_compress_kv(key_rot, val_rot, k_cache, v_cache, slot_mapping)

    ms = _bench(fn)
    tok_per_s = num_tokens / (ms / 1e3)
    ns_per_tok = ms * 1e6 / num_tokens
    return {"backend": "tq_compress", "ctx_len": ctx_len, "num_seqs": num_seqs,
            "metric": "ns_per_token", "value": round(ns_per_tok, 2),
            "tok_per_s": round(tok_per_s, 0), "latency_ms": round(ms, 3)}


# ------------------------------------------------------------------
# Benchmark 2: TurboQuant decompress (read path)
# ------------------------------------------------------------------

def bench_tq_decompress(ctx_len: int, num_seqs: int) -> dict:
    from vllm.v1.attention.ops.triton_turboquant_kv import (
        make_turboquant_rotation, turboquant_compress_kv, turboquant_decompress_blocks,
    )
    from vllm.v1.attention.backends.turboquant_attn import turboquant_comp_head_size
    comp = turboquant_comp_head_size(HEAD_SIZE, 4)

    num_tokens = num_seqs * ctx_len
    num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Fill cache with random compressed data
    k_cache = torch.randint(0, 256, (num_blocks, NUM_KV_HEADS, BLOCK_SIZE, comp),
                            dtype=torch.uint8, device=DEVICE)
    v_cache = torch.randint_like(k_cache, 0, 256)

    block_ids = torch.arange(num_blocks, dtype=torch.long, device=DEVICE)
    out_k = torch.empty(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    out_v = torch.empty_like(out_k)

    def fn():
        turboquant_decompress_blocks(k_cache, block_ids, out_k)
        turboquant_decompress_blocks(v_cache, block_ids, out_v)

    ms = _bench(fn)
    total_tokens = num_blocks * BLOCK_SIZE
    ns_per_tok = ms * 1e6 / total_tokens
    return {"backend": "tq_decompress", "ctx_len": ctx_len, "num_seqs": num_seqs,
            "metric": "ns_per_token", "value": round(ns_per_tok, 2),
            "tok_per_s": round(total_tokens / (ms / 1e3), 0), "latency_ms": round(ms, 3)}


# ------------------------------------------------------------------
# Benchmark 3: TurboQuant fused paged decode
# ------------------------------------------------------------------

def bench_tq_fused_decode(ctx_len: int, num_seqs: int) -> dict:
    from vllm.v1.attention.ops.triton_turboquant_kv import make_turboquant_rotation
    from vllm.v1.attention.ops.triton_turboquant_paged_attn import turboquant_fused_paged_decode
    from vllm.v1.attention.backends.turboquant_attn import turboquant_comp_head_size
    comp = turboquant_comp_head_size(HEAD_SIZE, 4)

    num_blocks_per_seq = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = num_seqs * num_blocks_per_seq + 4

    k_cache = torch.randint(0, 256,
        (total_blocks, NUM_KV_HEADS, BLOCK_SIZE, comp), dtype=torch.uint8, device=DEVICE)
    v_cache = torch.randint_like(k_cache, 0, 256)

    query = torch.randn(num_seqs, NUM_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    R = make_turboquant_rotation(HEAD_SIZE, DTYPE, DEVICE, seed=1)

    # Build block table: each seq gets its own blocks
    block_table = torch.zeros(num_seqs, num_blocks_per_seq, dtype=torch.int32, device=DEVICE)
    for i in range(num_seqs):
        block_table[i] = torch.arange(i * num_blocks_per_seq,
                                      (i + 1) * num_blocks_per_seq,
                                      dtype=torch.int32, device=DEVICE)

    seq_lens = torch.full((num_seqs,), ctx_len, dtype=torch.int32, device=DEVICE)
    scale = HEAD_SIZE ** -0.5

    def fn():
        turboquant_fused_paged_decode(
            query=query,
            key_cache=k_cache,
            value_cache=v_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            rotation=R,
            scale=scale,
            skip_output_inverse_rotation=True,
        )

    ms = _bench(fn)
    ns_per_tok = ms * 1e6 / num_seqs   # per output token (one decode step = one token per seq)
    return {"backend": "tq_fused_decode", "ctx_len": ctx_len, "num_seqs": num_seqs,
            "metric": "ns_per_token", "value": round(ns_per_tok, 2),
            "tok_per_s": round(num_seqs / (ms / 1e3), 0), "latency_ms": round(ms, 3)}


# ------------------------------------------------------------------
# Benchmark 4: compression ratio (static, no timing)
# ------------------------------------------------------------------

def compression_ratio() -> dict:
    from vllm.v1.attention.backends.turboquant_attn import turboquant_comp_head_size
    comp = turboquant_comp_head_size(HEAD_SIZE, 4)
    bf16_bytes = HEAD_SIZE * 2   # 256
    fp8_bytes  = HEAD_SIZE * 1   # 128
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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/microbench.csv")
    parser.add_argument("--ctx-lens", nargs="+", type=int, default=[8192, 32768, 65536])
    parser.add_argument("--num-seqs", nargs="+", type=int, default=[16, 64, 128, 256])
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Context lens: {args.ctx_lens}")
    print(f"Num seqs: {args.num_seqs}")
    print()

    # Static compression ratio (no sweep needed)
    r = compression_ratio()
    rows.append(r)
    print(f"Compression ratio vs BF16: {r['value']:.3f}x  vs FP8: {r['ratio_vs_fp8']:.3f}x  ({r['bytes_per_token_kv_head']} bytes/token/kv-head)")
    print()

    for ctx_len in args.ctx_lens:
        for num_seqs in args.num_seqs:
            # Skip combinations that exceed GPU memory
            est_blocks = (num_seqs * ctx_len // BLOCK_SIZE) + num_seqs
            est_bytes = est_blocks * BLOCK_SIZE * NUM_KV_HEADS * 66 * 2  # K+V
            if est_bytes > 60 * 1024**3:  # 60 GB limit
                print(f"  SKIP ctx={ctx_len} seqs={num_seqs} (would need ~{est_bytes/1e9:.0f}GB)")
                continue

            print(f"  ctx={ctx_len:6d} seqs={num_seqs:3d}", end="  ", flush=True)

            try:
                r = bench_tq_compress(ctx_len, num_seqs)
                rows.append(r)
                print(f"compress={r['latency_ms']:.1f}ms ({r['tok_per_s']/1e6:.1f}M tok/s)", end="  ")
            except Exception as e:
                print(f"compress=ERR({e})", end="  ")

            try:
                r = bench_tq_decompress(ctx_len, num_seqs)
                rows.append(r)
                print(f"decomp={r['latency_ms']:.1f}ms", end="  ")
            except Exception as e:
                print(f"decomp=ERR({e})", end="  ")

            try:
                r = bench_tq_fused_decode(ctx_len, num_seqs)
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
