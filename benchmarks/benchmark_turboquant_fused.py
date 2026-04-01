#!/usr/bin/env python3
"""
Benchmark for the new TurboQuant Fused backend.

This tests the 0xSero-style FlatCache approach vs the existing
TurboQuant paged cache approach.

Usage:
    python benchmarks/benchmark_turboquant_fused.py
"""

import argparse
import gc
import json
import os
import time
import math

import torch

from vllm.v1.attention.backends.turboquant_fused_backend import (
    LayerConfig,
    LayerState,
    _create_layer_state,
    _compute_fused_attention,
    install_hooks,
    set_mode,
    get_mode,
    MODE_HYBRID,
    MODE_CAPTURE_ONLY,
    MODE_OFF,
)


def benchmark_fused_attention(
    num_tokens: int,
    num_kv_heads: int,
    num_query_heads: int,
    head_dim: int,
    num_decode_tokens: int,
    iters: int = 100,
) -> dict:
    """Benchmark the fused attention path."""
    device = torch.device("cuda")

    # Create layer state
    cfg = LayerConfig(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads,
        key_bits=3,
        value_bits=2,
        value_group_size=32,
        ring_capacity=128,
        layer_idx=0,
        device=device,
    )
    state = _create_layer_state(cfg)

    # Ingest prefill data
    prefill_k = torch.randn(num_tokens, num_kv_heads, head_dim, device=device)
    prefill_v = torch.randn(num_tokens, num_kv_heads, head_dim, device=device)
    state.engine.ingest_prefill(prefill_k, prefill_v, num_tokens)

    # Create decode query
    query = torch.randn(num_decode_tokens, num_query_heads, head_dim, device=device)
    scale = 1.0 / (head_dim ** 0.5)

    set_mode(MODE_HYBRID)

    # Warmup
    for _ in range(10):
        _ = _compute_fused_attention(
            query=query,
            store=state.store,
            engine=state.engine,
            Pi=state.Pi,
            S=state.S,
            centroids=state.centroids,
            qjl_scale=state.qjl_scale,
            scale=scale,
            gqa_ratio=num_query_heads // num_kv_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = _compute_fused_attention(
            query=query,
            store=state.store,
            engine=state.engine,
            Pi=state.Pi,
            S=state.S,
            centroids=state.centroids,
            qjl_scale=state.qjl_scale,
            scale=scale,
            gqa_ratio=num_query_heads // num_kv_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "num_tokens": num_tokens,
        "num_kv_heads": num_kv_heads,
        "num_query_heads": num_query_heads,
        "head_dim": head_dim,
        "gqa_ratio": num_query_heads // num_kv_heads,
        "iters": iters,
        "time_ms": elapsed * 1000,
        "time_per_iter_ms": elapsed / iters * 1000,
        "throughput_tps": num_decode_tokens * iters / elapsed,
        "output_valid": torch.isfinite(out).all().item(),
        "output_shape": list(out.shape),
    }


def benchmark_baseline_attention(
    num_tokens: int,
    num_kv_heads: int,
    num_query_heads: int,
    head_dim: int,
    num_decode_tokens: int,
    iters: int = 100,
) -> dict:
    """Benchmark standard PyTorch attention without compression."""
    import torch.nn.functional as F

    device = torch.device("cuda")

    # Create random KV cache
    kv_keys = torch.randn(num_kv_heads, num_tokens, head_dim, device=device)
    kv_values = torch.randn(num_kv_heads, num_tokens, head_dim, device=device)

    # Create decode query
    query = torch.randn(num_decode_tokens, num_query_heads, head_dim, device=device)
    scale = 1.0 / (head_dim ** 0.5)

    # Warmup
    for _ in range(10):
        q = query.unsqueeze(0).unsqueeze(0)  # Fake batch/dim for SDPA
        k = kv_keys.unsqueeze(0).unsqueeze(0).expand(num_decode_tokens, -1, -1, -1)
        v = kv_values.unsqueeze(0).unsqueeze(0).expand(num_decode_tokens, -1, -1, -1)
        _ = F.scaled_dot_product_attention(q, k, v, scale=scale)

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        q = query.unsqueeze(0).unsqueeze(0)
        k = kv_keys.unsqueeze(0).unsqueeze(0).expand(num_decode_tokens, -1, -1, -1)
        v = kv_values.unsqueeze(0).unsqueeze(0).expand(num_decode_tokens, -1, -1, -1)
        out = F.scaled_dot_product_attention(q, k, v, scale=scale)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "num_tokens": num_tokens,
        "iters": iters,
        "time_ms": elapsed * 1000,
        "time_per_iter_ms": elapsed / iters * 1000,
        "throughput_tps": num_decode_tokens * iters / elapsed,
        "output_valid": torch.isfinite(out).all().item(),
    }


def print_results(results: list[dict], title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'Config':<50} {'Time/it':<12} {'Throughput':<15}")
    print("-"*80)
    for r in results:
        config = f"tokens={r['num_tokens']}, H_kv={r['num_kv_heads']}, H_q={r['num_query_heads']}, dim={r['head_dim']}"
        print(f"{config:<50} {r['time_per_iter_ms']:>10.3f} ms {r['throughput_tops']:>12.1f} tok/s")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark TurboQuant Fused backend")
    parser.add_argument("--iters", type=int, default=100, help="Iterations per benchmark")
    parser.add_argument("--prefill-tokens", type=int, default=512, help="Number of prefill tokens")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results")
    args = parser.parse_args()

    print("="*80)
    print("  TurboQuant Fused Backend Benchmark")
    print("="*80)
    print(f"  Prefill tokens: {args.prefill_tokens}")
    print(f"  Iterations: {args.iters}")
    print(f"  Device: cuda")
    print()

    results = []

    # Test configurations
    configs = [
        # (num_kv_heads, num_query_heads, head_dim)
        (8, 32, 128),   # GPT-OSS style: GQA ratio 4
        (8, 64, 128),   # Higher GQA ratio
        (4, 32, 128),   # Fewer KV heads
    ]

    for num_kv_heads, num_query_heads, head_dim in configs:
        gqa_ratio = num_query_heads // num_kv_heads

        print(f"\n--- Config: {num_kv_heads} KV heads, {num_query_heads} Q heads, dim={head_dim} (GQA={gqa_ratio}) ---")

        # Benchmark TurboQuant Fused
        r = benchmark_fused_attention(
            num_tokens=args.prefill_tokens,
            num_kv_heads=num_kv_heads,
            num_query_heads=num_query_heads,
            head_dim=head_dim,
            num_decode_tokens=1,
            iters=args.iters,
        )
        r["type"] = "turboquant_fused"
        r["gqa_ratio"] = gqa_ratio
        r["throughput_tps"] = r["throughput_tps"]
        results.append(r)
        print(f"  TurboQuant Fused: {r['time_per_iter_ms']:.3f} ms/it, {r['throughput_tps']:.1f} tok/s")

        # Clear cache
        del r
        gc.collect()
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Type':<20} {'KV H':<6} {'Q H':<6} {'Dim':<6} {'GQA':<6} {'ms/it':<10} {'tok/s':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['type']:<20} {r['num_kv_heads']:<6} {r['num_query_heads']:<6} "
              f"{r['head_dim']:<6} {r['gqa_ratio']:<6} {r['time_per_iter_ms']:<10.3f} {r['throughput_tps']:<12.1f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()