#!/usr/bin/env python3
"""A/B check: TurboQuant fused decode vs MiniKV adapter entrypoint."""

from __future__ import annotations

import argparse
import time

import torch

from vllm.v1.attention.ops.triton_turboquant_paged_attn import (
    turboquant_fused_paged_decode,
)
from vllm.v1.attention.ops.turboquant_minikv_fused_attention import (
    turboquant_minikv_fused_decode,
)


def _run(fn, kwargs, iters: int) -> tuple[torch.Tensor, float]:
    for _ in range(3):
        _ = fn(**kwargs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn(**kwargs)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    assert out is not None
    return out, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    assert args.head_size % 2 == 0
    assert args.q_heads % args.kv_heads == 0
    assert args.ctx % args.block_size == 0

    device = "cuda"
    dtype = torch.bfloat16
    num_blocks = args.ctx // args.block_size
    comp_head = args.head_size // 2 + 2

    query = torch.randn(
        (args.seqs, args.q_heads, args.head_size), dtype=dtype, device=device
    )
    key_cache = torch.randint(
        0,
        256,
        (num_blocks, args.kv_heads, args.block_size, comp_head),
        dtype=torch.uint8,
        device=device,
    )
    value_cache = torch.randint(
        0,
        256,
        (num_blocks, args.kv_heads, args.block_size, comp_head),
        dtype=torch.uint8,
        device=device,
    )
    # Ensure scale bytes decode into finite fp16 values.
    scale_k = (
        torch.rand((num_blocks, args.kv_heads, args.block_size), device=device) * 1.5 + 0.25
    ).to(torch.float16)
    scale_v = (
        torch.rand((num_blocks, args.kv_heads, args.block_size), device=device) * 1.5 + 0.25
    ).to(torch.float16)
    scale_k_i32 = scale_k.view(torch.uint16).to(torch.int32)
    scale_v_i32 = scale_v.view(torch.uint16).to(torch.int32)
    key_cache[..., -2] = (scale_k_i32 & 0xFF).to(torch.uint8)
    key_cache[..., -1] = ((scale_k_i32 >> 8) & 0xFF).to(torch.uint8)
    value_cache[..., -2] = (scale_v_i32 & 0xFF).to(torch.uint8)
    value_cache[..., -1] = ((scale_v_i32 >> 8) & 0xFF).to(torch.uint8)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    block_table = block_table.expand(args.seqs, num_blocks).contiguous()
    seq_lens = torch.full((args.seqs,), args.ctx, dtype=torch.int32, device=device)
    rotation = torch.eye(args.head_size, dtype=dtype, device=device)
    scale = args.head_size ** -0.5

    common = dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        rotation=rotation,
        scale=scale,
        skip_output_inverse_rotation=False,
        rotate_inside_decode=True,
    )

    out_base, t_base = _run(turboquant_fused_paged_decode, common, args.iters)
    out_adpt, t_adpt = _run(turboquant_minikv_fused_decode, common, args.iters)

    max_diff = (out_base - out_adpt).abs().max().item()
    mean_diff = (out_base - out_adpt).abs().mean().item()

    print(f"base_ms={t_base * 1000:.3f}")
    print(f"adapter_ms={t_adpt * 1000:.3f}")
    print(f"speed_ratio(base/adapter)={t_base / t_adpt:.4f}")
    print(f"max_diff={max_diff:.6f}")
    print(f"mean_diff={mean_diff:.6f}")


if __name__ == "__main__":
    main()
