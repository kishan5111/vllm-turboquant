# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import time

import torch

from vllm.v1.attention.ops.qjl_decode_proto import (
    decode_paged_qjl,
    make_qjl_projection,
    pack_paged_qjl_cache,
    pack_qjl_prefix_tail_cache,
)


def exact_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch, q_heads, head_size = query.shape
    _, kv_heads, num_blocks, block_size, _ = key_cache.shape
    q_per_kv = q_heads // kv_heads

    keys = key_cache.reshape(batch, kv_heads, num_blocks * block_size, head_size)
    values = value_cache.reshape(batch, kv_heads, num_blocks * block_size, head_size)
    keys = keys.repeat_interleave(q_per_kv, dim=1)
    values = values.repeat_interleave(q_per_kv, dim=1)

    scores = torch.einsum("bhd,bhkd->bhk", query, keys) * scale
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.einsum("bhk,bhkd->bhd", probs, values)


def benchmark_once(
    *,
    batch_size: int,
    ctx_len: int,
    q_heads: int,
    kv_heads: int,
    head_size: int,
    sketch_dim: int,
    block_size: int,
    tail_tokens: int,
    warmup: int,
    iters: int,
) -> None:
    assert ctx_len % block_size == 0
    num_blocks = ctx_len // block_size
    scale = head_size**-0.5
    device = torch.device("cuda")
    dtype = torch.bfloat16

    query = torch.randn(batch_size, q_heads, head_size, device=device, dtype=dtype)
    key_cache = torch.randn(
        batch_size, kv_heads, num_blocks, block_size, head_size,
        device=device, dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)
    block_table = block_table.unsqueeze(0).expand(batch_size, -1).contiguous()
    seq_lens = torch.full(
        (batch_size,), ctx_len, device=device, dtype=torch.int32
    )

    proj_dir_score, proj_dir_quant = make_qjl_projection(
        head_size,
        sketch_dim,
        dtype=torch.float32,
        device=device,
        seed=0,
    )
    if tail_tokens > 0:
        qjl_cache = pack_qjl_prefix_tail_cache(
            key_cache,
            value_cache,
            proj_dir_quant=proj_dir_quant,
            tail_tokens=tail_tokens,
        )
    else:
        qjl_cache = pack_paged_qjl_cache(
            key_cache,
            value_cache,
            proj_dir_quant=proj_dir_quant,
        )

    ref = exact_decode(query, key_cache, value_cache, scale)
    out = decode_paged_qjl(
        query,
        qjl_cache=qjl_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        proj_dir_score=proj_dir_score,
        scale=scale,
    )
    cos = torch.nn.functional.cosine_similarity(
        ref.reshape(batch_size, -1).float(),
        out.reshape(batch_size, -1).float(),
        dim=-1,
    ).mean()

    for _ in range(warmup):
        decode_paged_qjl(
            query,
            qjl_cache=qjl_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            proj_dir_score=proj_dir_score,
            scale=scale,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        decode_paged_qjl(
            query,
            qjl_cache=qjl_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            proj_dir_score=proj_dir_score,
            scale=scale,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = elapsed / iters * 1000
    toks_per_s = batch_size / (elapsed / iters)

    print(
        f"bs={batch_size:2d} ctx={ctx_len:5d}  "
        f"{ms:7.3f} ms/token/layer  {toks_per_s:8.1f} tok/s/layer  "
        f"cos={cos.item():.4f}  block={block_size} tail={tail_tokens}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--ctx-lens", nargs="+", type=int, default=[8192, 16384])
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--sketch-dim", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tail-tokens", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    for batch_size in args.batch_sizes:
        for ctx_len in args.ctx_lens:
            benchmark_once(
                batch_size=batch_size,
                ctx_len=ctx_len,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                head_size=args.head_size,
                sketch_dim=args.sketch_dim,
                block_size=args.block_size,
                tail_tokens=args.tail_tokens,
                warmup=args.warmup,
                iters=args.iters,
            )


if __name__ == "__main__":
    main()
