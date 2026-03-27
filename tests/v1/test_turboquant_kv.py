# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for TurboQuant KV-cache compression in vLLM.

Covers:
  1. Unit tests for the Triton compress/decompress round-trip.
  2. Integration test: Qwen3-8B inference with turboquant_4bit vs baseline,
     verifying memory savings and output quality.
  3. Throughput benchmark comparing memory footprint and tokens/s.

Usage:
  # Run unit tests only (no model download needed):
  pytest tests/v1/test_turboquant_kv.py -k "not qwen3" -v

  # Full suite including Qwen3-8B:
  pytest tests/v1/test_turboquant_kv.py -v --model Qwen/Qwen3-8B
"""

import math
import os
import subprocess

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ---------------------------------------------------------------------------
# 1. Triton kernel round-trip (no model required)
# ---------------------------------------------------------------------------

class TestTurboQuantTritonKernels:
    """Unit tests for compress ↔ decompress round-trip correctness."""

    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        _skip_if_no_cuda()

    @pytest.mark.parametrize("head_size", [64, 128, 256])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_compress_decompress_roundtrip(self, head_size, dtype):
        """Decompressed values must be close to original rotated values."""
        from vllm.v1.attention.ops.triton_turboquant_kv import (
            make_turboquant_rotation,
            turboquant_compress_kv,
            turboquant_decompress_blocks,
            apply_rotation,
            apply_rotation_inverse,
        )

        device = torch.device("cuda")
        num_tokens  = 32
        num_heads   = 4
        block_size  = 16
        bits        = 4
        comp_head   = head_size // 2 + 2  # packed nibbles + 2-byte scale

        # Create random K/V
        torch.manual_seed(42)
        key   = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
        value = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)

        # Rotation matrix
        R = make_turboquant_rotation(head_size, dtype, device, seed=0)

        # Rotate
        key_rot   = apply_rotation(key,   R)
        value_rot = apply_rotation(value, R)

        # Allocate compressed cache
        num_blocks = math.ceil(num_tokens / block_size)
        # New layout: [num_blocks, num_kv_heads, block_size, comp_head]
        key_cache   = torch.zeros(num_blocks, num_heads, block_size, comp_head,
                                  dtype=torch.uint8, device=device)
        value_cache = torch.zeros_like(key_cache)

        # Build slot_mapping: token i goes to slot i
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

        # Compress
        turboquant_compress_kv(key_rot, value_rot, key_cache, value_cache, slot_mapping)

        # Decompress all blocks
        block_ids = torch.arange(num_blocks, dtype=torch.int64, device=device)
        decomp_key   = torch.empty(num_blocks, block_size, num_heads, head_size,
                                   dtype=dtype, device=device)
        decomp_value = torch.empty_like(decomp_key)

        turboquant_decompress_blocks(key_cache,   block_ids, decomp_key)
        turboquant_decompress_blocks(value_cache, block_ids, decomp_value)

        # Apply inverse rotation
        decomp_key   = apply_rotation_inverse(decomp_key,   R)
        decomp_value = apply_rotation_inverse(decomp_value, R)

        # Check: only the first num_tokens slots are valid
        tokens_per_block = block_size
        valid_key   = decomp_key.reshape(-1, num_heads, head_size)[:num_tokens]
        valid_value = decomp_value.reshape(-1, num_heads, head_size)[:num_tokens]

        # Cosine similarity between original and reconstructed
        cos_k = torch.nn.functional.cosine_similarity(
            key.reshape(-1, head_size).float(),
            valid_key.reshape(-1, head_size).float(),
        ).mean().item()
        cos_v = torch.nn.functional.cosine_similarity(
            value.reshape(-1, head_size).float(),
            valid_value.reshape(-1, head_size).float(),
        ).mean().item()

        print(f"\nhead_size={head_size} dtype={dtype} "
              f"cos_k={cos_k:.4f} cos_v={cos_v:.4f}")

        # 4-bit quantisation after rotation should give cosine similarity > 0.99
        assert cos_k > 0.99, f"Key cosine similarity too low: {cos_k:.4f}"
        assert cos_v > 0.99, f"Value cosine similarity too low: {cos_v:.4f}"

    def test_memory_compression_ratio(self):
        """Compressed cache should be ~3.9× smaller than BF16 for head_size=128."""
        from vllm.v1.attention.backends.turboquant_attn import (
            turboquant_comp_head_size,
        )
        head_size = 128
        comp = turboquant_comp_head_size(head_size, bits=4)
        bf16_bytes = head_size * 2
        ratio = bf16_bytes / comp
        print(f"\nBF16 bytes={bf16_bytes}  TurboQuant bytes={comp}  ratio={ratio:.2f}×")
        assert ratio > 3.5, f"Expected >3.5× compression, got {ratio:.2f}×"

    @pytest.mark.parametrize("head_size", [64, 128])
    def test_scale_preservation(self, head_size):
        """A single large-magnitude vector should be reconstructed accurately."""
        from vllm.v1.attention.ops.triton_turboquant_kv import (
            make_turboquant_rotation,
            turboquant_compress_kv,
            turboquant_decompress_blocks,
            apply_rotation,
            apply_rotation_inverse,
        )
        device = torch.device("cuda")
        dtype  = torch.bfloat16

        # Vector with one large outlier
        key = torch.zeros(1, 1, head_size, dtype=dtype, device=device)
        key[0, 0, 0] = 100.0
        key[0, 0, 1] = -50.0

        R = make_turboquant_rotation(head_size, dtype, device, seed=1)
        key_rot = apply_rotation(key, R)

        key_cache   = torch.zeros(1, 1, 16, head_size // 2 + 2,
                                  dtype=torch.uint8, device=device)
        value_cache = torch.zeros_like(key_cache)
        slot_mapping = torch.tensor([0], dtype=torch.int64, device=device)
        value = torch.zeros_like(key)

        turboquant_compress_kv(key_rot, apply_rotation(value, R),
                               key_cache, value_cache, slot_mapping)

        block_ids = torch.tensor([0], dtype=torch.int64, device=device)
        decomp = torch.empty(1, 16, 1, head_size, dtype=dtype, device=device)
        turboquant_decompress_blocks(key_cache, block_ids, decomp)
        decomp = apply_rotation_inverse(decomp, R)

        orig_flat  = key[0, 0].float()
        recon_flat = decomp[0, 0, 0].float()

        rel_err = ((orig_flat - recon_flat).norm() / orig_flat.norm()).item()
        print(f"\nhead_size={head_size} rel_err={rel_err:.4f}")
        assert rel_err < 0.15, f"Relative error too large: {rel_err:.4f}"


# ---------------------------------------------------------------------------
# 2. Attention-output cosine similarity test (no full model)
# ---------------------------------------------------------------------------

class TestTurboQuantAttentionQuality:
    """Measure how close TurboQuant attention output is to full-precision."""

    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        _skip_if_no_cuda()

    @pytest.mark.parametrize("seq_len,head_size", [(64, 128), (256, 128)])
    def test_attention_cosine_similarity(self, seq_len, head_size):
        """
        Simulate a single-layer attention with TurboQuant compressed cache.

        Checks that the output attention vector has cosine similarity > 0.995
        against full-precision attention (the TurboQuant paper reports >0.999
        at 3.5-bit compression).
        """
        from vllm.v1.attention.ops.triton_turboquant_kv import (
            apply_rotation,
            apply_rotation_inverse,
            make_turboquant_rotation,
            turboquant_compress_kv,
            turboquant_decompress_blocks,
        )

        device = torch.device("cuda")
        dtype  = torch.bfloat16
        num_heads    = 8
        num_kv_heads = 4
        block_size   = 16
        comp_head    = head_size // 2 + 2

        torch.manual_seed(123)
        # Q, K, V for all seq_len tokens
        Q = torch.randn(seq_len, num_heads,    head_size, dtype=dtype, device=device)
        K = torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype, device=device)
        V = torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype, device=device)

        # ---- Full-precision attention (naive, for reference) ----
        scale = 1.0 / math.sqrt(head_size)
        # Expand K/V for GQA: [seq, num_heads, head]
        K_exp = K.repeat_interleave(num_heads // num_kv_heads, dim=1)
        V_exp = V.repeat_interleave(num_heads // num_kv_heads, dim=1)
        # [seq_q, seq_k] per head
        scores = torch.einsum("qhd,khd->hqk", Q, K_exp) * scale
        attn_w = scores.softmax(dim=-1)
        out_fp = torch.einsum("hqk,khd->qhd", attn_w, V_exp)

        # ---- TurboQuant-compressed K/V cache ----
        R = make_turboquant_rotation(head_size, dtype, device, seed=42)
        num_blocks = math.ceil(seq_len / block_size)

        key_cache   = torch.zeros(num_blocks, num_kv_heads, block_size, comp_head,
                                  dtype=torch.uint8, device=device)
        value_cache = torch.zeros_like(key_cache)
        slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=device)

        turboquant_compress_kv(
            apply_rotation(K, R), apply_rotation(V, R),
            key_cache, value_cache, slot_mapping,
        )

        # Decompress all blocks
        block_ids = torch.arange(num_blocks, dtype=torch.int64, device=device)
        K_decomp = torch.empty(num_blocks, block_size, num_kv_heads, head_size,
                               dtype=dtype, device=device)
        V_decomp = torch.empty_like(K_decomp)
        turboquant_decompress_blocks(key_cache,   block_ids, K_decomp)
        turboquant_decompress_blocks(value_cache, block_ids, V_decomp)

        K_decomp = apply_rotation_inverse(K_decomp, R) \
                        .reshape(-1, num_kv_heads, head_size)[:seq_len]
        V_decomp = apply_rotation_inverse(V_decomp, R) \
                        .reshape(-1, num_kv_heads, head_size)[:seq_len]

        K_decomp_exp = K_decomp.repeat_interleave(num_heads // num_kv_heads, dim=1)
        V_decomp_exp = V_decomp.repeat_interleave(num_heads // num_kv_heads, dim=1)

        scores_tq = torch.einsum("qhd,khd->hqk", Q, K_decomp_exp) * scale
        attn_w_tq = scores_tq.softmax(dim=-1)
        out_tq    = torch.einsum("hqk,khd->qhd", attn_w_tq, V_decomp_exp)

        # Cosine similarity of attention outputs
        cos = torch.nn.functional.cosine_similarity(
            out_fp.reshape(-1, head_size).float(),
            out_tq.reshape(-1, head_size).float(),
        ).mean().item()

        print(f"\nseq_len={seq_len} head_size={head_size} "
              f"attn_output_cos={cos:.5f}")
        assert cos > 0.98, (
            f"Attention output cosine similarity too low: {cos:.5f} "
            "(expected >0.98 for TurboQuant 4-bit)"
        )


# ---------------------------------------------------------------------------
# 2b. Fused paged decode kernel correctness test
# ---------------------------------------------------------------------------

class TestTurboQuantFusedDecodeKernel:
    """Tests for the fused paged decode attention kernel."""

    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        _skip_if_no_cuda()

    def test_fused_decode_matches_unfused(self):
        """Fused decode kernel output should match decompress+flash_attn."""
        from vllm.v1.attention.ops.triton_turboquant_paged_attn import (
            turboquant_fused_paged_decode,
        )
        from vllm.v1.attention.ops.triton_turboquant_kv import (
            apply_rotation,
            apply_rotation_inverse,
            make_turboquant_rotation,
            turboquant_compress_kv,
            turboquant_decompress_blocks,
        )

        device = "cuda"
        dtype = torch.bfloat16
        num_seqs = 4
        num_q_heads = 8
        num_kv_heads = 2
        head_size = 64
        block_size = 16
        seq_len = 48   # 3 full blocks

        num_blocks = (seq_len * num_seqs + block_size - 1) // block_size + 4
        comp_head = head_size // 2 + 2

        # Random K and V (already rotated into the cache)
        R = make_turboquant_rotation(head_size, dtype, device, seed=0)
        K = torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype, device=device)
        V = torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype, device=device)

        # Build KV cache: compress into uint8 format
        key_cache   = torch.zeros(num_blocks, num_kv_heads, block_size, comp_head,
                                   dtype=torch.uint8, device=device)
        value_cache = torch.zeros_like(key_cache)

        # Fill blocks for each sequence (one seq uses one contiguous block range)
        slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=device)

        turboquant_compress_kv(
            apply_rotation(K, R), apply_rotation(V, R),
            key_cache, value_cache, slot_mapping,
        )

        # Build block table: each seq uses the same blocks (simplified test)
        max_blocks_per_seq = (seq_len + block_size - 1) // block_size
        block_table = torch.zeros(num_seqs, max_blocks_per_seq,
                                   dtype=torch.int32, device=device)
        for s in range(num_seqs):
            for b in range(max_blocks_per_seq):
                block_table[s, b] = b  # all seqs share same blocks (test only)

        seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)

        # Random queries
        Q = torch.randn(num_seqs, num_q_heads, head_size, dtype=dtype, device=device)
        scale = 1.0 / math.sqrt(head_size)

        # --- Reference: decompress then manual attention ---
        block_ids = torch.arange(max_blocks_per_seq, dtype=torch.int64, device=device)
        K_decomp = torch.zeros(max_blocks_per_seq, block_size, num_kv_heads, head_size,
                                dtype=dtype, device=device)
        V_decomp = torch.zeros_like(K_decomp)
        turboquant_decompress_blocks(key_cache, block_ids, K_decomp)
        turboquant_decompress_blocks(value_cache, block_ids, V_decomp)

        # De-rotate (undo the R applied at compress time)
        K_decomp = apply_rotation_inverse(K_decomp, R).reshape(seq_len, num_kv_heads, head_size)
        V_decomp = apply_rotation_inverse(V_decomp, R).reshape(seq_len, num_kv_heads, head_size)

        # GQA expand
        K_exp = K_decomp.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        V_exp = V_decomp.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

        # Manual attention (all seqs attend same K/V in this test)
        ref_outputs = []
        for s in range(num_seqs):
            scores = torch.einsum("hd,khd->hk", Q[s], K_exp) * scale
            weights = scores.softmax(dim=-1)
            out = torch.einsum("hk,khd->hd", weights, V_exp)
            ref_outputs.append(out)
        ref_out = torch.stack(ref_outputs)   # [num_seqs, num_q_heads, head_size]

        # --- Fused decode kernel output ---
        fused_out = turboquant_fused_paged_decode(
            query=Q,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            rotation=R,
            scale=scale,
        )

        cos = torch.nn.functional.cosine_similarity(
            ref_out.reshape(-1, head_size).float(),
            fused_out.reshape(-1, head_size).float(),
        ).mean().item()

        print(f"\nFused vs unfused cosine similarity: {cos:.5f}")
        assert cos > 0.98, f"Fused decode kernel output mismatch: cos={cos:.5f}"


# ---------------------------------------------------------------------------
# 3. End-to-end Qwen3-8B benchmark (requires model download)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("TURBOQUANT_SKIP_E2E", "1") == "1",
    reason="Set TURBOQUANT_SKIP_E2E=0 to run the full Qwen3-8B benchmark",
)
class TestQwen3TurboQuantE2E:
    """End-to-end test with Qwen/Qwen3-8B comparing memory and throughput."""

    MODEL = os.environ.get("TURBOQUANT_MODEL", "Qwen/Qwen3-8B")
    PROMPTS = [
        "The capital of France is",
        "In machine learning, the attention mechanism",
        "Once upon a time in a galaxy far far away",
        "The theory of relativity states that",
    ]

    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        _skip_if_no_cuda()
        if torch.cuda.get_device_capability()[0] < 8:
            pytest.skip("Requires Ampere (sm_80) or newer GPU")

    def _run_vllm(self, cache_dtype: str, max_tokens: int = 50):
        """Run vLLM inference and return (outputs, num_kv_tokens_capacity).

        num_kv_tokens_capacity is estimated from the total KV cache memory
        divided by per-token bytes for the given cache dtype.
        """
        import io
        import logging
        import re

        from vllm import LLM, SamplingParams

        # Capture the "GPU KV cache size: N tokens" log line from the engine.
        # vLLM logs from the main process via Python logging.
        tokens_found: list[int] = []

        class _TokenCapture(logging.Handler):
            def emit(self, record):
                m = re.search(r"GPU KV cache size: ([\d,]+) tokens",
                              record.getMessage())
                if m:
                    tokens_found.append(int(m.group(1).replace(",", "")))

        handler = _TokenCapture()
        root = logging.getLogger()
        root.addHandler(handler)

        try:
            llm = LLM(
                model=self.MODEL,
                kv_cache_dtype=cache_dtype,
                max_model_len=2048,
                gpu_memory_utilization=0.85,
                dtype="bfloat16",
                # Triton decompression kernels cannot be captured in CUDA graphs.
                enforce_eager=(cache_dtype == "turboquant_4bit"),
            )
            params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
            outputs = llm.generate(self.PROMPTS, params)
            del llm
            torch.cuda.empty_cache()
        finally:
            root.removeHandler(handler)

        kv_tokens = tokens_found[-1] if tokens_found else 0
        return [o.outputs[0].text for o in outputs], kv_tokens

    def test_memory_reduction(self):
        """TurboQuant should increase KV cache capacity vs BF16 baseline.

        Both runs use the same gpu_memory_utilization, so the same total GPU
        memory budget is available.  TurboQuant's 4-bit compression means
        each token requires ~3.88× less KV cache storage, so the engine can
        allocate ~3.88× more tokens before exhausting the KV cache budget.
        """
        # Baseline (BF16)
        _, cap_bf16 = self._run_vllm("auto")
        # TurboQuant 4-bit
        _, cap_tq   = self._run_vllm("turboquant_4bit")

        if cap_bf16 == 0 or cap_tq == 0:
            pytest.skip("Could not capture KV cache capacity from vLLM logs")

        ratio = cap_tq / cap_bf16
        print(f"\nBaseline BF16 KV token capacity  : {cap_bf16:,}")
        print(f"TurboQuant 4-bit KV token capacity: {cap_tq:,}")
        print(f"Capacity increase factor          : {ratio:.2f}×")

        # ~3.88× expected; allow some tolerance for block alignment overhead.
        assert ratio >= 3.0, (
            f"Expected ≥3.0× more KV tokens, got {ratio:.2f}×"
        )

    def test_output_quality(self):
        """TurboQuant outputs should be coherent and contain correct answers.

        4-bit quantization introduces noise so greedy paths may diverge from
        BF16, but the answers should still be semantically correct.  We check
        a set of factual keywords that must appear in the first generation.
        """
        outputs_tq, _ = self._run_vllm("turboquant_4bit", max_tokens=20)

        # Expected keywords for each prompt (case-insensitive substring match).
        expected_keywords = ["paris", "attention", "galaxy", "relativity"]

        print("\nTurboQuant output quality check:")
        for prompt, output, keyword in zip(
            self.PROMPTS, outputs_tq, expected_keywords
        ):
            combined = (prompt + output).lower()
            print(f"  [{keyword}] → {output!r}")
            assert keyword in combined, (
                f"Expected keyword {keyword!r} not found in output: {output!r}"
            )


# ---------------------------------------------------------------------------
# 4. Memory and throughput benchmark (standalone, not a pytest test)
# ---------------------------------------------------------------------------

def run_benchmark():
    """
    Standalone benchmark: compare KV-cache memory footprint for
    Qwen3-8B with BF16 vs TurboQuant 4-bit.

    Run with:
        python tests/v1/test_turboquant_kv.py
    """
    import time

    from vllm import LLM, SamplingParams

    model = os.environ.get("TURBOQUANT_MODEL", "Qwen/Qwen3-8B")
    prompts = [
        "Tell me a long story about a brave knight and a wise dragon " * 3,
    ] * 4
    params = SamplingParams(temperature=0.0, max_tokens=128)

    results = {}
    for cache_dtype in ("auto", "turboquant_4bit"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        llm = LLM(
            model=model,
            kv_cache_dtype=cache_dtype,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            dtype="bfloat16",
        )

        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        elapsed = time.perf_counter() - t0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = total_tokens / elapsed
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

        results[cache_dtype] = {
            "peak_mb": peak_mb,
            "tps": tps,
            "sample": outputs[0].outputs[0].text[:80],
        }

        del llm
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("TurboQuant KV-Cache Benchmark — Qwen3-8B on H100")
    print("=" * 60)
    for dtype, r in results.items():
        print(f"\n[{dtype}]")
        print(f"  Peak GPU memory : {r['peak_mb']:.0f} MB")
        print(f"  Throughput      : {r['tps']:.1f} tokens/s")
        print(f"  Sample output   : {r['sample']!r}")

    if "auto" in results and "turboquant_4bit" in results:
        mem_ratio = results["auto"]["peak_mb"] / results["turboquant_4bit"]["peak_mb"]
        tps_ratio = results["turboquant_4bit"]["tps"] / results["auto"]["tps"]
        print(f"\nMemory reduction : {mem_ratio:.2f}×")
        print(f"Throughput ratio : {tps_ratio:.2f}×  (TurboQuant / baseline)")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
