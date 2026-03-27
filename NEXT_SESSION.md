# Next Session Instructions

## Branch: `turboquant-kv`
Repo: https://github.com/kishan5111/vllm-turboquant.git

```bash
git clone https://github.com/kishan5111/vllm-turboquant.git
cd vllm-turboquant
git checkout turboquant-kv
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

---

## What was built

TurboQuant 4-bit KV cache for vLLM — a PolarQuant-style scheme:
- Random orthogonal rotation R ∈ ℝ^{D×D} per layer, folded into V/output projections at model load
- 4-bit L∞ uniform quantisation, AoS layout, 66 bytes/token/KV-head (vs 256 BF16, 128 FP8)
- Fused decode Triton kernel reads compressed cache directly (no full KV materialisation)
- `--kv-cache-dtype turboquant_4bit` flag

**Key files:**
- `vllm/v1/attention/backends/turboquant_attn.py` — attention backend + weight folding
- `vllm/v1/attention/ops/triton_turboquant_kv.py` — compress/decompress kernels
- `vllm/v1/attention/ops/triton_turboquant_paged_attn.py` — fused decode Triton kernel
- `SUMMARY.md` — full results and architecture doc

---

## Results so far

### Dense model (Qwen3-8B) ✅ — promotion gate PASSED
| Workload | FP8 tok/s | TurboQuant tok/s | vs FP8 |
|----------|----------:|-----------------:|--------|
| high_context (8k input) | 227.5 | **596.2** | **2.62×** |
| concurrency_16 | 1,270.5 | 787.8 | 0.62× |
| concurrency_64 | 1,973.9 | 1,300.3 | 0.66× |

### MoE model (GPT-OSS-20B)
| Workload | FP8 tok/s | TurboQuant tok/s | vs FP8 |
|----------|----------:|-----------------:|--------|
| high_context (8k input) | 618.0 | 371.0 | 0.60× |

MoE gap is expected: FFN/expert routing dominates ~70% of compute, KV savings don't move the needle.

---

## What to do next (in priority order)

### 1. Fuse Q@R and K@R rotation into Triton kernel (HIGH)
**Problem**: Q@R and K@R rotation GEMMs run as eager PyTorch outside CUDA graphs per decode step — this is the primary bottleneck for dense models at short context (~35% slower than FP8).

**Fix**: Modify `turboquant_fused_paged_decode` in `triton_turboquant_paged_attn.py` to accept pre-rotation Q/K tensors and fold the rotation into the Triton kernel's internal tiled GEMM (FlashAttention-style). The rotation is just a small [head_size, head_size] matmul that can be fused into the attention kernel's first tile.

Alternatively: modify `do_kv_cache_update` in `turboquant_attn.py` to fuse the K@R rotation into the compress kernel (same idea — combine rotation + pack into one Triton kernel).

### 2. Verify GPT-OSS-20B output correctness
Run greedy decode with FP8 vs TurboQuant on the same prompt for first 16 decode steps — confirm top-1 token match ≥ 90%.

### 3. Test GPT-OSS-120B if VRAM allows
80GB H100 may be tight. Try:
```bash
HF_HOME=/workspace/.hf_home .venv/bin/python scripts/run_bench.py \
    --model AI-Growth-Turbo/TurboQuant-GPT-OSS-120B \
    --kv-dtype fp8 --output results/fp8_gptoss120b.json

HF_HOME=/workspace/.hf_home .venv/bin/python scripts/run_bench.py \
    --model AI-Growth-Turbo/TurboQuant-GPT-OSS-120B \
    --kv-dtype turboquant_4bit --output results/tq_gptoss120b.json
```

### 4. Fused compress kernel (MEDIUM)
Combine `apply_rotation` + `turboquant_pack_kernel` into a single Triton kernel to eliminate the intermediate HBM write of rotated K. Expected ~1.5× compress speedup.

---

## Running benchmarks

```bash
# TurboQuant high_context (Qwen3-8B)
HF_HOME=/workspace/.hf_home .venv/bin/python scripts/run_bench.py \
    --kv-dtype turboquant_4bit --workloads high_context --max-model-len 16384

# FP8 high_context for comparison
HF_HOME=/workspace/.hf_home .venv/bin/python scripts/run_bench.py \
    --kv-dtype fp8 --workloads high_context --max-model-len 16384

# Unit tests
.venv/bin/python -m pytest tests/v1/test_turboquant_kv.py -v -k "not e2e"
```

---

## If you hit issues

- **CUDA OOM**: reduce `gpu_memory_utilization` or reduce number of sequences in the workload
- **Import errors**: make sure vllm is installed: `.venv/bin/python -c "import vllm; print(vllm.__version__)"`
- **Model not found**: check HF_HOME is set and model is downloaded
- **Weight folding returning 0 layers**: check server logs for "TurboQuant: folded V/output rotations" — if 0, the model architecture may use different attribute names for attention heads
