#!/usr/bin/env bash
# Launch vLLM OpenAI-compatible server with TurboQuant 4-bit KV cache.
# Usage: bash scripts/launch_tq_server.sh [port]
set -euo pipefail

PORT=${1:-8001}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MAX_LEN=${MAX_LEN:-32768}
GPU_UTIL=${GPU_UTIL:-0.85}
HF_HOME=${HF_HOME:-/workspace/.hf_home}

export HF_HOME

echo "[TQ-server] model=$MODEL port=$PORT max_len=$MAX_LEN gpu_util=$GPU_UTIL"
echo "[TQ-server] kv_cache_dtype=turboquant_4bit (V/output rotations will be folded at load)"
exec .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --kv-cache-dtype turboquant_4bit \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" \
    --enable-prefix-caching False \
    --disable-log-stats
