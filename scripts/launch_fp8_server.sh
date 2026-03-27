#!/usr/bin/env bash
# Launch vLLM OpenAI-compatible server with FP8 KV cache.
# Usage: bash scripts/launch_fp8_server.sh [port]
set -euo pipefail

PORT=${1:-8000}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MAX_LEN=${MAX_LEN:-32768}
GPU_UTIL=${GPU_UTIL:-0.85}
HF_HOME=${HF_HOME:-/workspace/.hf_home}

export HF_HOME

echo "[FP8-server] model=$MODEL port=$PORT max_len=$MAX_LEN gpu_util=$GPU_UTIL"
exec .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --kv-cache-dtype fp8 \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" \
    --enable-prefix-caching False \
    --disable-log-stats
