#!/usr/bin/env bash
#SBATCH --job-name=ttt-sampler-pool
#SBATCH --output=logs/sampler_pool_%A_%a.out
#SBATCH --error=logs/sampler_pool_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#
# Two processes per array task:
#   1. SGLang server (background) on PORT_SGLANG.
#   2. ttt_kernel.sampler.server (foreground) on PORT_SAMPLER, points at #1.
#
# RUN_ROOT, TTT_CONFIG required (see launch_env_pool.sbatch).
set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${TTT_CONFIG:?TTT_CONFIG must be set}"
: "${TTT_ROOT:=$(pwd)}"
: "${SAMPLER_PORT_BASE:=8200}"
: "${SGLANG_PORT_BASE:=30000}"
: "${SAMPLER_MAX_CONCURRENT:=32}"
: "${SAMPLER_MAX_LOADED_ADAPTERS:=8}"
: "${MODEL_NAME:=Qwen/Qwen3-4B-Instruct-2507}"
: "${SAMPLER_ADAPTER_DIR:=$RUN_ROOT/sampler_seed_adapter}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
PORT_SAMPLER=$(( SAMPLER_PORT_BASE + IDX ))
PORT_SGLANG=$(( SGLANG_PORT_BASE + IDX ))

if command -v module >/dev/null 2>&1; then
  module load cuda/13.0.2 || true
fi
export CUDA_HOME="${CUDA_HOME:-/apps/software/extern/cuda/13.0.2}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$RUN_ROOT/registry/sampler" "$SAMPLER_ADAPTER_DIR" logs

cd "$TTT_ROOT"

MODEL_NAME="$MODEL_NAME" \
  PORT="$PORT_SGLANG" \
  HOST=127.0.0.1 \
  ADAPTER_DIR="$SAMPLER_ADAPTER_DIR" \
  ADAPTER_NAME=ttt_seed \
  scripts/launch_sglang.sh > "$RUN_ROOT/logs/sglang_${IDX}.log" 2>&1 &
SGLANG_PID=$!

cleanup() {
  if kill -0 "$SGLANG_PID" 2>/dev/null; then
    kill "$SGLANG_PID" 2>/dev/null || true
    wait "$SGLANG_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

exec "$TTT_ROOT/.venv/bin/python" -u -m ttt_kernel.sampler.server \
    --sglang-url "http://127.0.0.1:${PORT_SGLANG}" \
    --model "$MODEL_NAME" \
    --port "$PORT_SAMPLER" \
    --max-concurrent "$SAMPLER_MAX_CONCURRENT" \
    --max-loaded-adapters "$SAMPLER_MAX_LOADED_ADAPTERS" \
    --wait-ready-s 600 \
    --run-root "$RUN_ROOT" \
    --idx "$IDX"
