#!/usr/bin/env bash
#SBATCH --job-name=ttt-env-pool
#SBATCH --output=logs/env_pool_%A_%a.out
#SBATCH --error=logs/env_pool_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#
# One env-pool service per array task. RUN_ROOT must be exported by the
# caller (or run_refactor.sh) so all jobs share a registry directory.
#
# Usage:
#   sbatch --array=0-1 \
#       --export=ALL,RUN_ROOT=/weka/.../runs/smoke,TTT_CONFIG=configs/small_smoke.yaml \
#       scripts/launch_env_pool.sbatch
set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT must be set (shared run dir on Weka)}"
: "${TTT_CONFIG:?TTT_CONFIG must be set (path to YAML)}"
: "${TTT_ROOT:=$(pwd)}"
: "${ENV_PORT_BASE:=8100}"
: "${ENV_MAX_CONCURRENT:=4}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
PORT=$(( ENV_PORT_BASE + IDX ))

if command -v module >/dev/null 2>&1; then
  module load cuda/13.0.2 || true
fi
export CUDA_HOME="${CUDA_HOME:-/apps/software/extern/cuda/13.0.2}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$RUN_ROOT/registry/env" logs

cd "$TTT_ROOT"
exec "$TTT_ROOT/.venv/bin/python" -u -m ttt_kernel.env.server \
    --config "$TTT_CONFIG" \
    --port "$PORT" \
    --max-concurrent "$ENV_MAX_CONCURRENT" \
    --sandbox-log "$RUN_ROOT/logs/env_${IDX}.sandbox.log" \
    --run-root "$RUN_ROOT" \
    --idx "$IDX"
