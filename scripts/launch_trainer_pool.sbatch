#!/usr/bin/env bash
#SBATCH --job-name=ttt-trainer-pool
#SBATCH --output=logs/trainer_pool_%A_%a.out
#SBATCH --error=logs/trainer_pool_%A_%a.err
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#
# Each array task = one trainer service. If TRAINER_FSDP=1, launch under
# torchrun --nproc-per-node=$TRAINER_GPUS and pass --fsdp to the server.
set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${TTT_CONFIG:?TTT_CONFIG must be set}"
: "${TTT_ROOT:=$(pwd)}"
: "${TRAINER_PORT_BASE:=8300}"
: "${TRAINER_MAX_CONCURRENT:=2}"
: "${TRAINER_MAX_RESIDENT_ADAPTERS:=8}"
: "${TRAINER_FSDP:=1}"
: "${TRAINER_GPUS:=8}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
PORT=$(( TRAINER_PORT_BASE + IDX ))

if command -v module >/dev/null 2>&1; then
  module load cuda/13.0.2 || true
fi
export CUDA_HOME="${CUDA_HOME:-/apps/software/extern/cuda/13.0.2}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$RUN_ROOT/registry/trainer" logs

cd "$TTT_ROOT"

if [ "$TRAINER_FSDP" = "1" ] && [ "$TRAINER_GPUS" -gt 1 ]; then
  exec "$TTT_ROOT/.venv/bin/torchrun" --nproc-per-node="$TRAINER_GPUS" \
      -m ttt_kernel.trainer.server \
      --fsdp \
      --config "$TTT_CONFIG" \
      --port "$PORT" \
      --max-concurrent "$TRAINER_MAX_CONCURRENT" \
      --max-resident-adapters "$TRAINER_MAX_RESIDENT_ADAPTERS" \
      --run-root "$RUN_ROOT" \
      --idx "$IDX"
else
  exec "$TTT_ROOT/.venv/bin/python" -u -m ttt_kernel.trainer.server \
      --config "$TTT_CONFIG" \
      --port "$PORT" \
      --max-concurrent "$TRAINER_MAX_CONCURRENT" \
      --max-resident-adapters "$TRAINER_MAX_RESIDENT_ADAPTERS" \
      --run-root "$RUN_ROOT" \
      --idx "$IDX"
fi
