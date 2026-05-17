#!/usr/bin/env bash
#SBATCH --job-name=ttt-orchestrator
#SBATCH --output=logs/orchestrator_%j.out
#SBATCH --error=logs/orchestrator_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#
# Single-node orchestrator. Polls $RUN_ROOT/registry until the pools have
# advertised the expected number of services.
set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${TTT_CONFIG:?TTT_CONFIG must be set}"
: "${TTT_ROOT:=$(pwd)}"
: "${NUM_SAMPLERS:=1}"
: "${NUM_ENVS:=1}"
: "${NUM_TRAINERS:=1}"

mkdir -p logs

cd "$TTT_ROOT"

PROBLEM_IDS_FLAG=""
if [ -n "${PROBLEM_IDS:-}" ]; then
  PROBLEM_IDS_FLAG="--problem-ids $PROBLEM_IDS"
fi

# shellcheck disable=SC2086
exec "$TTT_ROOT/.venv/bin/python" -u -m ttt_kernel.orchestrator.main \
    --config "$TTT_CONFIG" \
    --run-root "$RUN_ROOT" \
    --num-samplers "$NUM_SAMPLERS" \
    --num-envs "$NUM_ENVS" \
    --num-trainers "$NUM_TRAINERS" \
    $PROBLEM_IDS_FLAG
