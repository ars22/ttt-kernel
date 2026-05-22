#!/usr/bin/env bash
# Convenience wrapper: submit the four sbatch jobs (envs, samplers, trainers,
# orchestrator) for one end-to-end refactor run. The orchestrator job
# depends on the others having registered, which happens via the filesystem
# registry, so we just submit them all at once with --kill-on-invalid-dep=no.
#
# Usage:
#   ./scripts/run_refactor.sh [config.yaml] [run_name]
#
# Env knobs (with defaults shown):
#   NUM_SAMPLERS=1 NUM_ENVS=2 NUM_TRAINERS=1
#   TRAINER_FSDP=1 TRAINER_GPUS=8
#   MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
set -euo pipefail

CONFIG="${1:-configs/small_smoke.yaml}"
RUN_NAME="${2:-smoke_$(date +%s)}"
TTT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="$TTT_ROOT/runs/$RUN_NAME"

export RUN_ROOT TTT_CONFIG="$CONFIG" TTT_ROOT
export NUM_SAMPLERS="${NUM_SAMPLERS:-1}"
export NUM_ENVS="${NUM_ENVS:-2}"
export NUM_TRAINERS="${NUM_TRAINERS:-1}"

mkdir -p "$RUN_ROOT/registry" "$RUN_ROOT/logs" logs
echo "[run_refactor] RUN_ROOT=$RUN_ROOT"
echo "[run_refactor] config=$CONFIG"
echo "[run_refactor] samplers=$NUM_SAMPLERS envs=$NUM_ENVS trainers=$NUM_TRAINERS"

if [ "$NUM_ENVS" -gt 0 ]; then
  ENV_ARRAY=$(( NUM_ENVS - 1 ))
  ENV_JOB=$(sbatch --parsable \
      --array="0-${ENV_ARRAY}" \
      --export=ALL,RUN_ROOT="$RUN_ROOT",TTT_CONFIG="$CONFIG",TTT_ROOT="$TTT_ROOT" \
      "$TTT_ROOT/scripts/launch_env_pool.sbatch")
  echo "[run_refactor] env_pool job=$ENV_JOB (array 0-$ENV_ARRAY)"
fi

if [ "$NUM_SAMPLERS" -gt 0 ]; then
  S_ARRAY=$(( NUM_SAMPLERS - 1 ))
  SAMPLER_JOB=$(sbatch --parsable \
      --array="0-${S_ARRAY}" \
      --export=ALL,RUN_ROOT="$RUN_ROOT",TTT_CONFIG="$CONFIG",TTT_ROOT="$TTT_ROOT" \
      "$TTT_ROOT/scripts/launch_sampler_pool.sbatch")
  echo "[run_refactor] sampler_pool job=$SAMPLER_JOB (array 0-$S_ARRAY)"
fi

if [ "$NUM_TRAINERS" -gt 0 ]; then
  T_ARRAY=$(( NUM_TRAINERS - 1 ))
  TRAINER_JOB=$(sbatch --parsable \
      --array="0-${T_ARRAY}" \
      --export=ALL,RUN_ROOT="$RUN_ROOT",TTT_CONFIG="$CONFIG",TTT_ROOT="$TTT_ROOT" \
      "$TTT_ROOT/scripts/launch_trainer_pool.sbatch")
  echo "[run_refactor] trainer_pool job=$TRAINER_JOB (array 0-$T_ARRAY)"
fi

ORCH_JOB=$(sbatch --parsable \
    --export=ALL,RUN_ROOT="$RUN_ROOT",TTT_CONFIG="$CONFIG",TTT_ROOT="$TTT_ROOT" \
    "$TTT_ROOT/scripts/launch_orchestrator.sbatch")
echo "[run_refactor] orchestrator job=$ORCH_JOB"
echo "[run_refactor] watch logs in $RUN_ROOT/logs/ and tail -f logs/orchestrator_${ORCH_JOB}.out"
