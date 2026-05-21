#!/usr/bin/env bash
# Node-1 worker for run_multitask_multinode_4b.sh.
# Owns:
#   - ONE trainer: torchrun --nproc-per-node=4 (FSDP2 full-shard, GPUs 0-3)
#   - 4 env services on GPUs 4-7 (idx 0..3)
#
# Env vars: TTT_ROOT, RUN_ROOT, CONFIG, MODEL_NAME
set -euo pipefail

: "${TTT_ROOT:?}"
: "${RUN_ROOT:?}"
: "${CONFIG:?}"
: "${MODEL_NAME:?}"

LOGS="$RUN_ROOT/logs"
NODE_HOST="$(hostname)"
SUFFIX="${NODE_HOST}"
mkdir -p "$LOGS"

export HF_HOME="/project/flame/asetlur/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="/project/flame/asetlur/.cache/torch"
export TRITON_CACHE_DIR="/tmp/asetlur_triton_${NODE_HOST}_${SLURM_JOB_ID:-noid}"
export HF_DATASETS_CACHE="/tmp/asetlur_hf_datasets_${NODE_HOST}_${SLURM_JOB_ID:-noid}"
export XDG_CACHE_HOME="/project/flame/asetlur/.cache"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE"

PY="$TTT_ROOT/.venv/bin/python"
TORCHRUN="$TTT_ROOT/.venv/bin/torchrun"
export PATH="$TTT_ROOT/.venv/bin:$PATH"

# Same OOM-prevention as the 32B path.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Eval sandbox memory cap (envs run alone on their GPU).
export TTT_SANDBOX_MEM_FRACTION="${TTT_SANDBOX_MEM_FRACTION:-0.8}"

cd "$TTT_ROOT"

TRAINER_PORT=8300
MASTER_PORT_FSDP=29500
ENV_PORTS=(8100 8101 8102 8103)
ENV_GPUS=(4 5 6 7)

# Pre-warm HF datasets cache for the env services.
echo "[mt4b-node1:$NODE_HOST] pre-warming HF datasets cache at $HF_DATASETS_CACHE"
"$PY" - <<'PY'
from datasets import load_dataset
load_dataset("ScalingIntelligence/KernelBench", split="level_1")
print("[prewarm] node-local datasets cache ready")
PY

# ---- Trainer (GPUs 0-3, torchrun --nproc-per-node=4) -----------------------
echo "[mt4b-node1:$NODE_HOST] launching trainer (4-GPU FSDP full-shard, --full-model)"
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "$TORCHRUN" \
    --nproc-per-node=4 \
    --master_port="$MASTER_PORT_FSDP" \
    -m ttt_kernel.trainer.server \
    --fsdp \
    --full-model \
    --config "$CONFIG" \
    --port "$TRAINER_PORT" \
    --max-concurrent 1 \
    --max-resident-adapters 1 \
    --run-root "$RUN_ROOT" \
    --idx 0 \
    --advertise-host "$NODE_HOST" \
    > "$LOGS/trainer_0_${SUFFIX}.log" 2>&1 &
TRAINER_PID="$!"
echo "$TRAINER_PID" > "$RUN_ROOT/trainer_0_${SUFFIX}.pid"

# ---- Env services (GPUs 4-7, idx 0..3) -------------------------------------
ENV_IDXS=()
for k in 0 1 2 3; do
  EIDX="$k"
  EPORT="${ENV_PORTS[$k]}"
  EGPU="${ENV_GPUS[$k]}"
  ENV_IDXS+=("$EIDX")
  echo "[mt4b-node1:$NODE_HOST] launching env idx=$EIDX (GPU $EGPU, port $EPORT)"
  CUDA_VISIBLE_DEVICES="$EGPU" nohup "$PY" -u -m ttt_kernel.env.server \
      --config "$CONFIG" \
      --port "$EPORT" \
      --max-concurrent 12 \
      --sandbox-log "$LOGS/env_${EIDX}_${SUFFIX}.sandbox.log" \
      --run-root "$RUN_ROOT" \
      --idx "$EIDX" \
      --advertise-host "$NODE_HOST" \
      > "$LOGS/env_${EIDX}_${SUFFIX}.log" 2>&1 &
  echo "$!" > "$RUN_ROOT/env_${EIDX}_${SUFFIX}.pid"
done

PIDS=("$TRAINER_PID")
for EIDX in "${ENV_IDXS[@]}"; do
  PIDS+=("$(cat "$RUN_ROOT/env_${EIDX}_${SUFFIX}.pid" 2>/dev/null || true)")
done
trap 'echo "[mt4b-node1:$NODE_HOST] signal received, killing children"; for p in "${PIDS[@]}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; exit 0' INT TERM

echo "[mt4b-node1:$NODE_HOST] all services launched, waiting…"
wait -n
echo "[mt4b-node1:$NODE_HOST] a child exited, terminating remaining children"
for p in "${PIDS[@]}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
exit 1
