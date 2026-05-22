#!/usr/bin/env bash
# Node-1 worker for run_ttt_multinode_single.sh.
# Owns ONE trainer (single-problem TTT-LoRA RL).
# Trainer: torchrun --nproc-per-node=4 on GPUs 0-3 (tp=4 FSDP, --fsdp).
# GPUs 4-7 idle.
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
export XDG_CACHE_HOME="/project/flame/asetlur/.cache"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR"

PY="$TTT_ROOT/.venv/bin/python"
TORCHRUN="$TTT_ROOT/.venv/bin/torchrun"
export PATH="$TTT_ROOT/.venv/bin:$PATH"

# Avoid fragmentation OOM during the LoRA backward pass over 512 rollouts.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$TTT_ROOT"

TRAINER_PORT=8300
MASTER_PORT_FSDP=29500

echo "[ttt1-node1:$NODE_HOST] launching single trainer (tp=4 FSDP, LoRA mode)"
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "$TORCHRUN" \
    --nproc-per-node=4 \
    --master_port="$MASTER_PORT_FSDP" \
    -m ttt_kernel.trainer.server \
    --fsdp \
    --config "$CONFIG" \
    --port "$TRAINER_PORT" \
    --max-concurrent 1 \
    --max-resident-adapters 2 \
    --run-root "$RUN_ROOT" \
    --idx 0 \
    --advertise-host "$NODE_HOST" \
    > "$LOGS/trainer_0_${SUFFIX}.log" 2>&1 &
TRAINER_PID="$!"
echo "$TRAINER_PID" > "$RUN_ROOT/trainer_0_${SUFFIX}.pid"

trap 'echo "[ttt1-node1:$NODE_HOST] signal received, killing trainer"; kill "$TRAINER_PID" 2>/dev/null || true; exit 0' INT TERM

echo "[ttt1-node1:$NODE_HOST] trainer launched (pid=$TRAINER_PID), waiting…"
wait "$TRAINER_PID"
echo "[ttt1-node1:$NODE_HOST] trainer exited"
exit 1
