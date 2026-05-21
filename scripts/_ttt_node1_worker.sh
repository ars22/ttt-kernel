#!/usr/bin/env bash
# Node-1 worker for run_ttt_multinode.sh.
# Owns 4 trainer services. Each runs `torchrun --nproc-per-node=2` so the
# trainer model is split tp=2 across its pair of GPUs.
#
# Trainer layout on this node:
#   trainer idx=0  GPUs 0,1  port 8300
#   trainer idx=1  GPUs 2,3  port 8301
#   trainer idx=2  GPUs 4,5  port 8302
#   trainer idx=3  GPUs 6,7  port 8303
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

cd "$TTT_ROOT"

TRAINER_PORTS=(8300 8301 8302 8303)
TRAINER_GPUS=("0,1" "2,3" "4,5" "6,7")
MASTER_PORTS=(29500 29501 29502 29503)

echo "[ttt-node1:$NODE_HOST] starting 4 trainers (tp=2 each) on $NODE_HOST"

TRAINER_IDXS=()
for k in 0 1 2 3; do
  TIDX="$k"
  PORT="${TRAINER_PORTS[$k]}"
  GPUS="${TRAINER_GPUS[$k]}"
  MPORT="${MASTER_PORTS[$k]}"
  TRAINER_IDXS+=("$TIDX")
  echo "[ttt-node1:$NODE_HOST] launching trainer idx=$TIDX (GPUs $GPUS, port $PORT)"
  CUDA_VISIBLE_DEVICES="$GPUS" nohup "$TORCHRUN" \
      --nproc-per-node=2 \
      --master_port="$MPORT" \
      -m ttt_kernel.trainer.server \
      --fsdp \
      --config "$CONFIG" \
      --port "$PORT" \
      --max-concurrent 2 \
      --max-resident-adapters 4 \
      --run-root "$RUN_ROOT" \
      --idx "$TIDX" \
      --advertise-host "$NODE_HOST" \
      > "$LOGS/trainer_${TIDX}_${SUFFIX}.log" 2>&1 &
  echo "$!" > "$RUN_ROOT/trainer_${TIDX}_${SUFFIX}.pid"
done

PIDS=()
for TIDX in "${TRAINER_IDXS[@]}"; do
  PIDS+=("$(cat "$RUN_ROOT/trainer_${TIDX}_${SUFFIX}.pid" 2>/dev/null || true)")
done
trap 'echo "[ttt-node1:$NODE_HOST] signal received, killing trainers"; for p in "${PIDS[@]}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; exit 0' INT TERM

echo "[ttt-node1:$NODE_HOST] all trainers launched, waiting…"
wait -n
echo "[ttt-node1:$NODE_HOST] a trainer exited, terminating remaining"
for p in "${PIDS[@]}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
exit 1
