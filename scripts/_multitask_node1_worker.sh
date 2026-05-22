#!/usr/bin/env bash
# Node-1 worker for run_multitask_multinode.sh.
# Owns ONE trainer service running torchrun --nproc-per-node=8 (FSDP2
# full-shard across all 8 H100s). The trainer trains Qwen3-32B base model
# in place and pushes weight updates to SGLang via NCCL after each step.
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

# Dodge backward-pass OOM on Qwen3-32B FSDP at 8k seqlen: expandable segments
# avoid fragmentation when grad-accum allocates/frees 9-10 GB buffers repeatedly.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$TTT_ROOT"

TRAINER_PORT=8300
MASTER_PORT_FSDP=29500   # FSDP's intra-trainer NCCL group
# Multitask weight-broadcast master port (advertised to SGLang as the
# trainer-side rendezvous addr) is read by the trainer from the config —
# default 29600. It must be free on this node.

echo "[mt-node1:$NODE_HOST] launching trainer (8-GPU FSDP full-shard, --full-model)"
nohup "$TORCHRUN" \
    --nproc-per-node=8 \
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

trap 'echo "[mt-node1:$NODE_HOST] signal received, killing trainer"; kill "$TRAINER_PID" 2>/dev/null || true; exit 0' INT TERM

echo "[mt-node1:$NODE_HOST] trainer launched (pid=$TRAINER_PID), waiting…"
wait "$TRAINER_PID"
echo "[mt-node1:$NODE_HOST] trainer exited"
exit 1
