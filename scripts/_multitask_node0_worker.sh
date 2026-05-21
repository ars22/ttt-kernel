#!/usr/bin/env bash
# Node-0 worker for run_multitask_multinode.sh.
# Owns:
#   - SGLang (GPUs 0-3, tp=4 dp=1, --enable-update-weights-from-distributed)
#   - sampler shim (1 instance, idx=0)
#   - 4 env services (GPUs 4-7, idx 0..3)
#
# Difference from _ttt_node0_worker.sh: no --enable-lora (multitask has no
# adapters; weights are pushed in place via NCCL from the trainer).
#
# Env vars expected:
#   TTT_ROOT, RUN_ROOT, CONFIG, MODEL_NAME
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
export PATH="$TTT_ROOT/.venv/bin:$PATH"

# Eval workers run alone on GPUs 4-7 → safe to take most of the device.
export TTT_SANDBOX_MEM_FRACTION="${TTT_SANDBOX_MEM_FRACTION:-0.8}"

SGLANG_PORT=30100
SAMPLER_PORT=8200
ENV_PORTS=(8100 8101 8102 8103)
ENV_GPUS=(4 5 6 7)

cd "$TTT_ROOT"

echo "[mt-node0:$NODE_HOST] starting"
echo "[mt-node0:$NODE_HOST] RUN_ROOT=$RUN_ROOT model=$MODEL_NAME"

# Pre-warm node-local HF datasets cache.
echo "[mt-node0:$NODE_HOST] pre-warming HF datasets cache at $HF_DATASETS_CACHE"
"$PY" - <<'PY'
from datasets import load_dataset
load_dataset("ScalingIntelligence/KernelBench", split="level_1")
print("[prewarm] node-local datasets cache ready")
PY

# ---- SGLang (GPUs 0-3, tp=4 dp=1, weight-update-from-distributed enabled) --
# We do NOT pass --enable-lora (multitask has no adapters). The trainer will
# push updated base weights via the /update_weights_from_distributed endpoint.
echo "[mt-node0:$NODE_HOST] launching SGLang (GPUs 0-3, tp=4 dp=1, no LoRA) → $LOGS/sglang_${SUFFIX}.log"
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "$PY" -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$SGLANG_PORT" \
    --context-length 32768 \
    --dtype bfloat16 \
    --tp-size 4 \
    --dp-size 1 \
    --mem-fraction-static 0.7 \
    --attention-backend triton \
    --trust-remote-code \
    > "$LOGS/sglang_${SUFFIX}.log" 2>&1 &
echo "$!" > "$RUN_ROOT/sglang_${SUFFIX}.pid"

# ---- Sampler shim (1 instance on this node, idx=0) -------------------------
echo "[mt-node0:$NODE_HOST] launching sampler shim (idx=0, port $SAMPLER_PORT)"
nohup "$PY" -u -m ttt_kernel.sampler.server \
    --sglang-url "http://127.0.0.1:${SGLANG_PORT}" \
    --model "$MODEL_NAME" \
    --port "$SAMPLER_PORT" \
    --max-concurrent 256 \
    --max-loaded-adapters 1 \
    --wait-ready-s 1800 \
    --run-root "$RUN_ROOT" \
    --idx 0 \
    --advertise-host "$NODE_HOST" \
    > "$LOGS/sampler_${SUFFIX}.log" 2>&1 &
echo "$!" > "$RUN_ROOT/sampler_${SUFFIX}.pid"

# ---- Env services (GPUs 4-7, idx 0..3) -------------------------------------
ENV_IDXS=()
for k in 0 1 2 3; do
  EIDX="$k"
  EPORT="${ENV_PORTS[$k]}"
  EGPU="${ENV_GPUS[$k]}"
  ENV_IDXS+=("$EIDX")
  echo "[mt-node0:$NODE_HOST] launching env idx=$EIDX (GPU $EGPU, port $EPORT)"
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

PIDS=()
for f in sglang sampler; do
  PIDS+=("$(cat "$RUN_ROOT/${f}_${SUFFIX}.pid" 2>/dev/null || true)")
done
for EIDX in "${ENV_IDXS[@]}"; do
  PIDS+=("$(cat "$RUN_ROOT/env_${EIDX}_${SUFFIX}.pid" 2>/dev/null || true)")
done
trap 'echo "[mt-node0:$NODE_HOST] signal received, killing children"; for p in "${PIDS[@]}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; exit 0' INT TERM

echo "[mt-node0:$NODE_HOST] all services launched, waiting…"
wait -n
echo "[mt-node0:$NODE_HOST] a child exited, terminating remaining children"
for p in "${PIDS[@]}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
exit 1
