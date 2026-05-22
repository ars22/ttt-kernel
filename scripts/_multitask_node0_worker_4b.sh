#!/usr/bin/env bash
# Node-0 worker for run_multitask_multinode_4b.sh.
# Owns:
#   - SGLang (GPUs 0-7, tp=1 dp=8 → 8 independent Qwen3-4B replicas)
#   - sampler shim (1 instance, idx=0)
# NO env services on this node (envs live on node 1 GPUs 4-7).
#
# Env vars expected: TTT_ROOT, RUN_ROOT, CONFIG, MODEL_NAME
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

SGLANG_PORT=30100
SAMPLER_PORT=8200

cd "$TTT_ROOT"

echo "[mt4b-node0:$NODE_HOST] starting"
echo "[mt4b-node0:$NODE_HOST] RUN_ROOT=$RUN_ROOT model=$MODEL_NAME"

# Pre-warm node-local HF datasets cache (sampler itself doesn't use datasets,
# but keeps the cache layout consistent with the env-serving node).
echo "[mt4b-node0:$NODE_HOST] pre-warming HF datasets cache at $HF_DATASETS_CACHE"
"$PY" - <<'PY'
from datasets import load_dataset
load_dataset("ScalingIntelligence/KernelBench", split="level_1")
print("[prewarm] node-local datasets cache ready")
PY

# ---- SGLang (GPUs 0-7, tp=1 dp=8 — 8 replicas of Qwen3-4B) -----------------
# Qwen3-4B fits comfortably on one H100; dp=8 gives 8 parallel decode streams
# with no inter-GPU comm. mem-fraction-static 0.7 leaves headroom for KV cache.
# context-length 16k covers prompt (~2k) + max_tokens (8k) with margin.
echo "[mt4b-node0:$NODE_HOST] launching SGLang (GPUs 0-7, tp=1 dp=8, no LoRA) → $LOGS/sglang_${SUFFIX}.log"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup "$PY" -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$SGLANG_PORT" \
    --context-length 16384 \
    --dtype bfloat16 \
    --tp-size 1 \
    --dp-size 8 \
    --mem-fraction-static 0.7 \
    --attention-backend triton \
    --trust-remote-code \
    > "$LOGS/sglang_${SUFFIX}.log" 2>&1 &
echo "$!" > "$RUN_ROOT/sglang_${SUFFIX}.pid"

# ---- Sampler shim (1 instance on this node, idx=0) -------------------------
echo "[mt4b-node0:$NODE_HOST] launching sampler shim (idx=0, port $SAMPLER_PORT)"
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

PIDS=()
for f in sglang sampler; do
  PIDS+=("$(cat "$RUN_ROOT/${f}_${SUFFIX}.pid" 2>/dev/null || true)")
done
trap 'echo "[mt4b-node0:$NODE_HOST] signal received, killing children"; for p in "${PIDS[@]}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; exit 0' INT TERM

echo "[mt4b-node0:$NODE_HOST] all services launched, waiting…"
wait -n
echo "[mt4b-node0:$NODE_HOST] a child exited, terminating remaining children"
for p in "${PIDS[@]}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
exit 1
