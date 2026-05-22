#!/usr/bin/env bash
# Node-0 worker for run_ttt_multinode.sh.
# Owns sampler (SGLang tp=4 dp=1 on GPUs 0-3, with --enable-lora) and 4 env
# services (one per GPU 4-7).
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

# /home is quota-bound; route caches to /project/flame.
export HF_HOME="/project/flame/asetlur/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="/project/flame/asetlur/.cache/torch"
# Triton + HF datasets caches on NFS hit stale-handle races — pin per-node /tmp.
export TRITON_CACHE_DIR="/tmp/asetlur_triton_${NODE_HOST}_${SLURM_JOB_ID:-noid}"
export HF_DATASETS_CACHE="/tmp/asetlur_hf_datasets_${NODE_HOST}_${SLURM_JOB_ID:-noid}"
export XDG_CACHE_HOME="/project/flame/asetlur/.cache"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE"

PY="$TTT_ROOT/.venv/bin/python"
export PATH="$TTT_ROOT/.venv/bin:$PATH"

# Sandbox GPU memory cap: inference-eval workers run alone on GPUs 4-7 →
# safe to take most of the device. Default 0.06 (the in-code default) OOMs
# on real KernelBench tensors.
export TTT_SANDBOX_MEM_FRACTION="${TTT_SANDBOX_MEM_FRACTION:-0.8}"

SGLANG_PORT=30100
SAMPLER_PORT=8200
ENV_PORTS=(8100 8101 8102 8103)
ENV_GPUS=(4 5 6 7)

cd "$TTT_ROOT"

echo "[ttt-node0:$NODE_HOST] starting"
echo "[ttt-node0:$NODE_HOST] RUN_ROOT=$RUN_ROOT model=$MODEL_NAME"

# Pre-warm node-local HF datasets cache so env services don't collide.
echo "[ttt-node0:$NODE_HOST] pre-warming HF datasets cache at $HF_DATASETS_CACHE"
"$PY" - <<'PY'
from datasets import load_dataset
load_dataset("ScalingIntelligence/KernelBench", split="level_1")
print("[prewarm] node-local datasets cache ready")
PY

# ---- SGLang (GPUs 0-3, tp=4 dp=1, --enable-lora) ---------------------------
# tp=4 satisfies gpt-oss-120b vocab divisibility (201088). dp=1 is required
# for dynamic LoRA loading (SGLang rejects /load_lora_adapter when dp>1).
# --max-loras-per-batch 4 caps simultaneously-active adapters per batch.
echo "[ttt-node0:$NODE_HOST] launching SGLang (GPUs 0-3, tp=4 dp=1, --enable-lora) → $LOGS/sglang_${SUFFIX}.log"
SGLANG_CTX="${SGLANG_CONTEXT_LENGTH:-65536}"
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "$PY" -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$SGLANG_PORT" \
    --context-length "$SGLANG_CTX" \
    --dtype bfloat16 \
    --tp-size 4 \
    --dp-size 1 \
    --mem-fraction-static 0.7 \
    --attention-backend triton \
    --enable-lora \
    --max-loras-per-batch 4 \
    --max-lora-rank "${SGLANG_MAX_LORA_RANK:-32}" \
    --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --trust-remote-code \
    > "$LOGS/sglang_${SUFFIX}.log" 2>&1 &
echo "$!" > "$RUN_ROOT/sglang_${SUFFIX}.pid"

# ---- Sampler shim (1 instance on this node, idx=0) -------------------------
echo "[ttt-node0:$NODE_HOST] launching sampler shim (idx=0, port $SAMPLER_PORT)"
nohup "$PY" -u -m ttt_kernel.sampler.server \
    --sglang-url "http://127.0.0.1:${SGLANG_PORT}" \
    --model "$MODEL_NAME" \
    --port "$SAMPLER_PORT" \
    --max-concurrent 32 \
    --max-loaded-adapters 8 \
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
  echo "[ttt-node0:$NODE_HOST] launching env idx=$EIDX (GPU $EGPU, port $EPORT)"
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

# ---- Stay alive; propagate signals to children -----------------------------
PIDS=()
for f in sglang sampler; do
  PIDS+=("$(cat "$RUN_ROOT/${f}_${SUFFIX}.pid" 2>/dev/null || true)")
done
for EIDX in "${ENV_IDXS[@]}"; do
  PIDS+=("$(cat "$RUN_ROOT/env_${EIDX}_${SUFFIX}.pid" 2>/dev/null || true)")
done
trap 'echo "[ttt-node0:$NODE_HOST] signal received, killing children"; for p in "${PIDS[@]}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; exit 0' INT TERM

echo "[ttt-node0:$NODE_HOST] all services launched, waiting…"
wait -n
echo "[ttt-node0:$NODE_HOST] a child exited, terminating remaining children"
for p in "${PIDS[@]}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
exit 1
