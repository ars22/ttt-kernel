#!/usr/bin/env bash
# Per-node worker for run_inference_only_multinode.sh — vLLM backend.
# Launches: vLLM (GPUs 0-3, tp=4), sampler shim, 4 env services (GPUs 4-7).
# Args (env vars passed from caller):
#   TTT_ROOT, RUN_ROOT, CONFIG, MODEL_NAME
#   NODE_IDX (0 or 1 — used as sampler idx)
#   ENV_IDX_BASE (0 or 4 — env services idx start)
set -euo pipefail

: "${TTT_ROOT:?}"
: "${RUN_ROOT:?}"
: "${CONFIG:?}"
: "${MODEL_NAME:?}"
: "${NODE_IDX:?}"
: "${ENV_IDX_BASE:?}"

LOGS="$RUN_ROOT/logs"
NODE_HOST="$(hostname)"
SUFFIX="${NODE_HOST}"
mkdir -p "$LOGS"

# Redirect every cache off /home (which is over quota). All paths live on
# the shared /project/flame mount so both nodes see the same downloads.
export HF_HOME="/project/flame/asetlur/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="/project/flame/asetlur/.cache/torch"
# Triton's JIT cache must NOT live on NFS: multiple ranks within and across
# nodes concurrently write the same compile artifacts, and NFS returns
# "Errno 116 Stale file handle" mid-read. Use a per-node /tmp directory.
export TRITON_CACHE_DIR="/tmp/asetlur_triton_${NODE_HOST}_${SLURM_JOB_ID:-noid}"
# Same reasoning for HF datasets cache: the env service uses datasets.load_dataset()
# which acquires a FileLock; concurrent envs across nodes hit stale-handle in the
# fstat() check on NFS. Use a node-local cache and pre-warm it before launching envs.
export HF_DATASETS_CACHE="/tmp/asetlur_hf_datasets_${NODE_HOST}_${SLURM_JOB_ID:-noid}"
export XDG_CACHE_HOME="/project/flame/asetlur/.cache"
# vLLM's torch_compile_cache must NOT live on NFS either: with dp=4 the 4
# engine cores concurrently read/write the same AOT-compiled .py files, and
# NFS returns Errno 116 Stale file handle mid-import. Use per-node /tmp.
# (One-time ~2 min recompile per node-launch; acceptable for the throughput win.)
export VLLM_CACHE_ROOT="/tmp/asetlur_vllm_${NODE_HOST}_${SLURM_JOB_ID:-noid}"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE" "$VLLM_CACHE_ROOT"

PY="$TTT_ROOT/.venv/bin/python"
# SGLang JIT-compiles a rope kernel via tvm_ffi → ninja. Put venv bin on PATH so
# subprocess.run(['ninja', ...]) resolves to the venv-installed binary.
export PATH="$TTT_ROOT/.venv/bin:$PATH"

# Per-sandbox GPU memory cap. Inference-only has no trainer sharing the env
# GPU, so we can give each eval sandbox most of the device. With K=1 rollouts
# only one sandbox is active per env at a time, so 0.8 is safe.
# Default in eval_worker.py is 0.06 which OOMs on KernelBench problems with
# ~4-8 GB reference tensors (e.g. L1 p4/p5 strided convs).
export TTT_SANDBOX_MEM_FRACTION="${TTT_SANDBOX_MEM_FRACTION:-0.8}"

# vLLM + Qwen3.5 hybrid arch: the FlashInfer GDN prefill kernel JIT-compiles at
# startup and can hang for 10+ minutes; --gdn-prefill-backend triton avoids it.
# OMP_NUM_THREADS warning from vLLM is benign but we set it to silence the noise.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Ports (same per-node — different hosts, no conflict)
VLLM_PORT=30100
SAMPLER_PORT=8200
ENV0_PORT=8100
ENV1_PORT=8101
ENV2_PORT=8102
ENV3_PORT=8103

cd "$TTT_ROOT"

echo "[worker:$NODE_HOST] starting (NODE_IDX=$NODE_IDX, ENV_IDX_BASE=$ENV_IDX_BASE)"
echo "[worker:$NODE_HOST] RUN_ROOT=$RUN_ROOT model=$MODEL_NAME (inference-only, no LoRA)"

# Pre-warm the node-local HF datasets cache so the 2 env services on this node
# don't race the FileLock during load_dataset(). One serial call populates the
# cache; subsequent calls from the env services find it ready and skip locking.
echo "[worker:$NODE_HOST] pre-warming HF datasets cache at $HF_DATASETS_CACHE"
"$PY" - <<'PY'
from datasets import load_dataset
load_dataset("ScalingIntelligence/KernelBench", split="level_1")
print("[prewarm] node-local datasets cache ready")
PY

# ---- vLLM (GPUs 0-3, tp=1 dp=4 — 4 independent replicas) ------------------
# Qwen3.5-4B (~8 GB bf16) fits comfortably on one H100, so tp=1 is enough.
# dp=4 gives 4 parallel decode engines, ~3-4x the throughput of tp=4. Each
# replica owns one GPU and serves its own requests; vLLM round-robins the
# OpenAI API across DP ranks on the single bound port.
# --gdn-prefill-backend triton avoids the FlashInfer GDN JIT compile hang.
echo "[worker:$NODE_HOST] launching vLLM (GPUs 0-3, tp=1 dp=4, base model) → $LOGS/vllm_${SUFFIX}.log"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.7 \
    --gdn-prefill-backend triton \
    --trust-remote-code \
    > "$LOGS/vllm_${SUFFIX}.log" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > "$RUN_ROOT/vllm_${SUFFIX}.pid"

# ---- Sampler shim ----------------------------------------------------------
echo "[worker:$NODE_HOST] launching sampler shim (idx=$NODE_IDX) → $LOGS/sampler_${SUFFIX}.log"
nohup "$PY" -u -m ttt_kernel.sampler.server \
    --sglang-url "http://127.0.0.1:${VLLM_PORT}" \
    --model "$MODEL_NAME" \
    --port "$SAMPLER_PORT" \
    --max-concurrent 16 \
    --max-loaded-adapters 24 \
    --wait-ready-s 1800 \
    --run-root "$RUN_ROOT" \
    --idx "$NODE_IDX" \
    --advertise-host "$NODE_HOST" \
    > "$LOGS/sampler_${SUFFIX}.log" 2>&1 &
echo "$!" > "$RUN_ROOT/sampler_${SUFFIX}.pid"

# ---- Env services (GPUs 4, 5, 6, 7 — one env per GPU) ----------------------
# 4 envs per node × 2 nodes = 8 env services in the orchestrator pool, so eval
# can run 8 problems concurrently (vs 4 with the dp=3 layout).
ENV_PORTS=("$ENV0_PORT" "$ENV1_PORT" "$ENV2_PORT" "$ENV3_PORT")
ENV_GPUS=(4 5 6 7)
ENV_IDXS=()
for k in 0 1 2 3; do
  EIDX=$((ENV_IDX_BASE + k))
  ENV_IDXS+=("$EIDX")
  EPORT="${ENV_PORTS[$k]}"
  EGPU="${ENV_GPUS[$k]}"
  echo "[worker:$NODE_HOST] launching env service idx=$EIDX (GPU $EGPU, port $EPORT) → $LOGS/env_${EIDX}_${SUFFIX}.log"
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

# ---- Stay alive until killed; propagate signals to children ---------------
PIDS=()
for f in vllm sampler; do
  PIDS+=("$(cat "$RUN_ROOT/${f}_${SUFFIX}.pid" 2>/dev/null || true)")
done
for EIDX in "${ENV_IDXS[@]}"; do
  PIDS+=("$(cat "$RUN_ROOT/env_${EIDX}_${SUFFIX}.pid" 2>/dev/null || true)")
done
trap 'echo "[worker:$NODE_HOST] received signal, killing children"; for p in "${PIDS[@]}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; exit 0' INT TERM

echo "[worker:$NODE_HOST] all services launched, waiting…"
# Wait on any child; if any dies we exit so srun cleans up.
wait -n
echo "[worker:$NODE_HOST] a child exited, terminating remaining children"
for p in "${PIDS[@]}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
exit 1
