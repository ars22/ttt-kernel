#!/usr/bin/env bash
# Multi-node inference-only run: 2 nodes × {SGLang dp=3 tp=2 (6 GPUs) + 2 env (2 GPUs)}.
# Requires an existing SLURM allocation across 2 nodes (8 GPUs each).
# Orchestrator runs on the head node and discovers services via the shared
# filesystem registry under $RUN_ROOT/registry/.
#
# Usage:
#   ./scripts/run_inference_only_multinode.sh [config.yaml] [run_name]
set -euo pipefail

CONFIG="${1:-configs/inference_only_gpt_oss_20b.yaml}"
RUN_NAME="${2:-infer_$(date +%Y%m%d_%H%M%S)}"
TTT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="$TTT_ROOT/runs/$RUN_NAME"
LOGS="$RUN_ROOT/logs"
mkdir -p "$LOGS" "$RUN_ROOT/registry/sampler" "$RUN_ROOT/registry/env"

PY="$TTT_ROOT/.venv/bin/python"
# Honor an externally-set MODEL_NAME (e.g. when comparing models). Default
# matches the gpt-oss-120b config so existing invocations don't change.
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-120b}"

# ---- discover allocated nodes ---------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "[run] ERROR: no SLURM_JOB_ID — this script must run inside an allocation."
  exit 1
fi

NODE_LIST=$(scontrol show hostnames "$(squeue -h -j "$SLURM_JOB_ID" -O NodeList:200)")
NODES=($NODE_LIST)
if [ "${#NODES[@]}" -lt 2 ]; then
  echo "[run] ERROR: need at least 2 nodes in allocation, got: ${NODES[*]}"
  exit 1
fi
NODE0="${NODES[0]}"
NODE1="${NODES[1]}"

echo "[run] RUN_ROOT=$RUN_ROOT"
echo "[run] config=$CONFIG"
echo "[run] model=$MODEL_NAME"
echo "[run] node0=$NODE0  node1=$NODE1"

BACKEND="${BACKEND:-sglang}"
case "$BACKEND" in
  sglang)
    WORKER_SCRIPT="$TTT_ROOT/scripts/_inference_worker.sh"
    ;;
  vllm)
    WORKER_SCRIPT="$TTT_ROOT/scripts/_inference_worker_vllm.sh"
    ;;
  *)
    echo "[run] ERROR: BACKEND must be sglang or vllm, got: $BACKEND"
    exit 1
    ;;
esac
echo "[run] backend=$BACKEND worker=$WORKER_SCRIPT"

export TTT_ROOT RUN_ROOT CONFIG MODEL_NAME SGLANG_CONTEXT_LENGTH VLLM_MAX_MODEL_LEN

# Orchestrator runs on this node and also touches the HF cache (when it
# materializes per-problem seed adapters). Redirect every cache off /home
# (quota-bound) to the shared /project/flame mount.
export HF_HOME="/project/flame/asetlur/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="/project/flame/asetlur/.cache/torch"
export TRITON_CACHE_DIR="/project/flame/asetlur/.cache/triton"
export XDG_CACHE_HOME="/project/flame/asetlur/.cache"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR"

# ---- pre-warm the HF datasets cache so 4 concurrent env workers across the
#      two nodes don't collide on the NFS FileLock when they all try to call
#      load_dataset() at startup. Cache lives under $HF_HOME on /project/flame.
echo "[run] pre-warming HF datasets cache for ScalingIntelligence/KernelBench"
"$PY" - <<'PY' 2>&1 | tail -5
from datasets import load_dataset
load_dataset("ScalingIntelligence/KernelBench", split="level_1")
print("[prewarm] dataset cache ready")
PY

# ---- launch workers on each node via srun --------------------------------
echo "[run] launching worker on $NODE0 (NODE_IDX=0, ENV_IDX_BASE=0)"
NODE_IDX=0 ENV_IDX_BASE=0 \
  srun --nodes=1 --ntasks=1 --nodelist="$NODE0" --gres=gpu:8 --cpus-per-task=32 \
    --export=ALL,TTT_ROOT,RUN_ROOT,CONFIG,MODEL_NAME,NODE_IDX,ENV_IDX_BASE,SGLANG_CONTEXT_LENGTH,VLLM_MAX_MODEL_LEN \
    --output="$LOGS/worker_${NODE0}.out" --error="$LOGS/worker_${NODE0}.err" \
    bash "$WORKER_SCRIPT" &
W0_PID=$!

echo "[run] launching worker on $NODE1 (NODE_IDX=1, ENV_IDX_BASE=4)"
NODE_IDX=1 ENV_IDX_BASE=4 \
  srun --nodes=1 --ntasks=1 --nodelist="$NODE1" --gres=gpu:8 --cpus-per-task=32 \
    --export=ALL,TTT_ROOT,RUN_ROOT,CONFIG,MODEL_NAME,NODE_IDX,ENV_IDX_BASE,SGLANG_CONTEXT_LENGTH,VLLM_MAX_MODEL_LEN \
    --output="$LOGS/worker_${NODE1}.out" --error="$LOGS/worker_${NODE1}.err" \
    bash "$WORKER_SCRIPT" &
W1_PID=$!

# ---- cleanup on exit -------------------------------------------------------
trap_handler() {
  echo "[run] received signal, terminating workers (srun jobs $W0_PID $W1_PID)…"
  kill "$W0_PID" "$W1_PID" 2>/dev/null || true
  scancel --signal=TERM --jobid="$SLURM_JOB_ID" --steps 2>/dev/null || true
  exit 130
}
trap trap_handler INT TERM

# ---- orchestrator (foreground on this node) -------------------------------
# 2 samplers, 4 envs, 0 trainers.
echo "[run] giving workers 30s to advertise to the registry…"
sleep 30

echo "[run] starting orchestrator (inference-only, foreground)"
"$PY" -u -m ttt_kernel.orchestrator.main \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --num-samplers 2 \
    --num-envs 8 \
    --num-trainers 0 \
    --inference-only \
    --log-level info \
    2>&1 | tee "$LOGS/orchestrator.log"

ORCH_RC=${PIPESTATUS[0]}
echo "[run] orchestrator exited with rc=$ORCH_RC, cleaning up workers"
kill "$W0_PID" "$W1_PID" 2>/dev/null || true
scancel --signal=TERM --jobid="$SLURM_JOB_ID" --steps 2>/dev/null || true
exit "$ORCH_RC"
