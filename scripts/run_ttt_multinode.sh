#!/usr/bin/env bash
# Multi-node TTT (full GRPO loop) run.
#   Node 0: SGLang tp=4 dp=1 (GPUs 0-3) + 4 env services (GPUs 4-7)
#   Node 1: 4 trainers, each torchrun nproc-per-node=2 (GPUs 0,1 / 2,3 / 4,5 / 6,7)
#
# Orchestrator: 1 sampler, 4 envs, 4 trainers. Runs on Node 0.
# Requires an existing SLURM allocation across 2 nodes (8 GPUs each).
#
# Usage:
#   ./scripts/run_ttt_multinode.sh [config.yaml] [run_name]
set -euo pipefail

CONFIG="${1:-configs/ttt_gpt_oss_120b_smoke.yaml}"
RUN_NAME="${2:-ttt_$(date +%Y%m%d_%H%M%S)}"
TTT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="$TTT_ROOT/runs/$RUN_NAME"
LOGS="$RUN_ROOT/logs"
mkdir -p "$LOGS" "$RUN_ROOT/registry/sampler" "$RUN_ROOT/registry/env" "$RUN_ROOT/registry/trainer"

PY="$TTT_ROOT/.venv/bin/python"
MODEL_NAME="openai/gpt-oss-120b"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "[run-ttt] ERROR: no SLURM_JOB_ID — this script must run inside an allocation."
  exit 1
fi

NODE_LIST=$(scontrol show hostnames "$(squeue -h -j "$SLURM_JOB_ID" -O NodeList:200)")
NODES=($NODE_LIST)
if [ "${#NODES[@]}" -lt 2 ]; then
  echo "[run-ttt] ERROR: need at least 2 nodes; got: ${NODES[*]}"
  exit 1
fi
NODE0="${NODES[0]}"
NODE1="${NODES[1]}"

echo "[run-ttt] RUN_ROOT=$RUN_ROOT"
echo "[run-ttt] config=$CONFIG  model=$MODEL_NAME"
echo "[run-ttt] node0(sampler+envs)=$NODE0  node1(trainers)=$NODE1"

export TTT_ROOT RUN_ROOT CONFIG MODEL_NAME

# Orchestrator-side caches.
export HF_HOME="/project/flame/asetlur/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="/project/flame/asetlur/.cache/torch"
export TRITON_CACHE_DIR="/project/flame/asetlur/.cache/triton"
export XDG_CACHE_HOME="/project/flame/asetlur/.cache"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR"

# Pre-warm HF datasets cache on the launcher (orchestrator side); env workers
# will then redirect to per-node /tmp themselves.
echo "[run-ttt] pre-warming HF datasets cache for ScalingIntelligence/KernelBench"
"$PY" - <<'PY' 2>&1 | tail -5
from datasets import load_dataset
load_dataset("ScalingIntelligence/KernelBench", split="level_1")
print("[prewarm] dataset cache ready")
PY

# ---- node0 worker (sampler + envs) -----------------------------------------
echo "[run-ttt] launching node0 worker (sampler+envs) on $NODE0"
srun --jobid="$SLURM_JOB_ID" --overlap \
    --nodes=1 --ntasks=1 --nodelist="$NODE0" --gres=gpu:8 --cpus-per-task=32 \
    --export=ALL,TTT_ROOT,RUN_ROOT,CONFIG,MODEL_NAME \
    --output="$LOGS/worker_node0_${NODE0}.out" --error="$LOGS/worker_node0_${NODE0}.err" \
    bash "$TTT_ROOT/scripts/_ttt_node0_worker.sh" &
W0_PID=$!

# ---- node1 worker (trainers) -----------------------------------------------
echo "[run-ttt] launching node1 worker (trainers) on $NODE1"
srun --jobid="$SLURM_JOB_ID" --overlap \
    --nodes=1 --ntasks=1 --nodelist="$NODE1" --gres=gpu:8 --cpus-per-task=32 \
    --export=ALL,TTT_ROOT,RUN_ROOT,CONFIG,MODEL_NAME \
    --output="$LOGS/worker_node1_${NODE1}.out" --error="$LOGS/worker_node1_${NODE1}.err" \
    bash "$TTT_ROOT/scripts/_ttt_node1_worker.sh" &
W1_PID=$!

trap_handler() {
  echo "[run-ttt] received signal, terminating workers (srun jobs $W0_PID $W1_PID)…"
  kill "$W0_PID" "$W1_PID" 2>/dev/null || true
  exit 130
}
trap trap_handler INT TERM

# Give workers time to start SGLang (~3 min for 120b MXFP4) + trainers.
echo "[run-ttt] giving workers 60s before launching orchestrator…"
sleep 60

# ---- orchestrator (foreground on this node) --------------------------------
# --seed-skip is used because materialize_seeds would load the full 120b base
# model on the orchestrator side (slow, big CPU mem). Pre-materialize the v000
# adapter ahead of time (one-shot helper) — until then, every problem needs
# its v000 directory already present at $RUN_ROOT/adapters/v000_pXXX/.
echo "[run-ttt] starting orchestrator (1 sampler, 4 envs, 4 trainers, --seed-skip)"
"$PY" -u -m ttt_kernel.orchestrator.main \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --num-samplers 1 \
    --num-envs 4 \
    --num-trainers 4 \
    --seed-skip \
    --log-level info \
    2>&1 | tee "$LOGS/orchestrator.log"

ORCH_RC=${PIPESTATUS[0]}
echo "[run-ttt] orchestrator exited rc=$ORCH_RC, cleaning up workers"
kill "$W0_PID" "$W1_PID" 2>/dev/null || true
exit "$ORCH_RC"
