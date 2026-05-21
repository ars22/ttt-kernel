#!/usr/bin/env bash
# Multi-node multi-task RL on Qwen3-4B.
#   Node 0: SGLang dp=8 tp=1 (8 GPUs) + sampler shim — NO envs.
#   Node 1: trainer torchrun --nproc-per-node=4 (GPUs 0-3, FSDP2) + 4 env services (GPUs 4-7).
#
# Usage:
#   ./scripts/run_multitask_multinode_4b.sh [config.yaml] [run_name]
set -euo pipefail

CONFIG="${1:-configs/multitask_qwen3_4b.yaml}"
RUN_NAME="${2:-multitask4b_$(date +%Y%m%d_%H%M%S)}"
TTT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="$TTT_ROOT/runs/$RUN_NAME"
LOGS="$RUN_ROOT/logs"
mkdir -p "$LOGS" "$RUN_ROOT/registry/sampler" "$RUN_ROOT/registry/env" "$RUN_ROOT/registry/trainer"

PY="$TTT_ROOT/.venv/bin/python"
MODEL_NAME="Qwen/Qwen3-4B"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "[run-mt4b] ERROR: no SLURM_JOB_ID — must run inside an allocation."
  exit 1
fi

NODE_LIST=$(scontrol show hostnames "$(squeue -h -j "$SLURM_JOB_ID" -O NodeList:200)")
NODES=($NODE_LIST)
if [ "${#NODES[@]}" -lt 2 ]; then
  echo "[run-mt4b] ERROR: need ≥2 nodes; got: ${NODES[*]}"
  exit 1
fi
NODE0="${NODES[0]}"
NODE1="${NODES[1]}"

echo "[run-mt4b] RUN_ROOT=$RUN_ROOT"
echo "[run-mt4b] config=$CONFIG  model=$MODEL_NAME"
echo "[run-mt4b] node0(sglang dp=8 + sampler)=$NODE0  node1(trainer×4 + 4 envs)=$NODE1"

export TTT_ROOT RUN_ROOT CONFIG MODEL_NAME

export HF_HOME="/project/flame/asetlur/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="/project/flame/asetlur/.cache/torch"
export TRITON_CACHE_DIR="/project/flame/asetlur/.cache/triton"
export XDG_CACHE_HOME="/project/flame/asetlur/.cache"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR"

echo "[run-mt4b] pre-warming HF datasets cache for ScalingIntelligence/KernelBench"
"$PY" - <<'PY' 2>&1 | tail -5
from datasets import load_dataset
load_dataset("ScalingIntelligence/KernelBench", split="level_1")
print("[prewarm] dataset cache ready")
PY

# ---- node0 worker (sglang dp=8 + sampler, NO envs) -------------------------
echo "[run-mt4b] launching node0 worker (sglang dp=8 + sampler) on $NODE0"
srun --jobid="$SLURM_JOB_ID" --overlap \
    --nodes=1 --ntasks=1 --nodelist="$NODE0" --gres=gpu:8 --cpus-per-task=32 \
    --export=ALL,TTT_ROOT,RUN_ROOT,CONFIG,MODEL_NAME \
    --output="$LOGS/worker_node0_${NODE0}.out" --error="$LOGS/worker_node0_${NODE0}.err" \
    bash "$TTT_ROOT/scripts/_multitask_node0_worker_4b.sh" &
W0_PID=$!

# ---- node1 worker (trainer×4 + 4 envs) -------------------------------------
echo "[run-mt4b] launching node1 worker (trainer×4 + 4 envs) on $NODE1"
srun --jobid="$SLURM_JOB_ID" --overlap \
    --nodes=1 --ntasks=1 --nodelist="$NODE1" --gres=gpu:8 --cpus-per-task=32 \
    --export=ALL,TTT_ROOT,RUN_ROOT,CONFIG,MODEL_NAME \
    --output="$LOGS/worker_node1_${NODE1}.out" --error="$LOGS/worker_node1_${NODE1}.err" \
    bash "$TTT_ROOT/scripts/_multitask_node1_worker_4b.sh" &
W1_PID=$!

trap_handler() {
  echo "[run-mt4b] signal received, terminating workers ($W0_PID $W1_PID)…"
  kill "$W0_PID" "$W1_PID" 2>/dev/null || true
  exit 130
}
trap trap_handler INT TERM

echo "[run-mt4b] giving workers 120s before launching orchestrator…"
sleep 120

NODE1_IP=$(host -t A "$NODE1" 2>/dev/null | awk '/has address/{print $4; exit}')
if [ -z "$NODE1_IP" ]; then
  echo "[run-mt4b] ERROR: failed to resolve $NODE1 to an IPv4 address"
  exit 1
fi
echo "[run-mt4b] NODE1 IPv4 for weight-broadcast rendezvous: $NODE1 -> $NODE1_IP"

SGLANG_URL_FOR_TRAINER="http://${NODE0}:30100"
echo "[run-mt4b] starting multitask orchestrator (1 sampler, 4 envs, 1 trainer)"
"$PY" -u -m ttt_kernel.orchestrator.multitask \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --num-envs 4 \
    --sglang-url-for-trainer "$SGLANG_URL_FOR_TRAINER" \
    --trainer-master-addr "$NODE1_IP" \
    --log-level info \
    2>&1 | tee "$LOGS/orchestrator.log"

ORCH_RC=${PIPESTATUS[0]}
echo "[run-mt4b] orchestrator exited rc=$ORCH_RC, cleaning up workers"
kill "$W0_PID" "$W1_PID" 2>/dev/null || true
exit "$ORCH_RC"
