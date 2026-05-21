#!/usr/bin/env bash
# Multi-node multi-task RL baseline (full-model REINFORCE on Qwen3-32B).
#   Node 0: SGLang tp=4 dp=1 (GPUs 0-3) + 4 env services (GPUs 4-7) + sampler shim
#   Node 1: ONE trainer running torchrun --nproc-per-node=8 (FSDP2 full-shard)
#
# Orchestrator: ttt_kernel.orchestrator.multitask, runs on the launcher
# node (same as run_ttt_multinode.sh).
#
# Requires an existing SLURM allocation across 2 nodes (8 GPUs each).
#
# Usage:
#   ./scripts/run_multitask_multinode.sh [config.yaml] [run_name]
set -euo pipefail

CONFIG="${1:-configs/multitask_qwen3_32b.yaml}"
RUN_NAME="${2:-multitask_$(date +%Y%m%d_%H%M%S)}"
TTT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="$TTT_ROOT/runs/$RUN_NAME"
LOGS="$RUN_ROOT/logs"
mkdir -p "$LOGS" "$RUN_ROOT/registry/sampler" "$RUN_ROOT/registry/env" "$RUN_ROOT/registry/trainer"

PY="$TTT_ROOT/.venv/bin/python"
MODEL_NAME="Qwen/Qwen3-32B"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "[run-mt] ERROR: no SLURM_JOB_ID — this script must run inside an allocation."
  exit 1
fi

NODE_LIST=$(scontrol show hostnames "$(squeue -h -j "$SLURM_JOB_ID" -O NodeList:200)")
NODES=($NODE_LIST)
if [ "${#NODES[@]}" -lt 2 ]; then
  echo "[run-mt] ERROR: need at least 2 nodes; got: ${NODES[*]}"
  exit 1
fi
NODE0="${NODES[0]}"
NODE1="${NODES[1]}"

echo "[run-mt] RUN_ROOT=$RUN_ROOT"
echo "[run-mt] config=$CONFIG  model=$MODEL_NAME"
echo "[run-mt] node0(sampler+envs+sglang)=$NODE0  node1(trainer)=$NODE1"

export TTT_ROOT RUN_ROOT CONFIG MODEL_NAME

# Orchestrator-side caches.
export HF_HOME="/project/flame/asetlur/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="/project/flame/asetlur/.cache/torch"
export TRITON_CACHE_DIR="/project/flame/asetlur/.cache/triton"
export XDG_CACHE_HOME="/project/flame/asetlur/.cache"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR"

# Pre-warm HF datasets on the launcher.
echo "[run-mt] pre-warming HF datasets cache for ScalingIntelligence/KernelBench"
"$PY" - <<'PY' 2>&1 | tail -5
from datasets import load_dataset
load_dataset("ScalingIntelligence/KernelBench", split="level_1")
print("[prewarm] dataset cache ready")
PY

# ---- node0 worker ----------------------------------------------------------
echo "[run-mt] launching node0 worker (sglang+sampler+envs) on $NODE0"
srun --jobid="$SLURM_JOB_ID" --overlap \
    --nodes=1 --ntasks=1 --nodelist="$NODE0" --gres=gpu:8 --cpus-per-task=32 \
    --export=ALL,TTT_ROOT,RUN_ROOT,CONFIG,MODEL_NAME \
    --output="$LOGS/worker_node0_${NODE0}.out" --error="$LOGS/worker_node0_${NODE0}.err" \
    bash "$TTT_ROOT/scripts/_multitask_node0_worker.sh" &
W0_PID=$!

# ---- node1 worker ----------------------------------------------------------
echo "[run-mt] launching node1 worker (trainer FSDP×8) on $NODE1"
srun --jobid="$SLURM_JOB_ID" --overlap \
    --nodes=1 --ntasks=1 --nodelist="$NODE1" --gres=gpu:8 --cpus-per-task=32 \
    --export=ALL,TTT_ROOT,RUN_ROOT,CONFIG,MODEL_NAME \
    --output="$LOGS/worker_node1_${NODE1}.out" --error="$LOGS/worker_node1_${NODE1}.err" \
    bash "$TTT_ROOT/scripts/_multitask_node1_worker.sh" &
W1_PID=$!

trap_handler() {
  echo "[run-mt] received signal, terminating workers (srun jobs $W0_PID $W1_PID)…"
  kill "$W0_PID" "$W1_PID" 2>/dev/null || true
  exit 130
}
trap trap_handler INT TERM

echo "[run-mt] giving workers 180s before launching orchestrator (32B base model + SGLang startup)…"
sleep 180

# ---- orchestrator ----------------------------------------------------------
# Resolve NODE1 to its IPv4 — /etc/hosts on these nodes maps the hostname to
# link-local IPv6 (no scope id), which makes PyTorch's TCPStore bind fail with
# errno 22 (EINVAL). DNS knows the real IPv4. host(1) bypasses /etc/hosts.
NODE1_IP=$(host -t A "$NODE1" 2>/dev/null | awk '/has address/{print $4; exit}')
if [ -z "$NODE1_IP" ]; then
  echo "[run-mt] ERROR: failed to resolve $NODE1 to an IPv4 address"
  exit 1
fi
echo "[run-mt] NODE1 IPv4 for weight-broadcast rendezvous: $NODE1 -> $NODE1_IP"

# The trainer (on NODE1) needs to be reachable by SGLang (on NODE0) at the
# multitask master_port (default 29600). We advertise NODE1's IPv4.
SGLANG_URL_FOR_TRAINER="http://${NODE0}:30100"
echo "[run-mt] starting multitask orchestrator (1 sampler, 4 envs, 1 trainer)"
"$PY" -u -m ttt_kernel.orchestrator.multitask \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --num-envs 4 \
    --sglang-url-for-trainer "$SGLANG_URL_FOR_TRAINER" \
    --trainer-master-addr "$NODE1_IP" \
    --log-level info \
    2>&1 | tee "$LOGS/orchestrator.log"

ORCH_RC=${PIPESTATUS[0]}
echo "[run-mt] orchestrator exited rc=$ORCH_RC, cleaning up workers"
kill "$W0_PID" "$W1_PID" 2>/dev/null || true
exit "$ORCH_RC"
