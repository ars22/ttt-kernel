#!/usr/bin/env bash
# Multi-node single-problem TTT (LoRA RL on one problem).
#   Node 0: SGLang tp=4 dp=1 --enable-lora (GPUs 0-3) + 4 env services (GPUs 4-7) + sampler
#   Node 1: ONE trainer (tp=4 FSDP on GPUs 0-3, LoRA)
#
# Orchestrator: 1 sampler, 4 envs, 1 trainer.
#
# Usage:
#   ./scripts/run_ttt_multinode_single.sh [config.yaml] [run_name]
set -euo pipefail

CONFIG="${1:-configs/ttt_qwen3_4b_singleprob.yaml}"
RUN_NAME="${2:-ttt_single_$(date +%Y%m%d_%H%M%S)}"
TTT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="$TTT_ROOT/runs/$RUN_NAME"
LOGS="$RUN_ROOT/logs"
mkdir -p "$LOGS" "$RUN_ROOT/registry/sampler" "$RUN_ROOT/registry/env" "$RUN_ROOT/registry/trainer"

PY="$TTT_ROOT/.venv/bin/python"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "[run-ttt-single] ERROR: no SLURM_JOB_ID — must run inside an allocation."
  exit 1
fi

NODE_LIST=$(scontrol show hostnames "$(squeue -h -j "$SLURM_JOB_ID" -O NodeList:200)")
NODES=($NODE_LIST)
if [ "${#NODES[@]}" -lt 2 ]; then
  echo "[run-ttt-single] ERROR: need 2 nodes; got: ${NODES[*]}"
  exit 1
fi
NODE0="${NODES[0]}"
NODE1="${NODES[1]}"

echo "[run-ttt-single] RUN_ROOT=$RUN_ROOT"
echo "[run-ttt-single] config=$CONFIG  model=$MODEL_NAME"
echo "[run-ttt-single] node0(sampler+envs)=$NODE0  node1(trainer)=$NODE1"

# Qwen3-4B's derived context_length is 40960 (< the 65536 default in the
# worker script). Cap at 16k — plenty for prompt (~2k) + max_tokens (8k).
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-16384}"

export TTT_ROOT RUN_ROOT CONFIG MODEL_NAME SGLANG_CONTEXT_LENGTH

export HF_HOME="/project/flame/asetlur/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="/project/flame/asetlur/.cache/torch"
export TRITON_CACHE_DIR="/project/flame/asetlur/.cache/triton"
export XDG_CACHE_HOME="/project/flame/asetlur/.cache"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR"

echo "[run-ttt-single] pre-warming HF datasets cache"
"$PY" - <<'PY' 2>&1 | tail -5
from datasets import load_dataset
load_dataset("ScalingIntelligence/KernelBench", split="level_1")
print("[prewarm] dataset cache ready")
PY

# ---- node0 worker (sampler+envs, reused from existing TTT path) -----------
echo "[run-ttt-single] launching node0 worker (sampler+envs) on $NODE0"
srun --jobid="$SLURM_JOB_ID" --overlap \
    --nodes=1 --ntasks=1 --nodelist="$NODE0" --gres=gpu:8 --cpus-per-task=32 \
    --export=ALL,TTT_ROOT,RUN_ROOT,CONFIG,MODEL_NAME,SGLANG_CONTEXT_LENGTH \
    --output="$LOGS/worker_node0_${NODE0}.out" --error="$LOGS/worker_node0_${NODE0}.err" \
    bash "$TTT_ROOT/scripts/_ttt_node0_worker.sh" &
W0_PID=$!

# ---- node1 worker (single trainer, LoRA mode) ------------------------------
echo "[run-ttt-single] launching node1 worker (1 trainer) on $NODE1"
srun --jobid="$SLURM_JOB_ID" --overlap \
    --nodes=1 --ntasks=1 --nodelist="$NODE1" --gres=gpu:8 --cpus-per-task=32 \
    --export=ALL,TTT_ROOT,RUN_ROOT,CONFIG,MODEL_NAME,SGLANG_CONTEXT_LENGTH \
    --output="$LOGS/worker_node1_${NODE1}.out" --error="$LOGS/worker_node1_${NODE1}.err" \
    bash "$TTT_ROOT/scripts/_ttt_node1_worker_single.sh" &
W1_PID=$!

trap_handler() {
  echo "[run-ttt-single] signal received, terminating workers ($W0_PID $W1_PID)…"
  kill "$W0_PID" "$W1_PID" 2>/dev/null || true
  exit 130
}
trap trap_handler INT TERM

echo "[run-ttt-single] giving workers 180s before launching orchestrator (4B + SGLang startup + materialize_seeds)…"
sleep 180

# Orchestrator: 1 sampler, 4 envs, 1 trainer. v000 seed adapter is created
# in-process (materialize_seeds runs on the orchestrator host).
echo "[run-ttt-single] starting orchestrator (1 sampler, 4 envs, 1 trainer)"
"$PY" -u -m ttt_kernel.orchestrator.main \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --num-samplers 1 \
    --num-envs 4 \
    --num-trainers 1 \
    --log-level info \
    2>&1 | tee "$LOGS/orchestrator.log"

ORCH_RC=${PIPESTATUS[0]}
echo "[run-ttt-single] orchestrator exited rc=$ORCH_RC, cleaning up workers"
kill "$W0_PID" "$W1_PID" 2>/dev/null || true
exit "$ORCH_RC"
