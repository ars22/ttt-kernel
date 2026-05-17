#!/usr/bin/env bash
# Launch the three-pool architecture inside one 8x B200 allocation.
#
# Uses GPUs 0-5 (the user's allocation may have GPUs 6-7 held by others).
# Backgrounds SGLang + sampler shim + trainer + 2 env services; runs the
# orchestrator in the foreground so ^C cleanly terminates the run.
#
# Usage:
#   ./scripts/run_single_node.sh [config.yaml] [run_name]
set -euo pipefail

CONFIG="${1:-configs/thinking_max_util.yaml}"
RUN_NAME="${2:-thinking_$(date +%Y%m%d_%H%M%S)}"
TTT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="$TTT_ROOT/runs/$RUN_NAME"
LOGS="$RUN_ROOT/logs"
mkdir -p "$LOGS" "$RUN_ROOT/registry/sampler" "$RUN_ROOT/registry/env" "$RUN_ROOT/registry/trainer"

# CUDA
if command -v module >/dev/null 2>&1; then
  module load cuda/13.0.2 || true
fi
export CUDA_HOME="${CUDA_HOME:-/apps/software/extern/cuda/13.0.2}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

PY="$TTT_ROOT/.venv/bin/python"
SGLANG_PY="/weka/scratch/schmidt/ssci-aviralku/asetlur/sglang_venv/bin/python"
TORCHRUN="$TTT_ROOT/.venv/bin/torchrun"

# Ports
SGLANG_PORT=30100
SAMPLER_PORT=8200
TRAINER_PORT=8300
ENV0_PORT=8100
ENV1_PORT=8101

# Adapter seed dir for SGLang launch (one placeholder so the server can boot).
SGLANG_SEED_ADAPTER="$RUN_ROOT/sglang_seed_adapter"
mkdir -p "$SGLANG_SEED_ADAPTER"

MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"

cd "$TTT_ROOT"

echo "[run] RUN_ROOT=$RUN_ROOT"
echo "[run] config=$CONFIG"
echo "[run] model=$MODEL_NAME"

# ---- 0. Seed the SGLang LoRA placeholder if missing -----------------------
if [ ! -f "$SGLANG_SEED_ADAPTER/adapter_config.json" ]; then
  echo "[run] seeding SGLang placeholder adapter at $SGLANG_SEED_ADAPTER"
  "$PY" - <<PY
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch, os
m = AutoModelForCausalLM.from_pretrained("$MODEL_NAME", torch_dtype=torch.bfloat16, trust_remote_code=True)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                 target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                 bias="none", task_type="CAUSAL_LM")
p = get_peft_model(m, cfg)
p.save_pretrained("$SGLANG_SEED_ADAPTER")
# Strip tokenizer files — SGLang's LoRA manager rejects adapter dirs containing them.
for j in ("added_tokens.json","tokenizer.json","tokenizer_config.json",
          "special_tokens_map.json","vocab.json","merges.txt","chat_template.jinja"):
    p2 = os.path.join("$SGLANG_SEED_ADAPTER", j)
    if os.path.exists(p2):
        os.remove(p2)
print("seeded")
PY
fi

# ---- 1. SGLang (GPUs 0,1, tp=2) -------------------------------------------
echo "[run] launching SGLang (GPUs 0,1, tp=2) → $LOGS/sglang.log"
CUDA_VISIBLE_DEVICES=0,1 nohup "$SGLANG_PY" -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$SGLANG_PORT" \
    --context-length 65536 \
    --dtype bfloat16 \
    --reasoning-parser qwen3 \
    --tp-size 2 \
    --mem-fraction-static 0.7 \
    --attention-backend triton \
    --trust-remote-code \
    --enable-lora \
    --max-loras-per-batch 24 \
    --max-lora-rank 32 \
    --lora-paths "ttt_seed=$SGLANG_SEED_ADAPTER" \
    > "$LOGS/sglang.log" 2>&1 &
SGLANG_PID=$!
echo "$SGLANG_PID" > "$RUN_ROOT/sglang.pid"

# ---- 2. Sampler shim -------------------------------------------------------
echo "[run] launching sampler shim → $LOGS/sampler.log"
nohup "$PY" -u -m ttt_kernel.sampler.server \
    --sglang-url "http://127.0.0.1:${SGLANG_PORT}" \
    --model "$MODEL_NAME" \
    --port "$SAMPLER_PORT" \
    --max-concurrent 16 \
    --max-loaded-adapters 24 \
    --wait-ready-s 1200 \
    --run-root "$RUN_ROOT" \
    --idx 0 \
    --advertise-host 127.0.0.1 \
    > "$LOGS/sampler.log" 2>&1 &
SAMPLER_PID=$!
echo "$SAMPLER_PID" > "$RUN_ROOT/sampler.pid"

# ---- 3. Trainer (FSDP=2 on GPUs 2,3) --------------------------------------
echo "[run] launching trainer FSDP=2 (GPUs 2,3) → $LOGS/trainer.log"
CUDA_VISIBLE_DEVICES=2,3 nohup "$TORCHRUN" --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29509 \
    -m ttt_kernel.trainer.server \
    --fsdp \
    --config "$CONFIG" \
    --port "$TRAINER_PORT" \
    --max-concurrent 1 \
    --max-resident-adapters 16 \
    --run-root "$RUN_ROOT" \
    --idx 0 \
    --advertise-host 127.0.0.1 \
    > "$LOGS/trainer.log" 2>&1 &
TRAINER_PID=$!
echo "$TRAINER_PID" > "$RUN_ROOT/trainer.pid"

# ---- 4. Env services (GPUs 4 and 5) ---------------------------------------
echo "[run] launching env service 0 (GPU 4, 12 slots) → $LOGS/env_0.log"
CUDA_VISIBLE_DEVICES=4 nohup "$PY" -u -m ttt_kernel.env.server \
    --config "$CONFIG" \
    --port "$ENV0_PORT" \
    --max-concurrent 12 \
    --sandbox-log "$LOGS/env_0.sandbox.log" \
    --run-root "$RUN_ROOT" \
    --idx 0 \
    --advertise-host 127.0.0.1 \
    > "$LOGS/env_0.log" 2>&1 &
echo "$!" > "$RUN_ROOT/env_0.pid"

echo "[run] launching env service 1 (GPU 5, 12 slots) → $LOGS/env_1.log"
CUDA_VISIBLE_DEVICES=5 nohup "$PY" -u -m ttt_kernel.env.server \
    --config "$CONFIG" \
    --port "$ENV1_PORT" \
    --max-concurrent 12 \
    --sandbox-log "$LOGS/env_1.sandbox.log" \
    --run-root "$RUN_ROOT" \
    --idx 1 \
    --advertise-host 127.0.0.1 \
    > "$LOGS/env_1.log" 2>&1 &
echo "$!" > "$RUN_ROOT/env_1.pid"

# ---- 5. Orchestrator (foreground) -----------------------------------------
trap_handler() {
  echo "[run] received signal, terminating background services…"
  for f in sglang sampler trainer env_0 env_1; do
    pid=$(cat "$RUN_ROOT/${f}.pid" 2>/dev/null || true)
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
  exit 130
}
trap trap_handler INT TERM

echo "[run] waiting 30s for services to start registering…"
sleep 30

echo "[run] starting orchestrator (foreground)"
exec "$PY" -u -m ttt_kernel.orchestrator.main \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --num-samplers 1 \
    --num-envs 2 \
    --num-trainers 1 \
    --log-level info \
    2>&1 | tee "$LOGS/orchestrator.log"
