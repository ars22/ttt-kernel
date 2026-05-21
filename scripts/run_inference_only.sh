#!/usr/bin/env bash
# Inference-only run: sample + evaluate, no training.
#
# GPU layout:
#   0-5  SGLang (tp=3, dp=2, gpt-oss-20b)
#   6    env service 0
#   7    env service 1
#
# Usage:
#   ./scripts/run_inference_only.sh [config.yaml] [run_name]
set -euo pipefail

CONFIG="${1:-configs/inference_only_gpt_oss_20b.yaml}"
RUN_NAME="${2:-infer_$(date +%Y%m%d_%H%M%S)}"
TTT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="$TTT_ROOT/runs/$RUN_NAME"
LOGS="$RUN_ROOT/logs"
mkdir -p "$LOGS" "$RUN_ROOT/registry/sampler" "$RUN_ROOT/registry/env"

# CUDA
if command -v module >/dev/null 2>&1; then
  module load cuda/13.0.2 || true
fi
export CUDA_HOME="${CUDA_HOME:-/apps/software/extern/cuda/13.0.2}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

PY="$TTT_ROOT/.venv/bin/python"
SGLANG_PY="$TTT_ROOT/.venv/bin/python"

# Ports
SGLANG_PORT=30100
SAMPLER_PORT=8200
ENV0_PORT=8100
ENV1_PORT=8101

MODEL_NAME="openai/gpt-oss-20b"

SGLANG_SEED_ADAPTER="$RUN_ROOT/sglang_seed_adapter"
mkdir -p "$SGLANG_SEED_ADAPTER"

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
for j in ("added_tokens.json","tokenizer.json","tokenizer_config.json",
          "special_tokens_map.json","vocab.json","merges.txt","chat_template.jinja"):
    p2 = os.path.join("$SGLANG_SEED_ADAPTER", j)
    if os.path.exists(p2):
        os.remove(p2)
print("seeded")
PY
fi

# ---- 1. SGLang (GPUs 0-5, tp=6) -------------------------------------------
echo "[run] launching SGLang (GPUs 0-5, tp=3 dp=2) → $LOGS/sglang.log"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup "$SGLANG_PY" -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$SGLANG_PORT" \
    --context-length 65536 \
    --dtype bfloat16 \
    --tp-size 3 \
    --dp-size 2 \
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

# ---- 3. Env services (GPUs 6 and 7) ----------------------------------------
echo "[run] launching env service 0 (GPU 6) → $LOGS/env_0.log"
CUDA_VISIBLE_DEVICES=6 nohup "$PY" -u -m ttt_kernel.env.server \
    --config "$CONFIG" \
    --port "$ENV0_PORT" \
    --max-concurrent 12 \
    --sandbox-log "$LOGS/env_0.sandbox.log" \
    --run-root "$RUN_ROOT" \
    --idx 0 \
    --advertise-host 127.0.0.1 \
    > "$LOGS/env_0.log" 2>&1 &
echo "$!" > "$RUN_ROOT/env_0.pid"

echo "[run] launching env service 1 (GPU 7) → $LOGS/env_1.log"
CUDA_VISIBLE_DEVICES=7 nohup "$PY" -u -m ttt_kernel.env.server \
    --config "$CONFIG" \
    --port "$ENV1_PORT" \
    --max-concurrent 12 \
    --sandbox-log "$LOGS/env_1.sandbox.log" \
    --run-root "$RUN_ROOT" \
    --idx 1 \
    --advertise-host 127.0.0.1 \
    > "$LOGS/env_1.log" 2>&1 &
echo "$!" > "$RUN_ROOT/env_1.pid"

# ---- 4. Orchestrator (foreground, inference-only) --------------------------
trap_handler() {
  echo "[run] received signal, terminating background services…"
  for f in sglang sampler env_0 env_1; do
    pid=$(cat "$RUN_ROOT/${f}.pid" 2>/dev/null || true)
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
  exit 130
}
trap trap_handler INT TERM

echo "[run] giving services 10s to advertise to the registry…"
sleep 10

echo "[run] starting orchestrator (inference-only, foreground)"
exec "$PY" -u -m ttt_kernel.orchestrator.main \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --num-samplers 1 \
    --num-envs 2 \
    --num-trainers 0 \
    --inference-only \
    --log-level info \
    2>&1 | tee "$LOGS/orchestrator.log"
