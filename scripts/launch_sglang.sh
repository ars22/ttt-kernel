#!/usr/bin/env bash
# Launch SGLang with the policy model + a writable LoRA adapter slot.
#
# This is intended to run on a B200 node. The cuda module load matches what
# we use for KernelBench JIT compilation; SGLang itself doesn't need nvcc.
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
ADAPTER_NAME="${ADAPTER_NAME:-ttt}"
ADAPTER_DIR="${ADAPTER_DIR:-$(pwd)/adapters/ttt}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
TP="${TP:-1}"          # tensor parallel
DTYPE="${DTYPE:-bfloat16}"

# Make sure the adapter dir exists (even if empty) before launch.
mkdir -p "$ADAPTER_DIR"

# Seed the adapter slot with a placeholder LoRA so SGLang can start.
# We do this in-process from python so we don't need a separate util script.
if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
  python - <<PY
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch, os
m = AutoModelForCausalLM.from_pretrained("$MODEL_NAME", torch_dtype=torch.bfloat16, trust_remote_code=True)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                 target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                 bias="none", task_type="CAUSAL_LM")
p = get_peft_model(m, cfg)
p.save_pretrained("$ADAPTER_DIR")
print("seeded placeholder adapter at $ADAPTER_DIR")
PY
fi

exec python -m sglang.launch_server \
  --model-path "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --tp "$TP" \
  --trust-remote-code \
  --lora-paths "${ADAPTER_NAME}=${ADAPTER_DIR}" \
  --enable-lora
