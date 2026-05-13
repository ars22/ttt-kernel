#!/usr/bin/env bash
# Launch SGLang with the policy model + a writable LoRA adapter slot.
#
# This is intended to run on a B200 node. The cuda module load matches what
# we use for KernelBench JIT compilation; SGLang itself doesn't need nvcc.
set -euo pipefail

# Make sure nvcc/CUDA target Blackwell. On this cluster /usr/local/cuda is 12.8
# and rejects compute_100a (sm_100a) — needed for any JIT compile flashinfer/sglang
# does at startup. Load CUDA 13.0.2 module and pin CUDA_HOME so torch.cpp_extension
# and flashinfer both pick the right nvcc.
if command -v module >/dev/null 2>&1; then
  module load cuda/13.0.2 || true
fi
export CUDA_HOME="${CUDA_HOME_OVERRIDE:-/apps/software/extern/cuda/13.0.2}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
ADAPTER_NAME="${ADAPTER_NAME:-ttt}"
ADAPTER_DIR="${ADAPTER_DIR:-$(pwd)/adapters/ttt}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
TP="${TP:-1}"          # tensor parallel
DP="${DP:-1}"          # data parallel (replicas)
DTYPE="${DTYPE:-bfloat16}"
# CUDA_VISIBLE_DEVICES is honored if exported by the caller; SGLang will see
# only those cards and round-robin replicas across them.

# Make sure the adapter dir exists (even if empty) before launch.
mkdir -p "$ADAPTER_DIR"

# Seed the adapter slot with a placeholder LoRA so SGLang can start.
# We do this in-process from python so we don't need a separate util script.
# CRITICAL: do NOT save the tokenizer alongside the adapter. SGLang's LoRA
# manager rejects adapter dirs containing added_tokens.json.
if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
  rm -f "$ADAPTER_DIR"/added_tokens.json "$ADAPTER_DIR"/tokenizer*.json \
        "$ADAPTER_DIR"/special_tokens_map.json "$ADAPTER_DIR"/vocab.json \
        "$ADAPTER_DIR"/merges.txt "$ADAPTER_DIR"/chat_template.jinja 2>/dev/null
  python - <<PY
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained("$MODEL_NAME", torch_dtype=torch.bfloat16, trust_remote_code=True)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                 target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                 bias="none", task_type="CAUSAL_LM")
p = get_peft_model(m, cfg)
p.save_pretrained("$ADAPTER_DIR")
print("seeded placeholder adapter at $ADAPTER_DIR")
PY
fi

ATTN_BACKEND="${ATTN_BACKEND:-triton}"  # avoid flashinfer JIT against system nvcc

exec python -m sglang.launch_server \
  --model-path "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --tp "$TP" \
  --dp "$DP" \
  --attention-backend "$ATTN_BACKEND" \
  --trust-remote-code \
  --lora-paths "${ADAPTER_NAME}=${ADAPTER_DIR}" \
  --enable-lora
