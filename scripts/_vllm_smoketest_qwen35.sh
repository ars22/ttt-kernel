#!/usr/bin/env bash
# Smoke test: does Qwen3.5-4B produce coherent output via vLLM?
# Launches vllm serve on GPUs 0-1 (tp=2), waits for /v1/models, sends one
# chat completion, prints the result, kills the server.
set -uo pipefail

TTT_ROOT=/project/flame/asetlur/ttt-kernel
MODEL=Qwen/Qwen3.5-4B
PORT=30199
LOG=/tmp/vllm_smoke_$(hostname)_$$.log

# Same cache env as the worker (off /home quota, on /project NFS).
export HF_HOME="/project/flame/asetlur/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="/project/flame/asetlur/.cache/torch"
export TRITON_CACHE_DIR="/tmp/asetlur_triton_$(hostname)_$$"
export XDG_CACHE_HOME="/project/flame/asetlur/.cache"
mkdir -p "$TRITON_CACHE_DIR"

PY="$TTT_ROOT/.venv/bin/python"
export PATH="$TTT_ROOT/.venv/bin:$PATH"

echo "[smoke] launching vLLM on port $PORT, log $LOG"
CUDA_VISIBLE_DEVICES=0,1 nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    --gdn-prefill-backend triton \
    --trust-remote-code \
    > "$LOG" 2>&1 &
VLLM_PID=$!
echo "[smoke] vllm pid=$VLLM_PID"

trap 'echo "[smoke] cleaning up vllm pid=$VLLM_PID"; kill -TERM $VLLM_PID 2>/dev/null || true; sleep 5; kill -KILL $VLLM_PID 2>/dev/null || true' EXIT

# Poll /v1/models up to 15 minutes (cold-start downloads + hybrid arch init).
echo "[smoke] waiting for /v1/models (max 900s)"
for i in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[smoke] vLLM ready after ${i}*5s"
        break
    fi
    sleep 5
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[smoke] FATAL: vllm process died. Last 60 lines:"
        tail -60 "$LOG"
        exit 1
    fi
done

if ! curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[smoke] FATAL: never became ready. Last 80 lines:"
    tail -80 "$LOG"
    exit 1
fi

echo "[smoke] === chat completion test ==="
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Write a one-paragraph Python function that returns the sum of two ints.\"}],
        \"temperature\": 0.7,
        \"top_p\": 0.95,
        \"max_tokens\": 256
    }" | tee /tmp/vllm_smoke_response.json

echo
echo "[smoke] === extracted content ==="
"$PY" -c "import json; d=json.load(open('/tmp/vllm_smoke_response.json')); print(d['choices'][0]['message']['content'])"

echo "[smoke] === server log tail (last 40 lines) ==="
tail -40 "$LOG"
