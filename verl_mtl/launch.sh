#!/usr/bin/env bash
# Single-node verl GRPO on KernelBench (Qwen3-4B).
#
# Layout (8 B200 GPUs):
#   GPUs 0-5  → verl (co-located FSDP=6 trainer + SGLang TP=2 DP=3 rollout)
#   GPUs 6,7  → 2x ttt-kernel env services (kernel eval) on ports 8100,8101
#
# Verl co-locates trainer + rollout on the same GPUs via Ray actor scheduling.
# TP=2 DP=3 spreads the 64 concurrent decode requests (4 problems × K=16) across
# 3 model replicas, avoiding the KV-cache bottleneck of a single TP=4 replica.
set -euo pipefail

TTT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-${TTT_ROOT}/verl_mtl/configs/grpo_qwen3_4b.yaml}"
RUN_NAME="${2:-verl_mtl_$(date +%Y%m%d_%H%M%S)}"
# Drop the two positional args we already consumed so "$@" forwards only
# trailing hydra overrides (if any) to verl.
shift 2 2>/dev/null || true
RUN_ROOT="${TTT_ROOT}/runs/${RUN_NAME}"
LOGS="${RUN_ROOT}/logs"
mkdir -p "${LOGS}"

# CUDA 13 for B200.
if command -v module >/dev/null 2>&1; then
  module load cuda/13.0.2 || true
fi
export CUDA_HOME="${CUDA_HOME:-/apps/software/extern/cuda/13.0.2}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

PY="${TTT_ROOT}/.venv-verl/bin/python"
# The env services live in the original .venv (they're pure ttt-kernel code).
ENV_PY="${TTT_ROOT}/.venv/bin/python"

# Per-eval sandbox memory cap (32 slots/env, 0.15 fraction = ~27 GB max per slot).
export TTT_SANDBOX_MEM_FRACTION="${TTT_SANDBOX_MEM_FRACTION:-0.15}"
# expandable_segments helps eval sandboxes but breaks SGLang's TorchMemorySaver.
# Set it only for the env services (inline below), not globally.
ENV_ALLOC_CONF="expandable_segments:True"

cd "${TTT_ROOT}"

echo "[verl-mtl] RUN_ROOT=${RUN_ROOT}"
echo "[verl-mtl] config=${CONFIG}"

# ---- env services (GPUs 6 & 7) ---------------------------------------------
ENV0_PORT=8100
ENV1_PORT=8101
# Reusing ttt-kernel's env service. It needs a kernelbench config with the
# repo_path / level / arch fields — we hand it a stub yaml that has just that
# section. The trainer / sampler sections of the same yaml are ignored by the
# env service.
ENV_CFG="${TTT_ROOT}/verl_mtl/configs/env_kernelbench.yaml"

# Host virtual-memory cap per env service subtree (in KB). Inherited by all
# sandbox subprocesses. Surfaces a runaway allocation as a catchable
# MemoryError instead of a silent SIGKILL. 500 GB is intentionally generous —
# torch+CUDA mmap a lot of libs; tighter caps will OOM legitimate work.
ENV_VMEM_KB=$((500 * 1024 * 1024))

echo "[verl-mtl] launching env_0 (GPU 6, port ${ENV0_PORT})"
( ulimit -v "${ENV_VMEM_KB}"
  PYTORCH_CUDA_ALLOC_CONF="${ENV_ALLOC_CONF}" CUDA_VISIBLE_DEVICES=6 nohup "${ENV_PY}" -u -m ttt_kernel.env.server \
      --config "${ENV_CFG}" \
      --port "${ENV0_PORT}" \
      --max-concurrent 16 \
      --sandbox-log "${LOGS}/env_0.sandbox.log" \
      --run-root "${RUN_ROOT}" \
      --idx 0 \
      --advertise-host 127.0.0.1 \
      > "${LOGS}/env_0.log" 2>&1 &
  echo "$!" > "${RUN_ROOT}/env_0.pid"
)

echo "[verl-mtl] launching env_1 (GPU 7, port ${ENV1_PORT})"
( ulimit -v "${ENV_VMEM_KB}"
  PYTORCH_CUDA_ALLOC_CONF="${ENV_ALLOC_CONF}" CUDA_VISIBLE_DEVICES=7 nohup "${ENV_PY}" -u -m ttt_kernel.env.server \
      --config "${ENV_CFG}" \
      --port "${ENV1_PORT}" \
      --max-concurrent 16 \
      --sandbox-log "${LOGS}/env_1.sandbox.log" \
      --run-root "${RUN_ROOT}" \
      --idx 1 \
      --advertise-host 127.0.0.1 \
      > "${LOGS}/env_1.log" 2>&1 &
  echo "$!" > "${RUN_ROOT}/env_1.pid"
)

trap_handler() {
  echo "[verl-mtl] received signal, killing env services…"
  for f in env_0 env_1; do
    pid=$(cat "${RUN_ROOT}/${f}.pid" 2>/dev/null || true)
    [ -n "${pid}" ] && kill "${pid}" 2>/dev/null || true
  done
  exit 130
}
trap trap_handler INT TERM

# Wait for envs to come up. /healthz returns 200 once the FastAPI lifespan
# finishes (which includes spawning all sandbox subprocesses). We only need
# HTTP 200 — the JSON body's exact spacing varies by FastAPI version.
echo "[verl-mtl] waiting up to 10 min for env services to flip healthy…"
ok=0
for i in $(seq 1 120); do
  s0=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${ENV0_PORT}/healthz" 2>/dev/null || echo 0)
  s1=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${ENV1_PORT}/healthz" 2>/dev/null || echo 0)
  if [ "${s0}" = "200" ] && [ "${s1}" = "200" ]; then ok=1; break; fi
  sleep 5
done
if [ "${ok}" != "1" ]; then
  echo "[verl-mtl] ERROR: env services not healthy within 10 min (env_0=${s0} env_1=${s1})"; trap_handler
fi
echo "[verl-mtl] env services healthy"

# Tell the reward fn where to send /evaluate.
export TTT_ENV_URLS="http://127.0.0.1:${ENV0_PORT},http://127.0.0.1:${ENV1_PORT}"

# ---- verl entrypoint --------------------------------------------------------
DATA_DIR="${TTT_ROOT}/verl_mtl/data"
TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/val.parquet"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"

if [ ! -f "${TRAIN_FILE}" ]; then
  echo "[verl-mtl] dataset missing; running prepare_dataset.py"
  "${PY}" "${TTT_ROOT}/verl_mtl/prepare_dataset.py" \
      --kernelbench-repo "/weka/scratch/schmidt/ssci-aviralku/asetlur/KernelBench" \
      --out-dir "${DATA_DIR}" \
      --level 1
fi

REWARD_PATH="${TTT_ROOT}/verl_mtl/reward.py"

echo "[verl-mtl] starting verl trainer"
exec "${PY}" -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=6 \
    data.max_prompt_length=4096 \
    data.max_response_length=16000 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=16 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_custom_all_reduce=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20000 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20000 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.reward_manager=naive \
    custom_reward_function.path="${REWARD_PATH}" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=ttt-kernel \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=50 \
    trainer.default_local_dir="${RUN_ROOT}/ckpts" \
    "$@" \
    2>&1 | tee "${LOGS}/verl.log"
