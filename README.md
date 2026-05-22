# ttt-kernel

> **NOTE (refactor branch):** this branch is being rewritten into a
> three-pool (sampler / trainer / env) architecture with a central
> orchestrator. The instructions below describe the OLD pair-orchestrator
> layout (still on `main`) and are stale. The new entry points
> (`orchestrator/main.py`, SLURM launchers under `scripts/`) land
> incrementally; see `/home/schmidt/ssci-asetlur/.claude/plans/so-we-need-to-moonlit-flame.md`
> for the design.

Multi-turn online RL with LoRA over [KernelBench](https://github.com/ScalingIntelligence/KernelBench) problems.

For each problem in KernelBench we:

1. Take a thinking-style policy model served by SGLang (default `Qwen/Qwen3-4B-Thinking-2507`).
2. Sample **K parallel attempts** (default 8) at a CUDA kernel.
3. Score every attempt against the PyTorch reference: a speedup-shaped reward + a textual feedback string (stack trace, correctness diffs, perf numbers).
4. Run one **GRPO step on a LoRA adapter** using those K rollouts as a group.
5. Hot-swap the updated adapter into the SGLang server and start the next turn.

Default loop is **5 turns per problem**. Everything is configurable.

## Layout

```
ttt-kernel/
├── configs/default.yaml       # all hyperparameters live here
├── scripts/
│   ├── inference.py           # ★ THE inference script — run this
│   └── launch_sglang.sh       # start the SGLang policy server
├── src/ttt_kernel/
│   ├── config.py              # typed YAML loader + CLI overrides
│   ├── kernel_env.py          # KernelBench problem → reward + feedback
│   ├── sglang_client.py       # N-parallel sampling + LoRA hot-reload
│   ├── grpo_trainer.py        # PEFT LoRA + GRPO loss + optimizer step
│   ├── loop.py                # the orchestration loop (the actual TTT)
│   └── logger.py              # JSONL event log
├── pyproject.toml
└── README.md
```

## The inference script

The "inference script" — what makes this *online* RL — is
[`scripts/inference.py`](scripts/inference.py) which dispatches into
[`src/ttt_kernel/loop.py:run()`](src/ttt_kernel/loop.py). That function is the
whole shebang: it owns the trainer (HF model + LoRA + AdamW in-process), the
SGLang client (for K-way sampling), and the KernelBench env (for rewards). Per
problem it runs `loop.num_turns` turns; per turn it samples K times, scores,
GRPO-updates, and re-pushes the LoRA into SGLang.

## Setup

```bash
# 1. Clone alongside KernelBench
cd /weka/scratch/schmidt/ssci-aviralku/asetlur
git clone <this-repo-url> ttt-kernel
cd ttt-kernel

# 2. Env (uv is already on PATH on this cluster)
uv venv --python 3.10
uv pip install -e .
uv pip install "sglang[all]>=0.4"

# 3. Make sure KernelBench is cloned and the env var points at it
export KERNELBENCH_PATH=/weka/scratch/schmidt/ssci-aviralku/asetlur/KernelBench
# Also: KernelBench's deps must already be installed in this venv OR added to sys.path.
# Easiest: `uv pip install -e $KERNELBENCH_PATH`

# 4. B200 needs nvcc 13 for KernelBench JIT compilation
module load cuda/13.0.2
export CUDA_HOME=/apps/software/extern/cuda/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
```

## Run

Two processes — SGLang serves the model; the trainer drives the loop.

**Terminal A — start SGLang** (uses 1 GPU by default):

```bash
cd /weka/scratch/schmidt/ssci-aviralku/asetlur/ttt-kernel
MODEL_NAME=Qwen/Qwen3-4B-Thinking-2507 \
ADAPTER_DIR=$PWD/adapters/ttt \
PORT=30000 \
bash scripts/launch_sglang.sh
```

This boots SGLang, mounts an initially-zero LoRA at the `ttt` slot, and listens on `http://127.0.0.1:30000`.

**Terminal B — start the online-RL loop**:

```bash
cd /weka/scratch/schmidt/ssci-aviralku/asetlur/ttt-kernel
export SGLANG_BASE_URL=http://127.0.0.1:30000
export KERNELBENCH_PATH=/weka/scratch/schmidt/ssci-aviralku/asetlur/KernelBench

python scripts/inference.py --config configs/default.yaml
```

## Common knobs

All overridable on the CLI (dotted-key=value); the defaults below are in `configs/default.yaml`:

| Knob | Default | Meaning |
|---|---|---|
| `model.name` | `Qwen/Qwen3-4B-Thinking-2507` | Base policy (HF id) |
| `lora.r` | 16 | LoRA rank |
| `lora.alpha` | 32 | LoRA alpha |
| `rollout.num_samples` | 8 | K parallel attempts per turn |
| `rollout.temperature` | 1.0 | Sampling temperature |
| `loop.num_turns` | 5 | Turns per problem |
| `loop.persist_adapter_across_problems` | false | Reset LoRA between problems |
| `kernelbench.level` | 1 | KernelBench level (1/2/3/4) |
| `kernelbench.problem_ids` | null (= all) | E.g. `[1,2,3]` |
| `grpo.learning_rate` | 1e-5 | AdamW LR |
| `grpo.beta_kl` | 0.04 | KL coeff vs reference (adapter-disabled base) |
| `grpo.epsilon_clip` | 0.2 | PPO-style importance-ratio clip |
| `reward.speedup_log_scale` | true | log(speedup) so 2× and 0.5× are symmetric |
| `reward.error_penalty` | -1.0 | Reward for parse/compile error |
| `reward.incorrect_penalty` | -1.0 | Reward for compiled-but-wrong kernel |

Example: drop to 2 turns and rank-4 LoRA, run only problem 7:

```bash
python scripts/inference.py --config configs/default.yaml \
    loop.num_turns=2 lora.r=4 kernelbench.problem_ids=[7]
```

## Outputs

A run writes to `./runs/<run_name>/`:

- `events.jsonl` — every rollout, every turn summary, every train metric. One JSON per line.
- `adapters/ttt/` — the latest LoRA adapter (also where SGLang reads from).

## Reward shape

For one rollout we look at `(compiled, correct, speedup)` and produce one scalar:

- **parse / harness error** → `reward.error_penalty` (default `-1`).
- **compiled but incorrect** → `reward.incorrect_penalty` (default `-1`).
- **correct** → `log(speedup)` if `reward.speedup_log_scale` else `speedup - 1`, clipped to `±reward.clip`.

So a 2× speedup → `+0.69`; a 0.5× kernel (twice as slow as PyTorch) → `-0.69`. The advantage in GRPO is then `(reward − group_mean) / (group_std + ε)`.

## Where to read next

- `src/ttt_kernel/loop.py` — the loop itself, ~80 lines, reads top-to-bottom.
- `src/ttt_kernel/grpo_trainer.py` — GRPO loss + the LoRA save path.
- `src/ttt_kernel/kernel_env.py::KernelEnv.evaluate` — how a completion becomes a reward.
- `src/ttt_kernel/sglang_client.py::SGLangClient.reload_adapter_async` — the hot-swap call.

## Notes and gotchas

- KernelBench JIT-compiles CUDA per attempt. On the cluster B200 nodes, **`module load cuda/13.0.2`** before launching either process — `/usr/local/cuda` is 12.8 and can't target Blackwell's `compute_103`.
- Whatever `model.name` you pick must have a chat/text-completion template compatible with how KernelBench writes its prompts (raw text). We use `/v1/completions` so a base/instruct model both work; if you swap to chat, use `/v1/chat/completions` and template the prompt yourself.
- KernelBench's eval can hang on pathological kernels. Today we rely on the model not emitting infinite loops; if that becomes a real problem add a subprocess timeout around `KernelEnv.evaluate` (the harness already catches exceptions, just not hangs).
- The trainer holds the model in VRAM **in addition to** SGLang's copy. With `Qwen3-4B-Thinking` in bf16 that's ~8GB per process; one B200 fits both comfortably. Bump `model.dtype=float16` if you're tight.
