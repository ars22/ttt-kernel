"""Trainer-worker subprocess for the pool architecture.

One worker process == one (trainer, sampler) pair from the perspective of the
trainer side. It owns:

  - a single PEFT model + optimizer (no DDP; one GPU per worker by default)
  - a SGLangClient pointed at this worker's paired SGLang server URL
  - its own LoRA adapter directory (so concurrent pairs don't overwrite each
    other's adapters)

Protocol: the orchestrator writes one JSON line to our stdin per message; we
write one JSON line per event over a *dedicated* JSON channel (the fd that was
fd 1 when we started). At startup we redirect Python's sys.stdout to stderr
so ad-hoc `print()` / progress bars / library logging never pollute the JSON
stream. We emit `{"kind":"ready",...}` once, then for each problem assignment
one `kind:"turn"` per turn and one `kind:"done"` at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# --- Isolate the JSON channel ------------------------------------------------
# Save fd 1 (real stdout, parent's pipe), then redirect Python's sys.stdout
# and the underlying fd 1 to fd 2 (stderr → captured in worker_pair*.log). All
# subsequent print() / tqdm / library output goes to stderr; only _emit() goes
# back to the parent's JSON pipe via _JSON_OUT.
_JSON_FD = os.dup(1)
os.dup2(2, 1)
_JSON_OUT = os.fdopen(_JSON_FD, "w", buffering=1)
sys.stdout = sys.stderr

from .config import load_config
from .grpo_trainer import GRPOLoRATrainer, Rollout
from .kernel_env import KernelEnv
from .sglang_client import SGLangClient


def _emit(obj: dict) -> None:
    _JSON_OUT.write(json.dumps(obj, default=str) + "\n")
    _JSON_OUT.flush()


def _run_one_problem(
    problem_id: int,
    cfg,
    trainer: GRPOLoRATrainer,
    env: KernelEnv,
    sglang: SGLangClient,
    rollout_dir: Path,
    pair_idx: int,
) -> dict:
    if not cfg.loop.persist_adapter_across_problems:
        trainer.reset_adapter()
        trainer.save_adapter(cfg.sglang.adapter_out_dir)
        sglang.reload_adapter()

    problem = env.get_problem(problem_id)
    K = cfg.rollout.num_samples

    best_reward = float("-inf")
    best_speedup = 0.0
    any_correct = False

    for turn in range(cfg.loop.num_turns):
        gens = sglang.sample(
            prompt=problem.prompt,
            n=K,
            temperature=cfg.rollout.temperature,
            top_p=cfg.rollout.top_p,
            max_tokens=cfg.rollout.max_tokens,
            use_adapter=True,
        )
        results = [env.evaluate(problem, g.text) for g in gens]

        for k, r in enumerate(results):
            rec = {
                "problem_id": problem_id,
                "problem_name": problem.name,
                "turn": turn,
                "sample": k,
                "pair": pair_idx,
                "reward": r.reward,
                "compiled": r.compiled,
                "correct": r.correct,
                "speedup": r.speedup,
                "error_kind": r.error_kind,
                "runtime_ms": r.runtime_ms,
                "ref_runtime_ms": r.ref_runtime_ms,
                "feedback": r.feedback,
                "kernel_src": r.kernel_src,
                "completion": r.raw_completion,
                "prompt": problem.prompt,
            }
            fname = f"p{problem_id:03d}_t{turn}_s{k:02d}.json"
            with open(rollout_dir / fname, "w") as f:
                json.dump(rec, f, ensure_ascii=False)

        ros = [Rollout(prompt=problem.prompt, completion=r.raw_completion, reward=r.reward)
               for r in results]
        train_metrics = trainer.step(ros)

        rewards = [r.reward for r in results]
        speedups = [r.speedup for r in results if r.error_kind == "ok"]
        n_compiled = sum(1 for r in results if r.compiled)
        n_correct = sum(1 for r in results if r.correct)
        comp_tokens = [int(g.completion_tokens) for g in gens if g.completion_tokens]
        n_truncated = sum(1 for g in gens if g.finish_reason == "length")

        if rewards:
            mr = max(rewards)
            if mr > best_reward:
                best_reward = mr
        if speedups:
            ms = max(speedups)
            if ms > best_speedup:
                best_speedup = ms
        if n_correct > 0:
            any_correct = True

        _emit({
            "kind": "turn",
            "pair": pair_idx,
            "problem_id": problem_id,
            "problem_name": problem.name,
            "turn": turn,
            "n_rollouts": len(results),
            "n_compiled": n_compiled,
            "n_correct": n_correct,
            "frac_compiled": n_compiled / max(len(results), 1),
            "frac_correct": n_correct / max(len(results), 1),
            "reward_mean": sum(rewards) / max(len(rewards), 1),
            "reward_max": max(rewards) if rewards else 0.0,
            "reward_min": min(rewards) if rewards else 0.0,
            "speedup_mean": (sum(speedups) / len(speedups)) if speedups else 0.0,
            "speedup_max": max(speedups) if speedups else 0.0,
            "completion_tokens_mean": (sum(comp_tokens) / len(comp_tokens)) if comp_tokens else 0.0,
            "completion_tokens_max": max(comp_tokens) if comp_tokens else 0,
            "completion_tokens_min": min(comp_tokens) if comp_tokens else 0,
            "n_truncated": n_truncated,
            **{f"train_{k}": v for k, v in train_metrics.items()},
        })

        if cfg.logging.save_adapter_every_turn or turn == cfg.loop.num_turns - 1:
            trainer.save_adapter(cfg.sglang.adapter_out_dir)
            sglang.reload_adapter()

    return {
        "best_reward": best_reward if best_reward != float("-inf") else 0.0,
        "best_speedup": best_speedup,
        "any_correct": any_correct,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--pair-idx", type=int, required=True)
    ap.add_argument("--sglang-url", required=True)
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    # Single-process trainer; defensively clear distributed env so the
    # GRPO trainer's _init_distributed returns (0, 1, 0).
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        os.environ.pop(k, None)

    cfg = load_config(args.config, overrides=args.override)
    cfg.sglang.base_url = args.sglang_url
    cfg.sglang.adapter_out_dir = args.adapter_dir

    try:
        trainer = GRPOLoRATrainer(cfg.model, cfg.lora, cfg.grpo)
        env = KernelEnv(cfg.kernelbench, cfg.reward)
        sglang = SGLangClient(cfg.sglang, model_name=cfg.model.name)
        sglang.wait_ready(timeout_s=1800.0)
    except Exception as e:
        _emit({"kind": "fatal", "pair": args.pair_idx, "error": str(e),
               "traceback": traceback.format_exc()})
        raise

    rollout_dir = Path(args.run_dir) / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    _emit({"kind": "ready", "pair": args.pair_idx})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            _emit({"kind": "error", "pair": args.pair_idx, "error": f"bad-json: {line[:200]}"})
            continue

        cmd = msg.get("cmd")
        if cmd == "exit":
            break
        if cmd != "process":
            _emit({"kind": "error", "pair": args.pair_idx, "error": f"unknown cmd: {cmd}"})
            continue

        problem_id = msg["problem_id"]
        try:
            summary = _run_one_problem(
                problem_id, cfg, trainer, env, sglang, rollout_dir, args.pair_idx,
            )
            _emit({"kind": "done", "pair": args.pair_idx, "problem_id": problem_id, **summary})
        except Exception as e:
            # Any exception that escapes _run_one_problem (CUDA illegal access,
            # OOM, RuntimeError, etc.) tends to poison the CUDA context —
            # subsequent problems would all fail in microseconds. Emit `done`
            # with the error AND exit. The orchestrator detects EOF on stdout,
            # requeues the problem, and stops dispatching to this worker.
            _emit({"kind": "done", "pair": args.pair_idx, "problem_id": problem_id,
                   "error": str(e), "traceback": traceback.format_exc(),
                   "worker_exiting": True})
            sys.exit(2)


if __name__ == "__main__":
    main()
