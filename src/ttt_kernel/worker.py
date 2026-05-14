"""Trainer-worker subprocess for the pool architecture.

Runs as one process per (sampler, trainer) pair. Handles MULTIPLE in-flight
problems concurrently via asyncio:

  - One per-problem coroutine drives num_turns of (sample → eval → step → swap)
    for one problem, with its own LoRA adapter on disk and on SGLang.
  - The trainer GPU is shared; a single `trainer_lock` serializes the GRPO
    step across problems (PEFT's active adapter is global state).
  - Eval sandboxes are per-problem (kernel_env.open_sandbox(pid)); nvcc compiles
    happen on CPU, so multiple sandboxes' compiles overlap.
  - SGLang calls are unlocked: the same server handles K rollouts × P problems
    in one continuous-batching scheduler.

Protocol (one JSON line per message in/out, over stdin/stdout):
  in  : {"cmd": "process", "problem_id": int}
        {"cmd": "exit"}
  out : {"kind": "ready", "pair": int}
        {"kind": "turn", "problem_id": int, "turn": int, ...metrics...}
        {"kind": "done", "problem_id": int, ...summary...}
        {"kind": "error" | "fatal", ...}

fd 1 is reserved for JSON; sys.stdout + the underlying fd 1 are redirected to
stderr at startup so library prints never corrupt the protocol channel.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

# --- Isolate the JSON channel ------------------------------------------------
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


async def _run_one_problem(
    problem_id: int,
    pair_idx: int,
    cfg,
    trainer: GRPOLoRATrainer,
    env: KernelEnv,
    sglang: SGLangClient,
    rollout_dir: Path,
    trainer_lock: asyncio.Lock,
) -> None:
    """Drive one problem through num_turns of online RL.

    Owns its own adapter (created here, deleted at the end) and its own eval
    sandbox subprocess. Concurrent calls with other problem_ids share the
    trainer model (serialized via `trainer_lock`) and the SGLang server.
    """
    adapter_name = f"ttt_pair{pair_idx}_pid{problem_id:03d}"
    # PEFT.save_pretrained(parent, selected_adapters=[name]) writes the
    # adapter to `parent/<name>/` (a NESTED subdir, not the parent itself).
    # SGLang's /load_lora_adapter wants the directory that contains
    # adapter_config.json + adapter_model.safetensors directly — i.e. the
    # nested path. So we pass `parent` to save_adapter and `parent/name` to
    # SGLang for everything that follows.
    adapter_parent = os.path.abspath(f"{cfg.sglang.adapter_out_dir}_pid{problem_id:03d}")
    adapter_dir = os.path.join(adapter_parent, adapter_name)

    # --- per-problem setup ---------------------------------------------------
    # Trainer-side adapter creation must be serialized: PEFT mutates shared
    # model state when add_adapter is called (it walks every layer).
    async with trainer_lock:
        trainer.add_problem_adapter(adapter_name)
        trainer.save_adapter(adapter_name, adapter_parent)
    # Register the adapter with SGLang so we can sample with it. Done outside
    # the trainer lock because it's a remote HTTP call, not local CUDA work.
    try:
        await sglang.load_adapter_async(adapter_name, adapter_dir)
    except Exception as e:
        _emit({"kind": "done", "pair": pair_idx, "problem_id": problem_id,
               "error": f"sglang load_adapter failed: {e}",
               "traceback": traceback.format_exc()})
        async with trainer_lock:
            trainer.delete_problem_adapter(adapter_name)
        return

    # Per-rollout sandbox: K sandboxes per problem so all K kernels of a turn
    # nvcc-compile in parallel. With max_inflight_per_pair=1, only one set of
    # K sandboxes lives on a trainer GPU at a time → memory bounded.
    K = cfg.rollout.num_samples
    try:
        env.open_sandboxes(problem_id, n=K)
    except Exception as e:
        _emit({"kind": "done", "pair": pair_idx, "problem_id": problem_id,
               "error": f"open_sandbox failed: {e}",
               "traceback": traceback.format_exc()})
        try:
            await sglang.unload_adapter_async(adapter_name)
        finally:
            async with trainer_lock:
                trainer.delete_problem_adapter(adapter_name)
        return

    problem = env.get_problem(problem_id)
    # K already bound above for open_sandboxes.
    best_reward = float("-inf")
    best_speedup = 0.0
    any_correct = False

    try:
        for turn in range(cfg.loop.num_turns):
            # --- sample (async, unlocked: SGLang handles concurrent prompts)
            gens = await sglang.sample_async(
                prompt=problem.prompt,
                n=K,
                temperature=cfg.rollout.temperature,
                top_p=cfg.rollout.top_p,
                max_tokens=cfg.rollout.max_tokens,
                adapter_name=adapter_name,
            )

            # --- evaluate: K parallel evaluate() calls, one per rollout slot.
            # Each rollout has its own sandbox subprocess so nvcc compiles
            # run in parallel across the K rollouts on CPU cores.
            loop = asyncio.get_running_loop()
            results = await asyncio.gather(*[
                loop.run_in_executor(
                    None,
                    lambda g=g, s=k: env.evaluate(problem, g.text, slot=s),
                )
                for k, g in enumerate(gens)
            ])

            for k, r in enumerate(results):
                rec = {
                    "problem_id": problem_id, "problem_name": problem.name,
                    "turn": turn, "sample": k, "pair": pair_idx,
                    "reward": r.reward, "compiled": r.compiled, "correct": r.correct,
                    "speedup": r.speedup, "error_kind": r.error_kind,
                    "runtime_ms": r.runtime_ms, "ref_runtime_ms": r.ref_runtime_ms,
                    "feedback": r.feedback, "kernel_src": r.kernel_src,
                    "completion": r.raw_completion, "prompt": problem.prompt,
                }
                fname = f"p{problem_id:03d}_t{turn}_s{k:02d}.json"
                with open(rollout_dir / fname, "w") as f:
                    json.dump(rec, f, ensure_ascii=False)

            ros = [Rollout(prompt=problem.prompt, completion=r.raw_completion, reward=r.reward)
                   for r in results]
            # --- GRPO step (serialized across concurrent problems on this pair)
            async with trainer_lock:
                train_metrics = await loop.run_in_executor(
                    None, lambda: trainer.step(ros, adapter_name=adapter_name),
                )

            rewards = [r.reward for r in results]
            speedups = [r.speedup for r in results if r.error_kind == "ok"]
            n_compiled = sum(1 for r in results if r.compiled)
            n_correct = sum(1 for r in results if r.correct)
            comp_tokens = [int(g.completion_tokens) for g in gens if g.completion_tokens]
            n_truncated = sum(1 for g in gens if g.finish_reason == "length")

            if rewards:
                best_reward = max(best_reward, max(rewards))
            if speedups:
                best_speedup = max(best_speedup, max(speedups))
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

            # Save weights + hot-swap into SGLang for the next turn's rollouts.
            if cfg.logging.save_adapter_every_turn or turn == cfg.loop.num_turns - 1:
                async with trainer_lock:
                    trainer.save_adapter(adapter_name, adapter_parent)
                await sglang.reload_adapter_async(adapter_name, adapter_dir)

        _emit({
            "kind": "done",
            "pair": pair_idx,
            "problem_id": problem_id,
            "best_reward": best_reward if best_reward != float("-inf") else 0.0,
            "best_speedup": best_speedup,
            "any_correct": any_correct,
        })
    except Exception as e:
        # A per-problem failure shouldn't take down the whole worker — other
        # problems may still be running on this pair. We only mark this one as
        # failed and let the orchestrator decide whether to requeue.
        _emit({"kind": "done", "pair": pair_idx, "problem_id": problem_id,
               "error": str(e), "traceback": traceback.format_exc()})
    finally:
        try:
            env.close_sandboxes(problem_id)
        except Exception:
            pass
        try:
            await sglang.unload_adapter_async(adapter_name)
        except Exception:
            pass
        async with trainer_lock:
            trainer.delete_problem_adapter(adapter_name)


async def amain() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--pair-idx", type=int, required=True)
    ap.add_argument("--sglang-url", required=True)
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    # Single-process trainer; clear distributed env defensively.
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        os.environ.pop(k, None)

    cfg = load_config(args.config, overrides=args.override)
    cfg.sglang.base_url = args.sglang_url
    cfg.sglang.adapter_out_dir = args.adapter_dir

    try:
        trainer = GRPOLoRATrainer(cfg.model, cfg.lora, cfg.grpo)
        env = KernelEnv(cfg.kernelbench, cfg.reward)
        sglang = SGLangClient(cfg.sglang, model_name=cfg.model.name)
        # We're already inside an event loop — must use the async helper.
        await sglang.wait_ready_async(timeout_s=1800.0)
    except Exception as e:
        _emit({"kind": "fatal", "pair": args.pair_idx, "error": str(e),
               "traceback": traceback.format_exc()})
        raise

    rollout_dir = Path(args.run_dir) / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    _emit({"kind": "ready", "pair": args.pair_idx})

    # Hook stdin into asyncio so we can `await` cmd lines without blocking.
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    trainer_lock = asyncio.Lock()
    inflight: dict[int, asyncio.Task] = {}

    def _on_done(pid: int):
        def _cb(task: asyncio.Task) -> None:
            inflight.pop(pid, None)
            # Surface unhandled exceptions to stderr (the JSON channel already
            # got the done/error event from _run_one_problem).
            if task.exception() is not None:
                sys.stderr.write(f"[worker] problem {pid} task error: {task.exception()}\n")
        return _cb

    while True:
        raw = await reader.readline()
        if not raw:
            break  # parent closed our stdin
        line = raw.decode().strip()
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

        pid = int(msg["problem_id"])
        if pid in inflight:
            _emit({"kind": "error", "pair": args.pair_idx, "problem_id": pid,
                   "error": "already in flight"})
            continue
        task = asyncio.create_task(
            _run_one_problem(pid, args.pair_idx, cfg, trainer, env, sglang,
                             rollout_dir, trainer_lock),
            name=f"problem-{pid}",
        )
        task.add_done_callback(_on_done(pid))
        inflight[pid] = task

    # Drain anything still running before exit.
    if inflight:
        await asyncio.gather(*inflight.values(), return_exceptions=True)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
