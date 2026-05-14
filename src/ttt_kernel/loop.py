"""Online-RL loop: for each batch of problems, T turns of (rollout → reward → GRPO step → push LoRA).

A batch of B problems shares one adapter: the adapter is reset at the start of
the batch, trained jointly for `num_turns` turns over all B*K rollouts, with
GRPO advantages computed per-problem (so each problem's K rollouts form its own
group). This lets one round of LoRA updates absorb learning signal from
multiple problems at once.

When launched under torchrun, rank 0 owns all I/O (SGLang client, KernelBench
env, logger) and broadcasts rollouts to the other ranks; every rank then runs
the GRPO step collectively via DDP.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import List

import torch.distributed as dist

from .config import Config
from .grpo_trainer import GRPOLoRATrainer, Rollout
from .kernel_env import KernelEnv, Problem
from .logger import JsonlLogger
from .sglang_client import SGLangClient


def _summarize(name: str, values: list[float]) -> dict:
    if not values:
        return {f"{name}_mean": 0.0, f"{name}_max": 0.0, f"{name}_min": 0.0}
    return {
        f"{name}_mean": statistics.fmean(values),
        f"{name}_max": max(values),
        f"{name}_min": min(values),
    }


def _broadcast_object(obj, world_size: int):
    if world_size <= 1:
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run(cfg: Config) -> None:
    # Trainer initializes the process group (under torchrun).
    trainer = GRPOLoRATrainer(cfg.model, cfg.lora, cfg.grpo)
    is_main = trainer.rank == 0

    # I/O components only on rank 0 — KernelEnv, SGLang client, logger.
    env: KernelEnv | None = None
    sglang: SGLangClient | None = None
    logger: JsonlLogger | None = None
    if is_main:
        env = KernelEnv(cfg.kernelbench, cfg.reward)
        sglang = SGLangClient(cfg.sglang, model_name=cfg.model.name)
        logger = JsonlLogger(
            cfg.logging.out_dir,
            cfg.logging.run_name,
            wandb_cfg=cfg.logging.wandb,
            full_config=cfg.model_dump(),
        )
        logger.log("run_start", config=cfg.model_dump())
        sglang.wait_ready()

    problem_ids = env.list_problem_ids() if is_main else None
    problem_ids = _broadcast_object(problem_ids, trainer.world_size)
    if is_main:
        logger.log("problems", level=cfg.kernelbench.level, problem_ids=problem_ids)

    B = max(1, cfg.loop.problem_batch_size)
    K = cfg.rollout.num_samples

    for batch_idx, pid_batch in enumerate(_chunks(problem_ids, B)):
        if not cfg.loop.persist_adapter_across_problems:
            trainer.reset_adapter()
            trainer.save_adapter(cfg.sglang.adapter_out_dir)
            if is_main:
                sglang.reload_adapter()

        # Rank 0 owns Problem objects (prompt strings + ref code).
        problems = [env.get_problem(pid) for pid in pid_batch] if is_main else None
        prompts = [p.prompt for p in problems] if is_main else None

        if is_main:
            for p, pid in zip(problems, pid_batch):
                logger.log("problem_start", problem_id=pid, name=p.name, batch=batch_idx)

        for turn in range(cfg.loop.num_turns):
            ros: List[Rollout] = []
            group_ids: List[int] = []

            if is_main:
                # Fan out B*K parallel rollouts in a single event loop so SGLang
                # can interleave them across replicas.
                all_gens = sglang.sample_many(
                    prompts=prompts,
                    n=K,
                    temperature=cfg.rollout.temperature,
                    top_p=cfg.rollout.top_p,
                    max_tokens=cfg.rollout.max_tokens,
                    use_adapter=True,
                )

                rollout_dir = Path(logger.run_dir) / "rollouts"
                rollout_dir.mkdir(parents=True, exist_ok=True)

                per_problem_rewards: dict[int, list[float]] = {pid: [] for pid in pid_batch}
                per_problem_speedups: dict[int, list[float]] = {pid: [] for pid in pid_batch}
                per_problem_compiled: dict[int, int] = {pid: 0 for pid in pid_batch}
                per_problem_correct: dict[int, int] = {pid: 0 for pid in pid_batch}

                for b, (problem, pid, gens) in enumerate(zip(problems, pid_batch, all_gens)):
                    for k, g in enumerate(gens):
                        r = env.evaluate(problem, g.text)

                        per_problem_rewards[pid].append(r.reward)
                        if r.error_kind == "ok":
                            per_problem_speedups[pid].append(r.speedup)
                        if r.compiled:
                            per_problem_compiled[pid] += 1
                        if r.correct:
                            per_problem_correct[pid] += 1

                        logger.log(
                            "rollout",
                            problem_id=pid, turn=turn, sample=k, batch=batch_idx,
                            reward=r.reward, compiled=r.compiled, correct=r.correct,
                            speedup=r.speedup, error_kind=r.error_kind,
                            runtime_ms=r.runtime_ms, ref_runtime_ms=r.ref_runtime_ms,
                        )
                        rec = {
                            "problem_id": pid,
                            "problem_name": problem.name,
                            "turn": turn,
                            "sample": k,
                            "batch": batch_idx,
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
                        fname = f"p{pid:03d}_t{turn}_s{k:02d}.json"
                        with open(rollout_dir / fname, "w") as f:
                            json.dump(rec, f, ensure_ascii=False)

                        ros.append(Rollout(
                            prompt=problem.prompt,
                            completion=r.raw_completion,
                            reward=r.reward,
                        ))
                        group_ids.append(b)  # group = problem index within the batch

            # ---- all ranks: GRPO step on broadcast rollouts ----------------
            payload = _broadcast_object((ros, group_ids), trainer.world_size)
            ros, group_ids = payload
            train_metrics = trainer.step(ros, group_ids=group_ids)

            if is_main:
                # Per-problem summary
                for pid in pid_batch:
                    rewards = per_problem_rewards[pid]
                    speedups = per_problem_speedups[pid]
                    logger.log(
                        "turn_summary",
                        problem_id=pid, turn=turn, batch=batch_idx,
                        n_rollouts=len(rewards),
                        n_compiled=per_problem_compiled[pid],
                        n_correct=per_problem_correct[pid],
                        **_summarize("reward", rewards),
                        **_summarize("speedup", speedups),
                    )
                # Batch-level summary (one set per turn covering the whole batch)
                all_rewards = [r for pid in pid_batch for r in per_problem_rewards[pid]]
                all_speedups = [s for pid in pid_batch for s in per_problem_speedups[pid]]
                logger.log(
                    "batch_turn_summary",
                    batch=batch_idx, turn=turn,
                    n_problems=len(pid_batch),
                    n_rollouts=len(all_rewards),
                    **_summarize("reward", all_rewards),
                    **_summarize("speedup", all_speedups),
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                )

            # Push updated LoRA to the serving instance for the next turn.
            if cfg.logging.save_adapter_every_turn or turn == cfg.loop.num_turns - 1:
                trainer.save_adapter(cfg.sglang.adapter_out_dir)
                if is_main:
                    sglang.reload_adapter()

        if is_main:
            for pid in pid_batch:
                logger.log("problem_end", problem_id=pid, batch=batch_idx)

    if is_main:
        logger.log("run_end")
        logger.close()

    if trainer.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
