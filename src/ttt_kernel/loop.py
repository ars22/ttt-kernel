"""Online-RL loop: for each problem, T turns of (rollout → reward → GRPO step → push LoRA).

This is the "inference script" — it does both inference and weight updates;
the name reflects that we're using it like an inference procedure that happens
to also adapt the model on the fly (test-time training).
"""
from __future__ import annotations

import statistics
from typing import List

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


def run(cfg: Config) -> None:
    env = KernelEnv(cfg.kernelbench, cfg.reward)
    sglang = SGLangClient(cfg.sglang, model_name=cfg.model.name)
    trainer = GRPOLoRATrainer(cfg.model, cfg.lora, cfg.grpo)

    logger = JsonlLogger(cfg.logging.out_dir, cfg.logging.run_name)
    logger.log("run_start", config=cfg.model_dump())

    sglang.wait_ready()

    problem_ids = env.list_problem_ids()
    logger.log("problems", level=cfg.kernelbench.level, problem_ids=problem_ids)

    for problem_id in problem_ids:
        if not cfg.loop.persist_adapter_across_problems:
            trainer.reset_adapter()
            trainer.save_adapter(cfg.sglang.adapter_out_dir)
            sglang.reload_adapter()

        problem = env.get_problem(problem_id)
        logger.log("problem_start", problem_id=problem_id, name=problem.name)

        for turn in range(cfg.loop.num_turns):
            gens = sglang.sample(
                prompt=problem.prompt,
                n=cfg.rollout.num_samples,
                temperature=cfg.rollout.temperature,
                top_p=cfg.rollout.top_p,
                max_tokens=cfg.rollout.max_tokens,
                use_adapter=True,
            )

            results = [env.evaluate(problem, g.text) for g in gens]

            rewards = [r.reward for r in results]
            speedups = [r.speedup for r in results if r.error_kind == "ok"]
            n_compiled = sum(1 for r in results if r.compiled)
            n_correct = sum(1 for r in results if r.correct)

            for i, r in enumerate(results):
                logger.log(
                    "rollout",
                    problem_id=problem_id, turn=turn, sample=i,
                    reward=r.reward, compiled=r.compiled, correct=r.correct,
                    speedup=r.speedup, error_kind=r.error_kind,
                    runtime_ms=r.runtime_ms, ref_runtime_ms=r.ref_runtime_ms,
                )

            # GRPO update — train only on rollouts where we got a real reward signal.
            ros: List[Rollout] = [
                Rollout(prompt=problem.prompt, completion=r.raw_completion, reward=r.reward)
                for r in results
            ]
            train_metrics = trainer.step(ros)

            logger.log(
                "turn_summary",
                problem_id=problem_id, turn=turn,
                n_rollouts=len(results),
                n_compiled=n_compiled, n_correct=n_correct,
                **_summarize("reward", rewards),
                **_summarize("speedup", speedups),
                **train_metrics,
            )

            # Push updated LoRA to the serving instance for the next turn.
            if cfg.logging.save_adapter_every_turn or turn == cfg.loop.num_turns - 1:
                trainer.save_adapter(cfg.sglang.adapter_out_dir)
                sglang.reload_adapter()

        logger.log("problem_end", problem_id=problem_id)

    logger.log("run_end")
    logger.close()
