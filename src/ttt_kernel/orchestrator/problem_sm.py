"""Per-problem state machine.

For each problem we run `num_turns` of:
    1. Pick a sampler with capacity, sample K completions for the current
       (prompt, adapter_v_t) pair.
    2. Fan K completions out across env nodes — one /evaluate each — and
       gather (reward, feedback) per rollout.
    3. Pick a trainer with capacity, run one GRPO step on the K rollouts;
       trainer reads v_t and writes v_{t+1}.

A turn is fire-and-forget per stage: once K completions land, the sampler
slot is released; that sampler may never see this problem again. The only
state tying turns together is the adapter path on Weka.

Feedback from prior turns is concatenated into the prompt of the next turn,
mirroring the existing multi-turn loop in `worker.py`.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..shared.adapter_paths import AdapterRef, seed
from ..shared.types import (
    EvaluateRequest,
    EvaluateResponse,
    Rollout,
    SampleRequest,
    TrainRequest,
    TrainResponse,
)
from .scheduler import Pool

log = logging.getLogger("ttt_kernel.orchestrator.problem_sm")


def _feedback_block(prior: list[tuple[str, EvaluateResponse]], max_chars: int = 4000) -> str:
    """Build a 'previous attempts' header for the next turn's prompt."""
    if not prior:
        return ""
    parts = ["", "## Previous attempts (most recent last):"]
    for i, (_, r) in enumerate(prior):
        fb = r.feedback or "(no feedback)"
        parts.append(f"- attempt {i}: reward={r.reward:.3f} error_kind={r.error_kind}")
        # Trim very long compiler dumps so we don't blow the prompt budget.
        parts.append(f"  feedback: {fb[:max_chars]}")
    return "\n".join(parts) + "\n"


async def run_problem(
    *,
    problem_id: int,
    num_turns: int,
    K: int,
    sampler_pool: Pool,
    env_pool: Pool,
    trainer_pool: Pool,
    adapters_root: str,
    base_prompt_fetcher,        # async (problem_id) -> str
    logger,                      # JsonlLogger
    sample_kwargs: dict,
) -> dict:
    """Drive one problem through `num_turns` turns. Returns aggregate metrics."""
    prompt0 = await base_prompt_fetcher(problem_id)
    current = seed(adapters_root, problem_id)  # v000
    history: list[tuple[str, EvaluateResponse]] = []
    turn_metrics: list[dict] = []

    logger.log("problem_start", problem_id=problem_id, num_turns=num_turns, K=K)

    for turn in range(num_turns):
        prompt = prompt0 + _feedback_block(history)
        next_ref: AdapterRef = current.next()

        # ---- sample K rollouts on whichever sampler has capacity -----------
        async with sampler_pool.acquire() as (sclient, sentry):
            req = SampleRequest(
                problem_id=problem_id,
                turn=turn,
                prompt=prompt,
                adapter_path=str(current.path),
                adapter_name=current.name,
                n=K,
                **sample_kwargs,
            )
            logger.log("sample_start", problem_id=problem_id, turn=turn,
                       sampler_idx=sentry.idx, adapter=current.name)
            sresp = await sclient.sample(req)
            logger.log("sample_done", problem_id=problem_id, turn=turn,
                       sampler_idx=sentry.idx,
                       completion_tokens=sum(g.completion_tokens for g in sresp.completions))

        completions = [g.text for g in sresp.completions]

        # ---- evaluate K completions on env nodes (fan out) -----------------
        async def _eval_one(sample_idx: int, completion: str) -> EvaluateResponse:
            async with env_pool.acquire() as (eclient, eentry):
                ereq = EvaluateRequest(
                    problem_id=problem_id,
                    turn=turn,
                    sample=sample_idx,
                    completion=completion,
                )
                return await eclient.evaluate(ereq)

        import asyncio as _asyncio
        eval_results: list[EvaluateResponse] = await _asyncio.gather(*[
            _eval_one(i, c) for i, c in enumerate(completions)
        ])
        logger.log("evaluate_done", problem_id=problem_id, turn=turn,
                   rewards=[float(r.reward) for r in eval_results],
                   error_kinds=[r.error_kind for r in eval_results])

        # ---- one GRPO step on whichever trainer has capacity ---------------
        rollouts = [
            Rollout(prompt=prompt, completion=c, reward=float(r.reward))
            for c, r in zip(completions, eval_results)
        ]
        async with trainer_pool.acquire() as (tclient, tentry):
            treq = TrainRequest(
                problem_id=problem_id,
                turn=turn,
                adapter_in_path=str(current.path),
                adapter_in_name=current.name,
                adapter_out_path=str(next_ref.path),
                adapter_out_name=next_ref.name,
                rollouts=rollouts,
            )
            logger.log("train_start", problem_id=problem_id, turn=turn,
                       trainer_idx=tentry.idx,
                       adapter_in=current.name, adapter_out=next_ref.name)
            tresp: TrainResponse = await tclient.train(treq)
            logger.log("train_done", problem_id=problem_id, turn=turn,
                       trainer_idx=tentry.idx, **tresp.model_dump())

        # ---- per-turn rollup ----------------------------------------------
        rewards = [float(r.reward) for r in eval_results]
        turn_metrics.append({
            "problem_id": problem_id,
            "turn": turn,
            "reward_mean": sum(rewards) / len(rewards),
            "reward_max": max(rewards),
            "n_correct": sum(1 for r in eval_results if r.correct),
            "n_compiled": sum(1 for r in eval_results if r.compiled),
            "loss": tresp.loss,
            "pg": tresp.pg,
            "kl": tresp.kl,
            "grad_norm": tresp.grad_norm,
        })
        logger.log("turn", **turn_metrics[-1])

        # Carry the best attempt's feedback forward.
        best_idx = max(range(len(eval_results)), key=lambda i: eval_results[i].reward)
        history.append((completions[best_idx], eval_results[best_idx]))
        current = next_ref

    logger.log("problem_done", problem_id=problem_id,
               final_adapter=str(current.path),
               turn_metrics=turn_metrics)
    return {"problem_id": problem_id, "turn_metrics": turn_metrics,
            "final_adapter_path": str(current.path)}
