"""Reward shaping — pure functions from sandbox reply → EvaluateResponse.

Ported from `kernel_env.py::_score` (pair-orchestrator path). Behaviour is
identical so an algorithmic-parity check vs `main` is well-defined.

The `</think>` strip below catches thinking models (Qwen3-*-Thinking) which
emit draft kernels inside chain-of-thought that they then self-reject. Without
the strip ~19% of rollouts would be scored on the discarded draft.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..shared.types import ErrorKind, EvaluateResponse


@dataclass(frozen=True)
class RewardCfg:
    speedup_log_scale: bool = True
    error_penalty: float = -3.0
    incorrect_penalty: float = -3.0
    clip: float = 2.0


def extract_kernel_src(raw_completion: str, extract_fn) -> Optional[str]:
    # Strip the chain-of-thought prefix so the extractor only sees the model's
    # final answer. Different model families use different markers:
    #   - Qwen3-Thinking, R1-distill, etc.:  ...<think>...</think>FINAL ANSWER
    #   - gpt-oss (harmony format):          <|channel|>analysis<|message|>...
    #                                          <|channel|>final<|message|>FINAL ANSWER
    if "<|channel|>final<|message|>" in raw_completion:
        post = raw_completion.rsplit("<|channel|>final<|message|>", 1)[1]
    elif "</think>" in raw_completion:
        post = raw_completion.rsplit("</think>", 1)[1]
    else:
        post = raw_completion
    return extract_fn(post, ["python", "cpp"]) or None


def _result(
    *,
    raw_completion: str,
    kernel_src: Optional[str],
    compiled: bool,
    correct: bool,
    speedup: float,
    runtime_ms: float,
    ref_runtime_ms: float,
    reward: float,
    feedback: str,
    error_kind: ErrorKind,
) -> EvaluateResponse:
    return EvaluateResponse(
        raw_completion=raw_completion,
        kernel_src=kernel_src,
        compiled=compiled,
        correct=correct,
        speedup=speedup,
        runtime_ms=runtime_ms,
        ref_runtime_ms=ref_runtime_ms,
        reward=reward,
        feedback=feedback,
        error_kind=error_kind,
    )


def parse_failed(raw_completion: str, reward_cfg: RewardCfg) -> EvaluateResponse:
    return _result(
        raw_completion=raw_completion,
        kernel_src=None,
        compiled=False, correct=False, speedup=-1.0,
        runtime_ms=-1.0, ref_runtime_ms=-1.0,
        reward=reward_cfg.error_penalty,
        feedback=(
            "[parse-error] No fenced ```python``` code block was emitted. "
            "Wrap your kernel in ```python ... ``` so it can be extracted."
        ),
        error_kind="parse",
    )


def harness_none(raw_completion: str, kernel_src: str, reward_cfg: RewardCfg) -> EvaluateResponse:
    return _result(
        raw_completion=raw_completion, kernel_src=kernel_src,
        compiled=False, correct=False, speedup=-1.0,
        runtime_ms=-1.0, ref_runtime_ms=-1.0,
        reward=reward_cfg.error_penalty,
        feedback="[harness-error] eval_kernel_against_ref returned None (likely nvcc abort or missing .so).",
        error_kind="harness",
    )


def sandbox_or_exception(
    raw_completion: str,
    kernel_src: str,
    reward_cfg: RewardCfg,
    *,
    status: str,
    error: str,
    traceback: str,
) -> EvaluateResponse:
    tag = "[sandbox-died]" if status == "sandbox_died" else "[harness-error]"
    fb = f"{tag} eval_kernel_against_ref raised:\n{error}\n{traceback}".strip()
    kind: ErrorKind = "harness"
    return _result(
        raw_completion=raw_completion, kernel_src=kernel_src,
        compiled=False, correct=False, speedup=-1.0,
        runtime_ms=-1.0, ref_runtime_ms=-1.0,
        reward=reward_cfg.error_penalty,
        feedback=fb,
        error_kind=kind,
    )


def score_ok(
    raw_completion: str,
    kernel_src: str,
    reply: dict,
    reward_cfg: RewardCfg,
) -> EvaluateResponse:
    compiled = bool(reply.get("compiled", False))
    correct = bool(reply.get("correctness", False))
    runtime = float(reply.get("runtime", -1.0))
    ref_runtime = float(reply.get("ref_runtime", -1.0))
    feedback = reply.get("feedback", "")

    if not compiled:
        return _result(
            raw_completion=raw_completion, kernel_src=kernel_src,
            compiled=False, correct=False, speedup=-1.0,
            runtime_ms=runtime, ref_runtime_ms=ref_runtime,
            reward=reward_cfg.error_penalty,
            feedback=feedback, error_kind="compile",
        )
    if not correct:
        return _result(
            raw_completion=raw_completion, kernel_src=kernel_src,
            compiled=True, correct=False, speedup=-1.0,
            runtime_ms=runtime, ref_runtime_ms=ref_runtime,
            reward=reward_cfg.incorrect_penalty,
            feedback=feedback, error_kind="incorrect",
        )

    if runtime > 0 and ref_runtime > 0:
        speedup = ref_runtime / runtime
    else:
        speedup = 1.0
    if reward_cfg.speedup_log_scale:
        reward = math.log(max(speedup, 1e-6))
    else:
        reward = speedup - 1.0
    reward = max(-reward_cfg.clip, min(reward_cfg.clip, reward))

    return _result(
        raw_completion=raw_completion, kernel_src=kernel_src,
        compiled=True, correct=True, speedup=speedup,
        runtime_ms=runtime, ref_runtime_ms=ref_runtime,
        reward=reward, feedback=feedback, error_kind="ok",
    )
