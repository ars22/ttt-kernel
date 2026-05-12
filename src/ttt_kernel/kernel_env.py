"""KernelBench environment: prompt → reward + feedback.

We treat each KernelBench problem as an MDP with a single step:
    state  = problem source code (PyTorch reference module)
    action = a CUDA kernel string (the LLM completion)
    reward = scalar derived from compile/correctness/speedup
    info   = textual feedback the model can read on the next turn (stack trace, diffs, etc.)

This module isolates all KernelBench imports and gpu-arch / env-var setup so the
trainer/rollout code never has to touch it directly.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import List, Optional

import torch


def _add_kernelbench_to_path(repo_path: str) -> None:
    """Make `kernelbench` importable without a real install."""
    src = os.path.join(repo_path, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


@dataclass
class Problem:
    level: int
    problem_id: int
    name: str
    ref_src: str    # PyTorch reference module source
    prompt: str     # Full prompt string to feed the LLM


@dataclass
class RolloutResult:
    """One LLM attempt at one problem."""
    raw_completion: str           # raw text from the model
    kernel_src: Optional[str]     # extracted ```python``` block, may be None on parse fail
    compiled: bool
    correct: bool
    speedup: float                # ref_runtime / runtime; -1 if no timing
    runtime_ms: float
    ref_runtime_ms: float
    reward: float
    feedback: str                 # model-readable summary (stack trace, diffs, perf)
    error_kind: Optional[str]     # "parse" | "compile" | "incorrect" | "ok"


# ----- public API ------------------------------------------------------------


class KernelEnv:
    """Wraps KernelBench dataset + eval into a stateless reward function."""

    def __init__(self, cfg, reward_cfg):
        self.cfg = cfg
        self.reward_cfg = reward_cfg
        _add_kernelbench_to_path(cfg.repo_path)

        # Imports deferred until after path setup.
        from kernelbench.utils import set_gpu_arch  # noqa: WPS433
        set_gpu_arch([cfg.gpu_arch])

        # Cache module references — they're heavy.
        from kernelbench import dataset as kb_dataset
        from kernelbench import eval as kb_eval
        from kernelbench import prompt_constructor_toml as kb_prompts
        from kernelbench.utils import extract_first_code
        from kernelbench.eval import get_torch_dtype_from_string

        self._construct = kb_dataset.construct_kernelbench_dataset
        self._eval_kernel = kb_eval.eval_kernel_against_ref
        self._get_prompt = kb_prompts.get_prompt_for_backend
        self._extract = extract_first_code
        self._dtype = get_torch_dtype_from_string(cfg.precision)

        self._dataset = self._construct(
            level=cfg.level,
            source=cfg.dataset_src,
            dataset_name=cfg.dataset_name,
        )

    # ---- problem listing ----------------------------------------------------

    def list_problem_ids(self) -> List[int]:
        if self.cfg.problem_ids:
            return list(self.cfg.problem_ids)
        return [p.problem_id for p in self._dataset]

    def get_problem(self, problem_id: int) -> Problem:
        prob = self._dataset.get_problem_by_id(problem_id)
        prompt = self._get_prompt(
            prob.code,
            self.cfg.backend,
            option=self.cfg.prompt_option,
            precision=self.cfg.precision,
            include_hardware=False,
            gpu_name=None,
        )
        return Problem(
            level=self.cfg.level,
            problem_id=problem_id,
            name=prob.name,
            ref_src=prob.code,
            prompt=prompt,
        )

    # ---- the actual reward call --------------------------------------------

    def evaluate(self, problem: Problem, raw_completion: str) -> RolloutResult:
        """Score one rollout. Never raises — failures become negative reward + feedback."""
        kernel_src = self._extract(raw_completion, ["python", "cpp"])
        if not kernel_src:
            return RolloutResult(
                raw_completion=raw_completion,
                kernel_src=None,
                compiled=False, correct=False, speedup=-1.0,
                runtime_ms=-1.0, ref_runtime_ms=-1.0,
                reward=self.reward_cfg.error_penalty,
                feedback=(
                    "[parse-error] No fenced ```python``` code block was emitted. "
                    "Wrap your kernel in ```python ... ``` so it can be extracted."
                ),
                error_kind="parse",
            )

        try:
            kb_res = self._eval_kernel(
                original_model_src=problem.ref_src,
                custom_model_src=kernel_src,
                verbose=False,
                measure_performance=True,
                timing_method=self.cfg.timing_method,
                num_correct_trials=self.cfg.num_correct_trials,
                num_perf_trials=self.cfg.num_perf_trials,
                backend=self.cfg.backend,
                precision=self._dtype,
            )
        except Exception:  # noqa: BLE001 — KB can raise on bad code; treat as error
            tb = traceback.format_exc(limit=20)
            return RolloutResult(
                raw_completion=raw_completion,
                kernel_src=kernel_src,
                compiled=False, correct=False, speedup=-1.0,
                runtime_ms=-1.0, ref_runtime_ms=-1.0,
                reward=self.reward_cfg.error_penalty,
                feedback=f"[harness-error] eval_kernel_against_ref raised:\n{tb}",
                error_kind="compile",
            )

        return self._score(raw_completion, kernel_src, kb_res)

    # ---- scoring ------------------------------------------------------------

    def _score(self, raw_completion: str, kernel_src: str, kb_res) -> RolloutResult:
        compiled = bool(kb_res.compiled)
        correct = bool(kb_res.correctness)
        runtime = float(kb_res.runtime)
        ref_runtime = float(kb_res.ref_runtime)
        feedback = kb_res.summarize_for_feedback()

        if not compiled:
            return RolloutResult(
                raw_completion=raw_completion, kernel_src=kernel_src,
                compiled=False, correct=False, speedup=-1.0,
                runtime_ms=runtime, ref_runtime_ms=ref_runtime,
                reward=self.reward_cfg.error_penalty,
                feedback=feedback, error_kind="compile",
            )
        if not correct:
            return RolloutResult(
                raw_completion=raw_completion, kernel_src=kernel_src,
                compiled=True, correct=False, speedup=-1.0,
                runtime_ms=runtime, ref_runtime_ms=ref_runtime,
                reward=self.reward_cfg.incorrect_penalty,
                feedback=feedback, error_kind="incorrect",
            )

        # compiled + correct → speedup-shaped reward
        if runtime > 0 and ref_runtime > 0:
            speedup = ref_runtime / runtime
        else:
            speedup = 1.0
        if self.reward_cfg.speedup_log_scale:
            reward = math.log(max(speedup, 1e-6))
        else:
            reward = speedup - 1.0
        reward = max(-self.reward_cfg.clip, min(self.reward_cfg.clip, reward))

        return RolloutResult(
            raw_completion=raw_completion, kernel_src=kernel_src,
            compiled=True, correct=True, speedup=speedup,
            runtime_ms=runtime, ref_runtime_ms=ref_runtime,
            reward=reward, feedback=feedback, error_kind="ok",
        )
