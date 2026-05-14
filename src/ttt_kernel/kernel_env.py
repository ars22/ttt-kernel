"""KernelBench environment: prompt → reward + feedback.

We treat each KernelBench problem as an MDP with a single step:
    state  = problem source code (PyTorch reference module)
    action = a CUDA kernel string (the LLM completion)
    reward = scalar derived from compile/correctness/speedup
    info   = textual feedback the model can read on the next turn (stack trace, diffs, etc.)

This module isolates all KernelBench imports and gpu-arch / env-var setup so the
trainer/rollout code never has to touch it directly.

Eval runs in a sandbox subprocess (ttt_kernel.eval_worker). That process owns
its own CUDA context on the trainer GPU; if an LLM-generated kernel triggers
an illegal-memory-access or otherwise poisons CUDA, only the sandbox dies and
the parent transparently respawns it. The trainer's CUDA context stays clean.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
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
class _ReplyShim:
    """Adapter so the sandbox JSON reply quacks like a KernelBench eval result."""
    compiled: bool
    correctness: bool
    runtime: float
    ref_runtime: float
    feedback: str


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

        # We still import the lightweight pieces (dataset, prompt builder,
        # code extractor) in-process; only the HEAVY eval lives in a subprocess.
        from kernelbench.utils import set_gpu_arch  # noqa: WPS433
        set_gpu_arch([cfg.gpu_arch])
        from kernelbench import dataset as kb_dataset
        from kernelbench import prompt_constructor_toml as kb_prompts
        from kernelbench.utils import extract_first_code

        self._construct = kb_dataset.construct_kernelbench_dataset
        self._get_prompt = kb_prompts.get_prompt_for_backend
        self._extract = extract_first_code

        self._dataset = self._construct(
            level=cfg.level,
            source=cfg.dataset_src,
            dataset_name=cfg.dataset_name,
        )

        # One sandbox subprocess per in-flight problem_id. Spawned via
        # spawn_sandbox(pid) and torn down via close_sandbox(pid). nvcc compiles
        # inside a sandbox are CPU-bound and can overlap; GPU work serializes
        # naturally because all sandboxes share the trainer GPU.
        self._sandboxes: dict[int, subprocess.Popen] = {}
        self._sandbox_log_path: Optional[str] = os.environ.get("TTT_SANDBOX_LOG")

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

    # ---- eval subprocess plumbing ------------------------------------------

    def _spawn_one(self) -> subprocess.Popen:
        """Spawn a single eval subprocess, send init, wait for ready."""
        stderr_target: int | object
        if self._sandbox_log_path:
            stderr_target = open(self._sandbox_log_path, "a", buffering=1)
        else:
            stderr_target = subprocess.DEVNULL

        env = os.environ.copy()
        repo_src = os.path.join(os.path.dirname(os.path.dirname(__file__)))
        env["PYTHONPATH"] = repo_src + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "ttt_kernel.eval_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            env=env,
            text=True,
            bufsize=1,
        )
        init = {
            "repo_path": self.cfg.repo_path,
            "gpu_arch": self.cfg.gpu_arch,
            "backend": self.cfg.backend,
            "precision": self.cfg.precision,
            "timing_method": self.cfg.timing_method,
            "num_correct_trials": self.cfg.num_correct_trials,
            "num_perf_trials": self.cfg.num_perf_trials,
        }
        proc.stdin.write(json.dumps(init) + "\n")
        proc.stdin.flush()
        ready_line = proc.stdout.readline()
        if not ready_line:
            raise RuntimeError("eval sandbox closed stdout before sending ready")
        ready = json.loads(ready_line)
        if ready.get("status") != "ready":
            raise RuntimeError(f"eval sandbox failed to init: {ready}")
        return proc

    def open_sandbox(self, problem_id: int) -> None:
        """Spawn (or respawn) the per-problem eval sandbox."""
        existing = self._sandboxes.get(problem_id)
        if existing is not None and existing.poll() is None:
            return
        self._sandboxes[problem_id] = self._spawn_one()

    def close_sandbox(self, problem_id: int) -> None:
        proc = self._sandboxes.pop(problem_id, None)
        if proc is None:
            return
        try:
            proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
            proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    def _eval_via_sandbox(self, problem_id: int, ref_src: str, kernel_src: str) -> dict:
        """Send one evaluate request, return parsed reply. Auto-respawns on death."""
        try:
            proc = self._sandboxes.get(problem_id)
            if proc is None or proc.poll() is not None:
                # Sandbox died (e.g. CUDA context kill from a bad kernel) —
                # respawn transparently for the next request.
                self._sandboxes.pop(problem_id, None)
                self.open_sandbox(problem_id)
                proc = self._sandboxes[problem_id]
            req = json.dumps({"cmd": "evaluate", "ref_src": ref_src, "custom_src": kernel_src})
            proc.stdin.write(req + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                rc = proc.poll()
                self._sandboxes.pop(problem_id, None)
                return {
                    "status": "sandbox_died",
                    "error": f"sandbox stdout closed (returncode={rc})",
                }
            return json.loads(line)
        except (BrokenPipeError, ConnectionResetError) as e:
            self._sandboxes.pop(problem_id, None)
            return {"status": "sandbox_died", "error": str(e)}
        except Exception as e:  # noqa: BLE001
            self._sandboxes.pop(problem_id, None)
            return {"status": "sandbox_died", "error": str(e),
                    "traceback": traceback.format_exc()}

    def close(self) -> None:
        for pid in list(self._sandboxes.keys()):
            self.close_sandbox(pid)

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

        reply = self._eval_via_sandbox(problem.problem_id, problem.ref_src, kernel_src)
        status = reply.get("status")

        if status == "ok":
            return self._score_primitive(raw_completion, kernel_src, reply)

        if status == "harness_none":
            return RolloutResult(
                raw_completion=raw_completion, kernel_src=kernel_src,
                compiled=False, correct=False, speedup=-1.0,
                runtime_ms=-1.0, ref_runtime_ms=-1.0,
                reward=self.reward_cfg.error_penalty,
                feedback="[harness-error] eval_kernel_against_ref returned None (likely nvcc abort or missing .so).",
                error_kind="compile",
            )

        # exception or sandbox_died
        err = reply.get("error", "unknown")
        tb = reply.get("traceback", "")
        tag = "[sandbox-died]" if status == "sandbox_died" else "[harness-error]"
        fb = f"{tag} eval_kernel_against_ref raised:\n{err}\n{tb}".strip()
        return RolloutResult(
            raw_completion=raw_completion,
            kernel_src=kernel_src,
            compiled=False, correct=False, speedup=-1.0,
            runtime_ms=-1.0, ref_runtime_ms=-1.0,
            reward=self.reward_cfg.error_penalty,
            feedback=fb,
            error_kind="compile",
        )

    # ---- scoring ------------------------------------------------------------

    def _score_primitive(self, raw_completion: str, kernel_src: str, reply: dict) -> RolloutResult:
        """Same logic as _score, but inputs are the JSON fields from the sandbox."""
        return self._score(
            raw_completion, kernel_src,
            _ReplyShim(
                compiled=reply.get("compiled", False),
                correctness=reply.get("correctness", False),
                runtime=reply.get("runtime", -1.0),
                ref_runtime=reply.get("ref_runtime", -1.0),
                feedback=reply.get("feedback", ""),
            ),
        )

    def _score(self, raw_completion: str, kernel_src: str, kb_res) -> RolloutResult:
        compiled = bool(kb_res.compiled)
        correct = bool(kb_res.correctness)
        runtime = float(kb_res.runtime)
        ref_runtime = float(kb_res.ref_runtime)
        feedback = kb_res.summarize_for_feedback() if hasattr(kb_res, "summarize_for_feedback") else kb_res.feedback

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
