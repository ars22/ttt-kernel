"""Async pool of subprocess sandbox slots.

Each slot owns one long-lived `env.eval_worker` subprocess on its own CUDA
context. The pool exposes one async `evaluate(req)` entry point used by the
FastAPI handler.

Why subprocess + JSON?
- KernelBench JIT-compiles user kernels and runs them in-process. A bad kernel
  (illegal memory access, infinite loop, etc.) can poison the CUDA context.
  Isolating each eval slot in its own process means we just kill+respawn the
  poisoned slot — the service stays up.

Capacity model: `max_concurrent` = total slots on this node. The pool
maintains `in_flight` for self-reported capacity over /capacity.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional

from ..shared.types import EvaluateRequest, EvaluateResponse
from .problem_set import ProblemSet
from .scoring import (
    RewardCfg,
    extract_kernel_src,
    harness_none,
    parse_failed,
    sandbox_or_exception,
    score_ok,
)

log = logging.getLogger("ttt_kernel.env.pool")


@dataclass(frozen=True)
class SandboxCfg:
    """Static init payload sent to each eval_worker subprocess on startup."""
    repo_path: str
    gpu_arch: str
    backend: str
    precision: str
    timing_method: str
    num_correct_trials: int
    num_perf_trials: int
    eval_timeout_s: int = 120

    def init_msg(self) -> dict:
        return {
            "repo_path": self.repo_path,
            "gpu_arch": self.gpu_arch,
            "backend": self.backend,
            "precision": self.precision,
            "timing_method": self.timing_method,
            "num_correct_trials": self.num_correct_trials,
            "num_perf_trials": self.num_perf_trials,
        }


class _Slot:
    """One subprocess. Not thread-safe; serialized by the pool's slot queue."""

    def __init__(self, slot_id: int, sandbox_cfg: SandboxCfg, log_path: Optional[str]):
        self.slot_id = slot_id
        self.cfg = sandbox_cfg
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None

    def spawn(self) -> None:
        """Synchronous; called once from pool startup (off the event loop)."""
        stderr_target: int | object
        if self.log_path:
            stderr_target = open(self.log_path, "a", buffering=1)
        else:
            stderr_target = subprocess.DEVNULL

        env = os.environ.copy()
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "ttt_kernel.env.eval_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            env=env,
            text=True,
            bufsize=1,
        )
        proc.stdin.write(json.dumps(self.cfg.init_msg()) + "\n")
        proc.stdin.flush()
        ready_line = proc.stdout.readline()
        if not ready_line:
            raise RuntimeError(f"slot {self.slot_id}: sandbox closed stdout before ready")
        ready = json.loads(ready_line)
        if ready.get("status") != "ready":
            raise RuntimeError(f"slot {self.slot_id}: init failed: {ready}")
        self.proc = proc

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def kill(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.kill()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.proc = None

    def send_recv(self, ref_src: str, kernel_src: str) -> dict:
        """Send one evaluate command; read one reply. Blocking I/O — call from
        a thread, not the event loop. Caller owns respawn on death."""
        assert self.proc is not None
        req = json.dumps({"cmd": "evaluate", "ref_src": ref_src, "custom_src": kernel_src})
        try:
            self.proc.stdin.write(req + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ConnectionResetError) as e:
            return {"status": "sandbox_died", "error": str(e)}
        line = self.proc.stdout.readline()
        if not line:
            rc = self.proc.poll()
            return {"status": "sandbox_died", "error": f"stdout closed (rc={rc})"}
        try:
            return json.loads(line)
        except Exception as e:  # noqa: BLE001
            return {"status": "exception", "error": f"bad-json: {e}",
                    "traceback": traceback.format_exc()}

    def graceful_exit(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.kill()


class EnvPool:
    """Async pool. Acquire blocks on `asyncio.Queue` until a slot frees."""

    def __init__(
        self,
        max_concurrent: int,
        sandbox_cfg: SandboxCfg,
        reward_cfg: RewardCfg,
        problem_set: ProblemSet,
        sandbox_log_path: Optional[str] = None,
    ):
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >=1, got {max_concurrent}")
        self.max_concurrent = max_concurrent
        self.sandbox_cfg = sandbox_cfg
        self.reward_cfg = reward_cfg
        self.problem_set = problem_set
        self._slots: list[_Slot] = [
            _Slot(i, sandbox_cfg, sandbox_log_path) for i in range(max_concurrent)
        ]
        self._free: asyncio.Queue[_Slot] = asyncio.Queue()
        self._in_flight = 0
        self._extract_fn = None  # set in start()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def start(self) -> None:
        """Serially spawn all slots. Each grabs ~3 GB of CUDA context; doing
        them in parallel produces a memory burst that OOMs the GPU."""
        from kernelbench.utils import extract_first_code
        self._extract_fn = extract_first_code
        for slot in self._slots:
            await asyncio.to_thread(slot.spawn)
            self._free.put_nowait(slot)
        log.info("env pool ready: %d slots", self.max_concurrent)

    async def shutdown(self) -> None:
        for slot in self._slots:
            await asyncio.to_thread(slot.graceful_exit)

    async def evaluate(self, req: EvaluateRequest) -> EvaluateResponse:
        problem = self.problem_set.get_problem(req.problem_id)
        kernel_src = extract_kernel_src(req.completion, self._extract_fn)
        if not kernel_src:
            return parse_failed(req.completion, self.reward_cfg)

        slot = await self._free.get()
        self._in_flight += 1
        t0 = time.monotonic()
        try:
            if not slot.alive():
                log.warning("slot %d dead on dequeue; respawning", slot.slot_id)
                slot.kill()
                await asyncio.to_thread(slot.spawn)
            reply = await asyncio.to_thread(slot.send_recv, problem.ref_src, kernel_src)
            status = reply.get("status")
            if status == "ok":
                return score_ok(req.completion, kernel_src, reply, self.reward_cfg)
            if status == "harness_none":
                return harness_none(req.completion, kernel_src, self.reward_cfg)
            # sandbox_died or exception → mark slot dead so next caller respawns
            if status == "sandbox_died":
                slot.kill()
            return sandbox_or_exception(
                req.completion, kernel_src, self.reward_cfg,
                status=status or "exception",
                error=reply.get("error", "unknown"),
                traceback=reply.get("traceback", ""),
            )
        finally:
            self._in_flight -= 1
            log.debug("eval p%d t%d s%d in %.2fs",
                      req.problem_id, req.turn, req.sample, time.monotonic() - t0)
            self._free.put_nowait(slot)
