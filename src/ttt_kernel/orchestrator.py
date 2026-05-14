"""Async orchestrator for pool-based ttt-kernel training.

Launches N (sampler, trainer) pairs:
  - N SGLang subprocesses (each on its own GPU(s), its own port, its own adapter dir).
  - N trainer worker subprocesses (ttt_kernel.worker), each pinned to its own GPU(s)
    and pointed at its paired SGLang URL + adapter dir.

The orchestrator then maintains an asyncio.Queue of problem IDs and dispatches
them to whichever worker becomes free. Each worker, once given a problem, runs
num_turns of rollout+grpo+hot-swap *within its pair* and reports per-turn
metrics back via stdout.

All wandb / events.jsonl logging happens in the orchestrator (single writer);
workers only stream JSON lines describing what to log.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import List

import httpx

from .config import Config
from .kernel_env import KernelEnv
from .logger import JsonlLogger


def _bind_free_port(start: int, used: set[int]) -> int:
    """Probe for a free TCP port starting at `start`, skipping any in `used`."""
    p = start
    while True:
        if p in used:
            p += 1
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                p += 1


async def _wait_sglang_ready(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.time() < deadline:
            try:
                r = await client.get(f"{url}/v1/models")
                if r.status_code == 200:
                    return
            except Exception as e:  # noqa: BLE001
                last_err = e
            await asyncio.sleep(2.0)
    raise RuntimeError(f"SGLang at {url} not ready in {timeout_s}s: {last_err}")


class WorkerHandle:
    def __init__(self, idx: int, proc: asyncio.subprocess.Process):
        self.idx = idx
        self.proc = proc
        # A single concurrent writer is fine; this lock just keeps two
        # write_cmd() callers from interleaving JSON lines onto the pipe.
        self.write_lock = asyncio.Lock()
        # Set of problem_ids currently in flight on this worker. The scheduler
        # picks the worker with the smallest len(inflight).
        self.inflight: set[int] = set()
        self.dead = False

    async def read_event(self, timeout: float | None = None) -> dict:
        raw = await asyncio.wait_for(self.proc.stdout.readline(), timeout=timeout) \
            if timeout else await self.proc.stdout.readline()
        if not raw:
            raise RuntimeError(f"worker {self.idx} stdout closed")
        return json.loads(raw.decode())

    async def write_cmd(self, msg: dict) -> None:
        async with self.write_lock:
            line = json.dumps(msg) + "\n"
            self.proc.stdin.write(line.encode())
            await self.proc.stdin.drain()


async def _launch_sglang(
    pair_idx: int,
    sampler_gpus: List[int],
    port: int,
    adapter_dir: str,
    sampler_dp: int,
    sampler_tp: int,
    model_name: str,
    log_path: Path,
    repo_root: Path,
) -> tuple[asyncio.subprocess.Process, "io.TextIOBase"]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in sampler_gpus)
    env["MODEL_NAME"] = model_name
    env["ADAPTER_DIR"] = adapter_dir
    env["PORT"] = str(port)
    env["TP"] = str(sampler_tp)
    env["DP"] = str(sampler_dp)
    # Each SGLang process needs its own NCCL/distributed port if dp>1 or tp>1.
    env["SGLANG_PORT_BASE"] = str(port + 1000)
    log_f = open(log_path, "w")
    proc = await asyncio.create_subprocess_exec(
        "bash", str(repo_root / "scripts" / "launch_sglang.sh"),
        env=env, stdout=log_f, stderr=log_f, cwd=str(repo_root),
    )
    return proc, log_f


async def _launch_worker(
    pair_idx: int,
    trainer_gpus: List[int],
    sglang_url: str,
    adapter_dir: str,
    run_dir: Path,
    config_path: str,
    overrides: List[str],
    log_path: Path,
    repo_root: Path,
) -> WorkerHandle:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in trainer_gpus)
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        env.pop(k, None)
    # Make sure the package import resolves.
    env["PYTHONPATH"] = f"{repo_root / 'src'}:" + env.get("PYTHONPATH", "")
    # CRITICAL: each worker needs its OWN torch.utils.cpp_extension cache dir.
    # KernelBench calls load_inline(name="matmul") etc., which is shared by
    # name (not by source hash); two workers compiling different candidate
    # kernels under the same name will deadlock on the file-baton lock or
    # overwrite each other's .so. Per-worker dirs fully isolate them.
    env["TORCH_EXTENSIONS_DIR"] = str(run_dir / f"torch_ext_pair{pair_idx}")

    cmd = [
        sys.executable, "-u", "-m", "ttt_kernel.worker",
        "--config", config_path,
        "--pair-idx", str(pair_idx),
        "--sglang-url", sglang_url,
        "--adapter-dir", adapter_dir,
        "--run-dir", str(run_dir),
    ]
    for ov in overrides:
        cmd.extend(["--override", ov])

    log_f = open(log_path, "w")
    proc = await asyncio.create_subprocess_exec(
        *cmd, env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=log_f,
        cwd=str(repo_root),
    )
    # Stash the stderr log handle on the proc object so it stays alive.
    proc._log_f = log_f  # type: ignore[attr-defined]
    return WorkerHandle(pair_idx, proc)


async def run_pool(cfg: Config, config_path: str, overrides: List[str]) -> None:
    pool = cfg.pool
    n = pool.num_pairs
    if n < 1:
        raise ValueError("pool.num_pairs must be >= 1")

    s_gpus_per = pool.sampler.dp * pool.sampler.tp * pool.sampler.sp
    t_gpus_per = pool.trainer.dp * pool.trainer.tp * pool.trainer.sp
    per_pair = s_gpus_per + t_gpus_per
    total = n * per_pair

    if pool.trainer.dp != 1 or pool.trainer.tp != 1:
        raise NotImplementedError(
            "trainer.dp/tp > 1 not yet wired (would need torchrun inside worker)."
        )
    if pool.sampler.sp != 1 or pool.trainer.sp != 1:
        # sp is reserved; not yet wired through SGLang or torch.
        sys.stderr.write("[orchestrator] WARN: sp>1 has no effect yet; ignoring.\n")

    # Resolve which physical GPUs we can use.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        all_gpus = [int(x) for x in cvd.split(",") if x.strip()]
    else:
        # Count visible GPUs by querying torch (avoids hard-coding).
        try:
            import torch
            all_gpus = list(range(torch.cuda.device_count()))
        except Exception:
            all_gpus = list(range(total))
    if len(all_gpus) < total:
        raise RuntimeError(
            f"Pool needs {total} GPUs ({n} pairs × {per_pair}), but only "
            f"{len(all_gpus)} available: {all_gpus}"
        )

    # Important: once we launch subprocesses, we must NOT keep CUDA_VISIBLE_DEVICES
    # in the parent (so per-pair env wins). We don't import torch here either.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    repo_root = Path(__file__).resolve().parent.parent.parent

    logger = JsonlLogger(
        cfg.logging.out_dir, cfg.logging.run_name,
        wandb_cfg=cfg.logging.wandb, full_config=cfg.model_dump(),
    )
    logger.log("run_start", config=cfg.model_dump())
    run_dir = Path(logger.run_dir)

    # ---- assign per-pair resources ----------------------------------------
    pairs = []
    used_ports: set[int] = set()
    cursor = 0
    for i in range(n):
        s_gpus = all_gpus[cursor : cursor + s_gpus_per]; cursor += s_gpus_per
        t_gpus = all_gpus[cursor : cursor + t_gpus_per]; cursor += t_gpus_per
        port = _bind_free_port(pool.base_port + i, used_ports)
        used_ports.add(port)
        adapter_dir = os.path.abspath(f"{cfg.sglang.adapter_out_dir}_pair{i}")
        pairs.append({
            "idx": i,
            "sampler_gpus": s_gpus,
            "trainer_gpus": t_gpus,
            "port": port,
            "url": f"http://127.0.0.1:{port}",
            "adapter_dir": adapter_dir,
        })
        logger.log("pair_assignment", **pairs[-1])

    # ---- launch SGLang processes ------------------------------------------
    logger.log("sglang_launching", n=n)
    sglang_records = []
    for p in pairs:
        proc, log_f = await _launch_sglang(
            p["idx"], p["sampler_gpus"], p["port"], p["adapter_dir"],
            pool.sampler.dp, pool.sampler.tp,
            cfg.model.name,
            run_dir / f"sglang_pair{p['idx']}.log",
            repo_root,
        )
        sglang_records.append((proc, log_f, p))

    try:
        await asyncio.gather(*[_wait_sglang_ready(p["url"], 1800.0) for p in pairs])
        logger.log("sglang_ready")

        # ---- launch worker subprocesses -----------------------------------
        workers: List[WorkerHandle] = []
        for p in pairs:
            w = await _launch_worker(
                p["idx"], p["trainer_gpus"], p["url"], p["adapter_dir"],
                run_dir, config_path, overrides,
                run_dir / f"worker_pair{p['idx']}.log",
                repo_root,
            )
            workers.append(w)
        logger.log("workers_launched", n=len(workers))

        # Wait for each worker to emit a "ready" line.
        async def _wait_ready(w: WorkerHandle):
            while True:
                ev = await w.read_event(timeout=1800.0)
                if ev.get("kind") == "ready":
                    return
                if ev.get("kind") == "fatal":
                    raise RuntimeError(f"worker {w.idx} fatal: {ev}")
                # Drain any other early events into the logger.
                logger.log("worker_event", **ev)

        await asyncio.gather(*[_wait_ready(w) for w in workers])
        logger.log("workers_ready")

        # ---- build the problem queue --------------------------------------
        env_main = KernelEnv(cfg.kernelbench, cfg.reward)
        problem_ids = env_main.list_problem_ids()
        logger.log("problems", level=cfg.kernelbench.level, problem_ids=problem_ids)

        queue: asyncio.Queue = asyncio.Queue()
        for pid in problem_ids:
            queue.put_nowait(pid)

        # Progress + cumulative-rollout counters (single-threaded asyncio: safe).
        progress = {"total": len(problem_ids), "done": 0, "correct": 0, "failed": 0}
        rollouts_total = {"n": 0, "truncated": 0, "compiled": 0, "correct": 0}

        # Signaled whenever a problem completes so the scheduler can wake up
        # and re-check capacity (avoids busy-wait sleep loops).
        slot_event = asyncio.Event()
        max_inflight = max(1, int(pool.max_inflight_per_pair))

        def _alive() -> list[WorkerHandle]:
            return [w for w in workers if not w.dead]

        def _least_loaded() -> WorkerHandle | None:
            alive = [w for w in _alive() if len(w.inflight) < max_inflight]
            if not alive:
                return None
            return min(alive, key=lambda w: len(w.inflight))

        # ---- per-worker event listener ------------------------------------
        async def listen(w: WorkerHandle):
            try:
                while True:
                    ev = await w.read_event()
                    kind = ev.get("kind")
                    if kind == "turn":
                        logger.log("turn", **ev)
                        rollouts_total["n"] += int(ev.get("n_rollouts", 0))
                        rollouts_total["truncated"] += int(ev.get("n_truncated", 0))
                        rollouts_total["compiled"] += int(ev.get("n_compiled", 0))
                        rollouts_total["correct"] += int(ev.get("n_correct", 0))
                        n_tot = max(rollouts_total["n"], 1)
                        logger.log(
                            "rollouts_cum",
                            rollouts_total=rollouts_total["n"],
                            rollouts_truncated=rollouts_total["truncated"],
                            rollouts_compiled=rollouts_total["compiled"],
                            rollouts_correct=rollouts_total["correct"],
                            frac_truncated=rollouts_total["truncated"] / n_tot,
                            frac_compiled=rollouts_total["compiled"] / n_tot,
                            frac_correct=rollouts_total["correct"] / n_tot,
                        )
                    elif kind == "done":
                        pid = int(ev.get("problem_id", -1))
                        logger.log("problem_done", **ev)
                        w.inflight.discard(pid)
                        if ev.get("error"):
                            progress["failed"] += 1
                        else:
                            progress["done"] += 1
                            if ev.get("any_correct"):
                                progress["correct"] += 1
                        logger.log(
                            "progress",
                            problems_done=progress["done"],
                            problems_correct=progress["correct"],
                            problems_failed=progress["failed"],
                            problems_total=progress["total"],
                            frac_done=progress["done"] / max(progress["total"], 1),
                        )
                        slot_event.set()
                    elif kind == "error":
                        logger.log("worker_error", pair=w.idx, **ev)
                    elif kind == "fatal":
                        logger.log("worker_fatal", pair=w.idx, **ev)
                        w.dead = True
                        slot_event.set()
                        break
                    else:
                        logger.log("worker_event", **ev)
            except Exception as e:
                # EOF on stdout => worker process is gone. Mark all of its
                # in-flight problems as failed and let the scheduler reassign.
                logger.log("listener_error", pair=w.idx, error=str(e))
                w.dead = True
                for pid in list(w.inflight):
                    progress["failed"] += 1
                    logger.log("problem_done", pair=w.idx, problem_id=pid,
                               error="worker died", any_correct=False)
                    logger.log("progress",
                               problems_done=progress["done"],
                               problems_correct=progress["correct"],
                               problems_failed=progress["failed"],
                               problems_total=progress["total"],
                               frac_done=progress["done"] / max(progress["total"], 1))
                w.inflight.clear()
                slot_event.set()

        listener_tasks = [asyncio.create_task(listen(w), name=f"listen-{w.idx}")
                          for w in workers]

        # ---- central scheduler --------------------------------------------
        async def scheduler():
            while not queue.empty():
                pid = await queue.get()
                # Wait for a pair with capacity. slot_event is set by `listen`
                # on every `done` event, so this resolves the moment a slot
                # frees up — no polling.
                while True:
                    w = _least_loaded()
                    if w is not None:
                        break
                    if not _alive():
                        # All workers dead → drain the queue as failed.
                        progress["failed"] += 1
                        logger.log("problem_done", problem_id=pid,
                                   error="all workers dead", any_correct=False)
                        logger.log("progress",
                                   problems_done=progress["done"],
                                   problems_correct=progress["correct"],
                                   problems_failed=progress["failed"],
                                   problems_total=progress["total"],
                                   frac_done=progress["done"] / max(progress["total"], 1))
                        queue.task_done()
                        return
                    slot_event.clear()
                    await slot_event.wait()
                w.inflight.add(pid)
                logger.log("problem_dispatched", pair=w.idx, problem_id=pid,
                           pair_inflight=len(w.inflight))
                try:
                    await w.write_cmd({"cmd": "process", "problem_id": pid})
                except Exception as e:
                    logger.log("dispatch_error", pair=w.idx, problem_id=pid, error=str(e))
                    w.inflight.discard(pid)
                    w.dead = True
                    slot_event.set()
                    # Requeue this problem for someone else.
                    queue.put_nowait(pid)
                queue.task_done()

            # Drain phase: wait for every dispatched problem to finish.
            while any(w.inflight for w in _alive()):
                slot_event.clear()
                await slot_event.wait()

        await scheduler()
        logger.log("queue_drained")

        # ---- stop listeners + shut down workers ---------------------------
        for t in listener_tasks:
            t.cancel()
        await asyncio.gather(*listener_tasks, return_exceptions=True)
        for w in workers:
            try:
                await w.write_cmd({"cmd": "exit"})
            except Exception:
                pass
        for w in workers:
            try:
                await asyncio.wait_for(w.proc.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                w.proc.kill()
    finally:
        # ---- terminate SGLang processes -----------------------------------
        for proc, log_f, _p in sglang_records:
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        for proc, log_f, _p in sglang_records:
            try:
                await asyncio.wait_for(proc.wait(), timeout=20.0)
            except asyncio.TimeoutError:
                proc.kill()
            log_f.close()

        logger.log("run_end")
        logger.close()


def run(cfg: Config, config_path: str, overrides: List[str]) -> None:
    asyncio.run(run_pool(cfg, config_path, overrides))
