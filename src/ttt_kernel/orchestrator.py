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
        # Only one in-flight problem at a time per worker; lock guards the pipe.
        self.lock = asyncio.Lock()

    async def read_event(self, timeout: float | None = None) -> dict:
        raw = await asyncio.wait_for(self.proc.stdout.readline(), timeout=timeout) \
            if timeout else await self.proc.stdout.readline()
        if not raw:
            raise RuntimeError(f"worker {self.idx} stdout closed")
        return json.loads(raw.decode())

    async def write_cmd(self, msg: dict) -> None:
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

        # Track per-problem retry counts so a poisoned worker doesn't ping-pong
        # the same bad kernel to every other pair until they all die too.
        retries: dict[int, int] = {}
        MAX_RETRIES = 1
        dead_workers: set[int] = set()

        # ---- per-worker dispatch loop -------------------------------------
        async def dispatch(w: WorkerHandle):
            while True:
                try:
                    pid = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                if w.idx in dead_workers:
                    # Put it back for someone else and stop.
                    queue.put_nowait(pid)
                    return
                worker_died = False
                got_done = False
                async with w.lock:
                    try:
                        logger.log("problem_dispatched", pair=w.idx, problem_id=pid)
                        await w.write_cmd({"cmd": "process", "problem_id": pid})
                        # Stream events until we see kind=done for this problem.
                        while True:
                            ev = await w.read_event()
                            kind = ev.get("kind")
                            if kind == "turn":
                                logger.log("turn", **ev)
                            elif kind == "done":
                                logger.log("problem_done", **ev)
                                got_done = True
                                if ev.get("worker_exiting"):
                                    worker_died = True
                                break
                            elif kind == "error":
                                logger.log("worker_error", pair=w.idx, problem_id=pid, **ev)
                            else:
                                logger.log("worker_event", **ev)
                    except Exception as e:
                        logger.log("dispatch_error", pair=w.idx, problem_id=pid, error=str(e))
                        worker_died = True
                    finally:
                        queue.task_done()

                if worker_died:
                    dead_workers.add(w.idx)
                    logger.log("worker_dead", pair=w.idx)
                    # Requeue the problem (cap retries to avoid infinite ping-pong).
                    n = retries.get(pid, 0)
                    if not got_done or n < MAX_RETRIES:
                        retries[pid] = n + 1
                        queue.put_nowait(pid)
                        logger.log("problem_requeued", problem_id=pid, retries=retries[pid])
                    return

        await asyncio.gather(*[dispatch(w) for w in workers])
        logger.log("queue_drained")

        # ---- shutdown workers ---------------------------------------------
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
