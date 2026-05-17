"""FastAPI env service.

Endpoints:
- POST /evaluate            EvaluateRequest → EvaluateResponse
- GET  /healthz             {"ok": true}
- GET  /capacity            Capacity
- GET  /problems            {"problem_ids": [...]}
- GET  /problems/{id}       {"problem_id": ..., "name": ..., "prompt": ...}

Launch:
    python -m ttt_kernel.env.server \
        --config configs/default.yaml \
        --port 8001 \
        --max-concurrent 4

Config fields read from the YAML:
    kernelbench.{repo_path, dataset_src, dataset_name, level, problem_ids,
                 backend, precision, gpu_arch, prompt_option,
                 num_correct_trials, num_perf_trials, timing_method,
                 eval_timeout_s}
    reward.{speedup_log_scale, error_penalty, incorrect_penalty, clip}

CLI flags override config file. The orchestrator drives capacity via /capacity
and routes /evaluate by problem_id; ref_src never crosses the wire.
"""
from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException

from ..shared.types import Capacity, EvaluateRequest, EvaluateResponse
from .pool import EnvPool, SandboxCfg
from .problem_set import ProblemSet, ProblemSetCfg
from .scoring import RewardCfg

log = logging.getLogger("ttt_kernel.env.server")


def _build_problem_set(yaml_kb: dict) -> ProblemSet:
    return ProblemSet(ProblemSetCfg(
        repo_path=yaml_kb["repo_path"],
        dataset_src=yaml_kb.get("dataset_src", "huggingface"),
        dataset_name=yaml_kb.get("dataset_name", "ScalingIntelligence/KernelBench"),
        level=int(yaml_kb.get("level", 1)),
        problem_ids=yaml_kb.get("problem_ids"),
        backend=yaml_kb.get("backend", "cuda"),
        precision=yaml_kb.get("precision", "fp32"),
        gpu_arch=yaml_kb.get("gpu_arch", "Blackwell"),
        prompt_option=yaml_kb.get("prompt_option", "one_shot"),
    ))


def _build_sandbox_cfg(yaml_kb: dict) -> SandboxCfg:
    return SandboxCfg(
        repo_path=yaml_kb["repo_path"],
        gpu_arch=yaml_kb.get("gpu_arch", "Blackwell"),
        backend=yaml_kb.get("backend", "cuda"),
        precision=yaml_kb.get("precision", "fp32"),
        timing_method=yaml_kb.get("timing_method", "cuda_event"),
        num_correct_trials=int(yaml_kb.get("num_correct_trials", 5)),
        num_perf_trials=int(yaml_kb.get("num_perf_trials", 100)),
        eval_timeout_s=int(yaml_kb.get("eval_timeout_s", 120)),
    )


def _build_reward_cfg(yaml_rw: dict) -> RewardCfg:
    return RewardCfg(
        speedup_log_scale=bool(yaml_rw.get("speedup_log_scale", True)),
        error_penalty=float(yaml_rw.get("error_penalty", -1.0)),
        incorrect_penalty=float(yaml_rw.get("incorrect_penalty", -1.0)),
        clip=float(yaml_rw.get("clip", 2.0)),
    )


def build_app(
    config_path: str,
    max_concurrent: int,
    sandbox_log_path: Optional[str] = None,
) -> FastAPI:
    raw = yaml.safe_load(open(config_path))
    kb_cfg = raw["kernelbench"]
    rw_cfg = raw.get("reward", {})

    problem_set = _build_problem_set(kb_cfg)
    sandbox_cfg = _build_sandbox_cfg(kb_cfg)
    reward_cfg = _build_reward_cfg(rw_cfg)
    pool = EnvPool(
        max_concurrent=max_concurrent,
        sandbox_cfg=sandbox_cfg,
        reward_cfg=reward_cfg,
        problem_set=problem_set,
        sandbox_log_path=sandbox_log_path,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN001
        await pool.start()
        try:
            yield
        finally:
            await pool.shutdown()

    app = FastAPI(title="ttt-kernel env pool", lifespan=lifespan)
    app.state.pool = pool
    app.state.problem_set = problem_set

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/capacity", response_model=Capacity)
    async def capacity():
        return Capacity(
            pool="env",
            max_concurrent=pool.max_concurrent,
            in_flight=pool.in_flight,
        )

    @app.get("/problems")
    async def problems():
        return {"problem_ids": problem_set.list_problem_ids()}

    @app.get("/problems/{problem_id}")
    async def problem(problem_id: int):
        try:
            p = problem_set.get_problem(problem_id)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(404, str(e))
        return {"problem_id": p.problem_id, "name": p.name, "prompt": p.prompt}

    @app.post("/evaluate", response_model=EvaluateResponse)
    async def evaluate(req: EvaluateRequest):
        return await pool.evaluate(req)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML with kernelbench.* and reward.*")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-concurrent", type=int, required=True,
                        help="Total subprocess sandbox slots (capacity unit).")
    parser.add_argument("--sandbox-log", default=None,
                        help="Append all sandbox stderr to this file (else discarded).")
    parser.add_argument("--run-root", default=None,
                        help="If set, write a RegistryEntry to <run-root>/registry/env/<idx>.json on startup.")
    parser.add_argument("--idx", type=int, default=0,
                        help="SLURM array index (or 0 for single-node).")
    parser.add_argument("--advertise-host", default=None,
                        help="Hostname to advertise in the registry (default: socket.gethostname()).")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app = build_app(args.config, args.max_concurrent, args.sandbox_log)

    if args.run_root:
        import socket
        from ..orchestrator.registry import write_entry, mark_down
        from ..shared.types import RegistryEntry
        host = args.advertise_host or socket.gethostname()
        entry = RegistryEntry(
            pool="env", idx=args.idx, host=host, port=args.port,
            capacity=args.max_concurrent,
        )
        write_entry(args.run_root, entry)
        try:
            uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
        finally:
            mark_down(args.run_root, "env", args.idx)
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
