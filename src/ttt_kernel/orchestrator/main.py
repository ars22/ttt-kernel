"""Orchestrator entrypoint.

Reads the YAML config, polls the filesystem registry for each pool until
expected counts arrive, builds the three Pool[T]s, materializes v000 seed
adapters, and fans problems out across the per-problem state machine
coroutines.

Launch:
    python -m ttt_kernel.orchestrator.main \
        --config configs/default.yaml \
        --run-root runs/refactor_smoke \
        --num-samplers 1 --num-envs 1 --num-trainers 1
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel

from ..env.client import EnvClient
from ..sampler.client import SamplerClient
from ..shared.adapter_paths import seed as _seed
from ..shared.logger import JsonlLogger
from ..trainer.client import TrainerClient
from .problem_sm import run_problem
from .registry import read_entries, wait_for_pool
from .scheduler import build_pool
from .seed import materialize_seeds

log = logging.getLogger("ttt_kernel.orchestrator.main")


class _WandbCfg(BaseModel):
    enabled: bool = False
    project: str = "ttt-kernel"
    entity: str | None = None
    run_name: str | None = None
    mode: str = "online"
    tags: list[str] = []


async def _fetch_prompts_via_first_env(env_client: EnvClient, problem_ids: list[int]) -> dict[int, str]:
    out = {}
    for pid in problem_ids:
        out[pid] = await env_client.get_prompt(pid)
    return out


async def _wait_for_healthz(pools: list[tuple[str, "object"]], *, timeout_s: float) -> None:
    """Poll /healthz on each pool member until all return True (or timeout).

    The filesystem registry says "advertised" but services finish startup
    work (sandbox spawn / model load / SGLang ready) inside the FastAPI
    lifespan, which only completes once uvicorn starts serving. So after
    seeing the registry file we still need to wait on /healthz.
    """
    import time
    deadline = time.time() + timeout_s
    pending: list[tuple[str, object]] = [(kind, m) for kind, pool in pools for m in pool.members]
    last_log = 0.0
    while pending:
        now = time.time()
        if now > deadline:
            still = ", ".join(f"{k}/{m.entry.idx}@{m.entry.host}:{m.entry.port}" for k, m in pending)
            raise TimeoutError(f"/healthz did not return OK within {timeout_s}s for: {still}")
        still_pending: list[tuple[str, object]] = []
        for kind, m in pending:
            try:
                ok = await m.client.healthz()
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                log.info("%s/%03d healthy at %s:%d",
                         kind, m.entry.idx, m.entry.host, m.entry.port)
            else:
                still_pending.append((kind, m))
        pending = still_pending
        if pending and now - last_log > 30:
            still = ", ".join(f"{k}/{m.entry.idx}" for k, m in pending)
            log.info("still waiting on /healthz: %s", still)
            last_log = now
        if pending:
            await asyncio.sleep(2.0)


async def _drive(
    *,
    cfg: dict,
    run_root: Path,
    num_samplers: int,
    num_envs: int,
    num_trainers: int,
    problem_ids_override: List[int] | None,
    seed_skip: bool,
) -> None:
    # ---- wait for the three pools to register --------------------------
    log.info("waiting for pools to register at %s/registry", run_root)
    sampler_entries = await asyncio.to_thread(wait_for_pool, run_root, "sampler", num_samplers)
    env_entries = await asyncio.to_thread(wait_for_pool, run_root, "env", num_envs)
    trainer_entries = await asyncio.to_thread(wait_for_pool, run_root, "trainer", num_trainers)

    # ---- choose problem set (read from first env service) --------------
    sample_kwargs = {
        "temperature": float(cfg["rollout"].get("temperature", 1.0)),
        "top_p": float(cfg["rollout"].get("top_p", 0.95)),
        "max_tokens": int(cfg["rollout"].get("max_tokens", 16384)),
    }
    K = int(cfg["rollout"].get("num_samples", 8))
    num_turns = int(cfg["loop"].get("num_turns", 5))
    adapters_root = str(run_root / "adapters")

    # ---- build pools ----------------------------------------------------
    sampler_pool = build_pool(sampler_entries, lambda url: SamplerClient(url))
    env_pool = build_pool(env_entries, lambda url: EnvClient(url))
    trainer_pool = build_pool(trainer_entries, lambda url: TrainerClient(url))

    # ---- wait for each registered service to actually serve /healthz ----
    # Services write to the registry in main() BEFORE uvicorn lifespan starts
    # (env spawns 12 sandbox subprocs; sampler waits on SGLang; trainer loads
    # the base model). The registry says "up" the moment the entry is written;
    # /healthz only returns true once the lifespan has finished startup.
    log.info("probing /healthz on each pool member…")
    await _wait_for_healthz([
        ("sampler", sampler_pool),
        ("env", env_pool),
        ("trainer", trainer_pool),
    ], timeout_s=1800.0)

    try:
        # ---- collect the problem id list from env --------------------------
        e_entry = env_pool.members[0].entry
        async with EnvClient(f"http://{e_entry.host}:{e_entry.port}") as e0:
            if problem_ids_override is not None:
                pids = list(problem_ids_override)
            else:
                pids = await e0.list_problems()
            prompts_by_pid = await _fetch_prompts_via_first_env(e0, pids)

        # ---- seed v000 for each problem ------------------------------------
        if not seed_skip:
            model_cfg = cfg["model"]
            lora_cfg = cfg.get("lora", {})
            await asyncio.to_thread(
                materialize_seeds,
                adapters_root, pids,
                model_name=model_cfg["name"],
                lora_r=int(lora_cfg.get("r", 16)),
                lora_alpha=int(lora_cfg.get("alpha", 32)),
                target_modules=lora_cfg.get("target_modules", [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]),
                bias=lora_cfg.get("bias", "none"),
                dtype=model_cfg.get("dtype", "bfloat16"),
                trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            )
        else:
            log.warning("seed bootstrap skipped (--seed-skip); v000 must already exist")
            for pid in pids:
                if not (_seed(adapters_root, pid).path / "adapter_config.json").exists():
                    raise FileNotFoundError(
                        f"--seed-skip set but v000 missing for problem {pid} at "
                        f"{_seed(adapters_root, pid).path}"
                    )

        # ---- logger --------------------------------------------------------
        wandb_cfg_raw = cfg.get("logging", {}).get("wandb", {})
        wandb_cfg = _WandbCfg(**wandb_cfg_raw)
        out_dir = cfg.get("logging", {}).get("out_dir", str(run_root / "logs"))
        with JsonlLogger(
            out_dir, run_name=cfg.get("logging", {}).get("run_name"),
            wandb_cfg=wandb_cfg, full_config=cfg,
        ) as logger:
            async def _fetch_prompt(pid: int) -> str:
                return prompts_by_pid[pid]

            # ---- fan problems out ------------------------------------------
            tasks = [
                run_problem(
                    problem_id=pid,
                    num_turns=num_turns,
                    K=K,
                    sampler_pool=sampler_pool,
                    env_pool=env_pool,
                    trainer_pool=trainer_pool,
                    adapters_root=adapters_root,
                    base_prompt_fetcher=_fetch_prompt,
                    logger=logger,
                    sample_kwargs=sample_kwargs,
                )
                for pid in pids
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for pid, res in zip(pids, results):
                if isinstance(res, BaseException):
                    import traceback as _tb
                    tb_str = "".join(_tb.format_exception(type(res), res, res.__traceback__))
                    log.error("problem %d failed: %s: %r\n%s",
                              pid, type(res).__name__, res, tb_str)
                    logger.log("problem_failed",
                               problem_id=pid,
                               error_type=type(res).__name__,
                               error_repr=repr(res),
                               error_str=str(res),
                               traceback=tb_str)
    finally:
        await sampler_pool.close()
        await env_pool.close()
        await trainer_pool.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-root", required=True,
                   help="Top-level run directory; pools register under <run-root>/registry/.")
    p.add_argument("--num-samplers", type=int, required=True)
    p.add_argument("--num-envs", type=int, required=True)
    p.add_argument("--num-trainers", type=int, required=True)
    p.add_argument("--problem-ids", type=str, default=None,
                   help="Optional comma-separated overrides for problem_ids "
                        "(else read from the env service /problems).")
    p.add_argument("--seed-skip", action="store_true",
                   help="Skip v000 materialization (require pre-existing seeds).")
    p.add_argument("--log-level", default="info")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))

    pids_override = None
    if args.problem_ids:
        pids_override = [int(x) for x in args.problem_ids.split(",") if x.strip()]

    asyncio.run(_drive(
        cfg=cfg,
        run_root=run_root,
        num_samplers=args.num_samplers,
        num_envs=args.num_envs,
        num_trainers=args.num_trainers,
        problem_ids_override=pids_override,
        seed_skip=args.seed_skip,
    ))


if __name__ == "__main__":
    main()
