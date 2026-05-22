"""Multitask RL orchestrator (full-model REINFORCE baseline).

Per step:
  1. Pick N problems uniformly without replacement from the full list.
  2. For each: sample K rollouts via SGLang (current weights), evaluate
     via env services. All N×K rollouts collected as flat list.
  3. POST /train_base on the (single) trainer with group_ids = problem
     index in the batch. The trainer runs one REINFORCE step then
     broadcasts new weights to SGLang before returning.
  4. Loop num_steps times.

Layout assumption: ONE trainer (1 service × 8 GPUs FSDP), ONE sampler
shim sitting in front of ONE SGLang server, and M env services.

Launch:
    python -m ttt_kernel.orchestrator.multitask \
        --config configs/multitask_qwen3_32b.yaml \
        --run-root runs/multitask_smoke \
        --num-envs 4
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
import time
from pathlib import Path
from typing import List

import yaml

from ..env.client import EnvClient
from ..sampler.client import SamplerClient
from ..shared.logger import JsonlLogger
from ..shared.types import (
    EvaluateRequest,
    Rollout,
    SampleRequest,
    TrainBaseRequest,
)
from ..trainer.client import TrainerClient
from .registry import wait_for_pool
from .scheduler import build_pool

log = logging.getLogger("ttt_kernel.orchestrator.multitask")


async def _wait_healthz(pools, *, timeout_s: float) -> None:
    import time as _time
    deadline = _time.time() + timeout_s
    pending = [(kind, m) for kind, pool in pools for m in pool.members]
    while pending:
        if _time.time() > deadline:
            still = ", ".join(f"{k}/{m.entry.idx}" for k, m in pending)
            raise TimeoutError(f"/healthz timeout for: {still}")
        still = []
        for kind, m in pending:
            try:
                ok = await m.client.healthz()
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                log.info("%s/%03d healthy at %s:%d",
                         kind, m.entry.idx, m.entry.host, m.entry.port)
            else:
                still.append((kind, m))
        pending = still
        if pending:
            await asyncio.sleep(2.0)


async def _drive(
    *,
    cfg: dict,
    run_root: Path,
    num_envs: int,
    sglang_url_for_trainer: str,
    trainer_master_addr: str,
) -> None:
    # Multitask uses 1 sampler, M envs, 1 trainer.
    sampler_entries = await asyncio.to_thread(wait_for_pool, run_root, "sampler", 1)
    env_entries = await asyncio.to_thread(wait_for_pool, run_root, "env", num_envs)
    trainer_entries = await asyncio.to_thread(wait_for_pool, run_root, "trainer", 1)

    sampler_pool = build_pool(sampler_entries, lambda url: SamplerClient(url))
    env_pool = build_pool(env_entries, lambda url: EnvClient(url))
    trainer_pool = build_pool(trainer_entries, lambda url: TrainerClient(url))

    log.info("probing /healthz on each pool member…")
    await _wait_healthz(
        [("sampler", sampler_pool), ("env", env_pool), ("trainer", trainer_pool)],
        timeout_s=1800.0,
    )

    mt = cfg.get("multitask", {})
    rollout_cfg = cfg["rollout"]
    sglang_cfg = cfg.get("sglang", {})

    N = int(mt.get("problems_per_step", 32))
    K = int(rollout_cfg.get("num_samples", 8))
    num_steps = int(mt.get("num_steps", 200))
    weight_sync = str(mt.get("weight_sync", "nccl"))
    seed = int(cfg.get("loop", {}).get("seed", 0))
    sample_kwargs = {
        "temperature": float(rollout_cfg.get("temperature", 1.0)),
        "top_p": float(rollout_cfg.get("top_p", 0.95)),
        "max_tokens": int(rollout_cfg.get("max_tokens", 16384)),
    }

    sglang_tp = int(sglang_cfg.get("tp_size", 4))
    master_port = int(mt.get("weight_sync_master_port", 29600))

    try:
        # ---- get the problem list -----------------------------------------
        e_entry = env_pool.members[0].entry
        async with EnvClient(f"http://{e_entry.host}:{e_entry.port}") as e0:
            all_pids: List[int] = await e0.list_problems()
            prompts_by_pid = {pid: await e0.get_prompt(pid) for pid in all_pids}
        log.info("loaded %d problems", len(all_pids))

        # ---- bootstrap the trainer↔SGLang weight-sync channel -------------
        t_entry = trainer_pool.members[0].entry
        ckpt_root = str(run_root / "ckpts")
        log.info("init_broadcast: weight_sync=%s ckpt_root=%s", weight_sync, ckpt_root)
        async with TrainerClient(f"http://{t_entry.host}:{t_entry.port}") as tcli:
            if weight_sync == "nccl":
                await tcli.init_broadcast(
                    sglang_base_url=sglang_url_for_trainer,
                    sglang_tp=sglang_tp,
                    master_address=trainer_master_addr,
                    master_port=master_port,
                    weight_sync="nccl",
                )
            elif weight_sync == "disk":
                await tcli.init_broadcast(
                    sglang_base_url=sglang_url_for_trainer,
                    weight_sync="disk",
                    ckpt_root=ckpt_root,
                )
            else:
                raise ValueError(f"unsupported weight_sync: {weight_sync!r}")
        log.info("init_broadcast OK (%s)", weight_sync)

        # ---- logger -------------------------------------------------------
        out_dir = cfg.get("logging", {}).get("out_dir", str(run_root / "logs"))
        rng = random.Random(seed)
        # logger expects a wandb cfg model; reuse the same one as orchestrator.main
        from .main import _WandbCfg
        wandb_cfg = _WandbCfg(**cfg.get("logging", {}).get("wandb", {}))

        with JsonlLogger(
            out_dir, run_name=cfg.get("logging", {}).get("run_name"),
            wandb_cfg=wandb_cfg, full_config=cfg,
            inference_only=False,
        ) as logger:
            for step in range(num_steps):
                t_start = time.monotonic()
                batch_pids = rng.sample(all_pids, min(N, len(all_pids)))
                logger.log("step_start", step=step, problem_ids=batch_pids)

                # ---- fan out sample + eval per (problem, k) ---------------
                # Each problem becomes a coroutine that samples K rollouts on
                # the (one) sampler then evaluates each on whichever env is free.
                rollouts: list[Rollout] = []
                group_ids: list[int] = []
                rewards_by_pid: dict[int, list[float]] = {}

                async def _do_problem(idx: int, pid: int):
                    prompt = prompts_by_pid[pid]
                    async with sampler_pool.acquire() as (sclient, _sentry):
                        sreq = SampleRequest(
                            problem_id=pid,
                            turn=step,
                            prompt=prompt,
                            adapter_path="",
                            adapter_name="",
                            n=K,
                            **sample_kwargs,
                        )
                        sresp = await sclient.sample(sreq)
                    completions = [g.text for g in sresp.completions]

                    async def _eval_one(j: int, c: str):
                        async with env_pool.acquire() as (eclient, _e):
                            return await eclient.evaluate(EvaluateRequest(
                                problem_id=pid, turn=step, sample=j, completion=c,
                            ))
                    ev = await asyncio.gather(*[
                        _eval_one(j, c) for j, c in enumerate(completions)
                    ])
                    rewards_by_pid[pid] = [float(r.reward) for r in ev]
                    return idx, pid, prompt, completions, ev

                results = await asyncio.gather(*[
                    _do_problem(i, pid) for i, pid in enumerate(batch_pids)
                ])

                n_compiled = 0
                n_correct = 0
                for idx, pid, prompt, completions, ev in results:
                    for c, e in zip(completions, ev):
                        rollouts.append(Rollout(
                            prompt=prompt, completion=c, reward=float(e.reward),
                        ))
                        group_ids.append(idx)
                        if e.compiled:
                            n_compiled += 1
                        if e.correct:
                            n_correct += 1

                sample_ms = (time.monotonic() - t_start) * 1000.0
                logger.log("step_sample_done", step=step,
                           sample_ms=sample_ms,
                           total_rollouts=len(rollouts),
                           n_compiled=n_compiled,
                           n_correct=n_correct,
                           reward_mean=sum(r.reward for r in rollouts) / max(len(rollouts), 1),
                           rewards_by_pid=rewards_by_pid)

                # ---- one REINFORCE step + weight broadcast ----------------
                t_train = time.monotonic()
                async with trainer_pool.acquire() as (tcli, _tentry):
                    tresp = await tcli.train_base(TrainBaseRequest(
                        step=step,
                        rollouts=rollouts,
                        group_ids=group_ids,
                    ))
                train_ms = (time.monotonic() - t_train) * 1000.0

                logger.log("step",
                           step=step,
                           sample_ms=sample_ms,
                           train_ms=train_ms,
                           weight_sync_ms=tresp.weight_sync_ms,
                           **{k: getattr(tresp, k) for k in (
                               "loss", "pg", "kl", "grad_norm",
                               "reward_mean", "reward_std", "advantage_mean",
                           )},
                           n_compiled=n_compiled,
                           n_correct=n_correct)
                log.info(
                    "step %3d/%d  pg=%+.4f  R=%+.3f  ok=%d/%d  sample=%.1fs train=%.1fs sync=%.1fs",
                    step, num_steps, tresp.pg, tresp.reward_mean,
                    n_correct, len(rollouts),
                    sample_ms / 1000.0, train_ms / 1000.0,
                    tresp.weight_sync_ms / 1000.0,
                )
    finally:
        await sampler_pool.close()
        await env_pool.close()
        await trainer_pool.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--num-envs", type=int, required=True)
    p.add_argument("--sglang-url-for-trainer", required=True,
                   help="Base URL of SGLang as reachable from the TRAINER node, "
                        "e.g. http://node0:30100. The trainer hits this URL to "
                        "POST /init_weights_update_group and /update_weights_from_distributed.")
    p.add_argument("--trainer-master-addr", required=True,
                   help="Trainer node hostname/IP visible to SGLang (the master "
                        "for the secondary NCCL group). SGLang will rendezvous "
                        "at tcp://<addr>:<multitask.weight_sync_master_port>.")
    p.add_argument("--log-level", default="info")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))

    asyncio.run(_drive(
        cfg=cfg,
        run_root=run_root,
        num_envs=args.num_envs,
        sglang_url_for_trainer=args.sglang_url_for_trainer,
        trainer_master_addr=args.trainer_master_addr,
    ))


if __name__ == "__main__":
    main()
