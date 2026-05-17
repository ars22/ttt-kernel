"""FastAPI trainer service (single GPU here; FSDP2 wrap in task #6).

Endpoints:
- POST /train      TrainRequest → TrainResponse
- GET  /healthz    {ok: true}
- GET  /capacity   Capacity (in_flight = concurrent /train calls in progress)

Launch:
    python -m ttt_kernel.trainer.server \
        --config configs/default.yaml \
        --port 8003 \
        --max-concurrent 4

`max-concurrent` is concurrent /train calls on *different* adapters; calls on
the same adapter serialize via the per-adapter lock in AdapterManager.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException

from ..shared.types import Capacity, Rollout, TrainRequest, TrainResponse
from .adapter_manager import AdapterManager
from .grpo import GRPOStepCfg, RolloutT, grpo_step
from .model import (
    ModelInitCfg,
    build_peft_lora_cfg,
    build_peft_model,
    build_tokenizer,
    init_distributed,
)

log = logging.getLogger("ttt_kernel.trainer.server")


def _build_model_cfg(raw: dict) -> ModelInitCfg:
    m = raw["model"]
    lo = raw.get("lora", {})
    return ModelInitCfg(
        name=m["name"],
        dtype=m.get("dtype", "bfloat16"),
        trust_remote_code=bool(m.get("trust_remote_code", True)),
        r=int(lo.get("r", 16)),
        alpha=int(lo.get("alpha", 32)),
        dropout=float(lo.get("dropout", 0.05)),
        target_modules=tuple(lo.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ])),
        bias=lo.get("bias", "none"),
    )


def _build_grpo_cfg(raw: dict) -> GRPOStepCfg:
    g = raw["grpo"]
    return GRPOStepCfg(
        beta_kl=float(g.get("beta_kl", 0.04)),
        epsilon_clip=float(g.get("epsilon_clip", 0.2)),
        group_advantage_norm=bool(g.get("group_advantage_norm", True)),
        update_epochs=int(g.get("update_epochs", 1)),
        grad_clip=float(g.get("grad_clip", 1.0)),
        micro_batch_size=int(g.get("micro_batch_size", 1)),
        max_seq_len=int(g.get("max_seq_len", 16384)),
    )


def build_app(
    config_path: str,
    max_concurrent: int,
    max_resident_adapters: int,
) -> FastAPI:
    raw = yaml.safe_load(open(config_path))
    model_cfg = _build_model_cfg(raw)
    grpo_cfg = _build_grpo_cfg(raw)
    g = raw["grpo"]
    learning_rate = float(g.get("learning_rate", 1.0e-5))
    weight_decay = float(g.get("weight_decay", 0.0))

    rank, world, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    log.info("trainer init: rank=%d world=%d device=%s", rank, world, device)

    peft_model = build_peft_model(model_cfg, device)
    tokenizer = build_tokenizer(model_cfg)

    manager = AdapterManager(
        peft_model=peft_model,
        peft_config=build_peft_lora_cfg(model_cfg),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_resident=max_resident_adapters,
        device=device,
        rank=rank,
        world=world,
    )

    # capacity semaphore: caps concurrent /train calls in flight.
    sem = asyncio.Semaphore(max_concurrent)
    state = {"in_flight": 0}

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN001
        yield
        # nothing to clean up; PEFT model lives until process exit.

    app = FastAPI(title="ttt-kernel trainer pool", lifespan=lifespan)
    app.state.manager = manager
    app.state.counters = state

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/capacity", response_model=Capacity)
    async def capacity():
        return Capacity(
            pool="trainer",
            max_concurrent=max_concurrent,
            in_flight=state["in_flight"],
        )

    @app.post("/train", response_model=TrainResponse)
    async def train(req: TrainRequest):
        if not req.rollouts:
            raise HTTPException(400, "empty rollouts")
        # Concurrent /train on the same adapter must serialize: A/B + optimizer
        # state is per-adapter, but a single optimizer object can't run two
        # steps at once. Different adapters can proceed in parallel because
        # the base forward is shared (PEFT routes per active adapter).
        await sem.acquire()
        state["in_flight"] += 1
        try:
            await manager.ensure_resident(req.adapter_in_name, req.adapter_in_path)
            # Rename to the output adapter name (it's a new logical version).
            # Simplest implementation: ensure both names; the *_in_name was just
            # used to load the source weights; we train on a fresh out-name
            # initialized to the same weights.
            if req.adapter_in_name != req.adapter_out_name:
                # Reuse the loaded weights by copying them across slots.
                await manager.ensure_resident(req.adapter_out_name, req.adapter_in_path)
            opt = manager.optimizers[req.adapter_out_name]
            params = manager.adapter_params(req.adapter_out_name)
            async with manager.lock_for(req.adapter_out_name):
                metrics = await asyncio.to_thread(
                    grpo_step,
                    peft_model=manager.peft_model,
                    model_call=manager.peft_model,
                    optimizer=opt,
                    tokenizer=tokenizer,
                    adapter_name=req.adapter_out_name,
                    adapter_params=params,
                    rollouts=[
                        RolloutT(prompt=r.prompt, completion=r.completion, reward=r.reward)
                        for r in req.rollouts
                    ],
                    cfg=grpo_cfg,
                    device=device,
                    group_ids=req.group_ids,
                )
            await manager.save(req.adapter_out_name, req.adapter_out_path)
            return TrainResponse(
                loss=metrics["loss"],
                pg=metrics["pg"],
                kl=metrics["kl"],
                grad_norm=metrics["grad_norm"],
                reward_mean=metrics["reward_mean"],
                reward_std=metrics["reward_std"],
                advantage_mean=metrics["advantage_mean"],
            )
        finally:
            state["in_flight"] -= 1
            sem.release()

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--max-concurrent", type=int, required=True,
                   help="Concurrent /train calls (on different adapters).")
    p.add_argument("--max-resident-adapters", type=int, default=8,
                   help="LRU cap on adapters held in GPU memory.")
    p.add_argument("--run-root", default=None,
                   help="If set, write a RegistryEntry to <run-root>/registry/trainer/<idx>.json on startup.")
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--advertise-host", default=None,
                   help="Hostname to advertise in the registry (default: socket.gethostname()).")
    p.add_argument("--log-level", default="info")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app = build_app(args.config, args.max_concurrent, args.max_resident_adapters)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        if args.run_root:
            import socket
            from ..orchestrator.registry import write_entry, mark_down
            from ..shared.types import RegistryEntry
            host = args.advertise_host or socket.gethostname()
            entry = RegistryEntry(
                pool="trainer", idx=args.idx, host=host, port=args.port,
                capacity=args.max_concurrent,
            )
            write_entry(args.run_root, entry)
            try:
                uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
            finally:
                mark_down(args.run_root, "trainer", args.idx)
        else:
            uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    else:
        # Non-rank-0 just blocks here in task #5 (FSDP wiring in task #6 will
        # replace this with a collective wait-loop).
        import time
        while True:
            time.sleep(60)


if __name__ == "__main__":
    main()
