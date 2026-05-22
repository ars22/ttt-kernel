"""FastAPI trainer service.

Two modes:
- Single-GPU (--fsdp not set, WORLD_SIZE=1): rank-0 PEFT model, train runs
  inline in a thread.
- FSDP2 multi-GPU (launched under torchrun --nproc-per-node=N): each rank
  builds its FSDP-wrapped shard, all ranks participate in every forward via
  the Dispatcher (rank 0 owns HTTP; non-rank ranks run the collective loop).

Endpoints:
- POST /train      TrainRequest → TrainResponse
- GET  /healthz    {ok: true}
- GET  /capacity   Capacity (in_flight = concurrent /train calls in progress)

Launch single-GPU:
    python -m ttt_kernel.trainer.server --config configs/default.yaml \\
        --port 8003 --max-concurrent 2

Launch FSDP2 multi-GPU:
    torchrun --nproc-per-node=8 -m ttt_kernel.trainer.server --fsdp \\
        --config configs/default.yaml --port 8003 --max-concurrent 2

`max-concurrent` caps in-flight /train calls; calls on the same adapter
serialize via the per-adapter lock in AdapterManager.
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
from pydantic import BaseModel

from ..shared.types import (
    Capacity,
    TrainBaseRequest,
    TrainBaseResponse,
    TrainRequest,
    TrainResponse,
)
from .adapter_manager import AdapterManager
from .dispatcher import Dispatcher
from .grpo import GRPOStepCfg, RolloutT, grpo_step
from .model import (
    ModelInitCfg,
    build_peft_lora_cfg,
    build_peft_model,
    build_tokenizer,
    init_distributed,
)

log = logging.getLogger("ttt_kernel.trainer.server")


class _InitBroadcastReq(BaseModel):
    """Body for /init_broadcast — kept at module level so FastAPI's type
    introspection can resolve it (Pydantic v2 forward refs defined inside
    a function body trip TypeAdapter('not fully defined')).
    """
    sglang_base_url: str
    sglang_tp: int = 0                  # ignored when weight_sync='disk'
    master_address: str = ""             # ignored when weight_sync='disk'
    master_port: int = 0                  # ignored when weight_sync='disk'
    group_name: str = "ttt_weight_update"
    # 'nccl' (broken on this cluster; needs further debugging) or 'disk'.
    weight_sync: str = "disk"
    # For weight_sync='disk': where the trainer writes step_<N>/ HF dirs.
    # Must be readable by SGLang too (i.e., a shared FS path).
    ckpt_root: str = ""


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
        group_advantage_norm=bool(g.get("group_advantage_norm", True)),
        update_epochs=int(g.get("update_epochs", 1)),
        grad_clip=float(g.get("grad_clip", 1.0)),
        micro_batch_size=int(g.get("micro_batch_size", 1)),
        max_seq_len=int(g.get("max_seq_len", 16384)),
    )


def _train_step_handler(manager, tokenizer, grpo_cfg, device):
    """Build the collective handler that runs on EVERY rank for a /train call.

    Inputs come via the dispatcher payload (broadcasted by rank 0):
        adapter_in_path, adapter_in_name, adapter_out_path, adapter_out_name,
        rollouts (list of {prompt, completion, reward}), group_ids (optional).
    Outputs (returned on rank 0; ignored elsewhere): the metrics dict.
    """
    def handler(payload: dict) -> dict:
        # ensure_resident is async; run it on a freshly-created loop here so
        # the dispatcher thread doesn't need to spin up an event loop per call.
        # We don't actually need async here — adapter_manager methods only
        # await the internal locks which the dispatcher already serializes.
        ai_name = payload["adapter_in_name"]
        ai_path = payload["adapter_in_path"]
        ao_name = payload["adapter_out_name"]
        ao_path = payload["adapter_out_path"]
        rollouts = [
            RolloutT(prompt=r["prompt"], completion=r["completion"], reward=float(r["reward"]))
            for r in payload["rollouts"]
        ]
        group_ids = payload.get("group_ids")

        # Sync versions of the manager API (we're in a worker thread; locks
        # aren't needed because the dispatcher serializes all collective ops).
        _ensure_resident_sync(manager, ai_name, ai_path)
        if ai_name != ao_name:
            _ensure_resident_sync(manager, ao_name, ai_path)
        opt = manager.optimizers[ao_name]
        params = manager.adapter_params(ao_name)
        metrics = grpo_step(
            peft_model=manager.peft_model,
            model_call=manager.peft_model,
            optimizer=opt,
            tokenizer=tokenizer,
            adapter_name=ao_name,
            adapter_params=params,
            rollouts=rollouts,
            cfg=grpo_cfg,
            device=device,
            group_ids=group_ids,
        )
        _save_sync(manager, ao_name, ao_path)
        return metrics
    return handler


def _ensure_resident_sync(manager: AdapterManager, name: str, source_dir: str) -> None:
    """Synchronous variant of AdapterManager.ensure_resident for collective workers.
    We're already serialized by the dispatcher, so we don't need the async lock."""
    from .adapter_manager import (
        _adapter_params, _load_adapter_weights_from_dir, _zero_init_adapter,
    )
    if name in manager.optimizers:
        manager.optimizers.move_to_end(name)
        return
    while len(manager.optimizers) >= manager.max_resident:
        evict_name, opt = manager.optimizers.popitem(last=False)
        opt.zero_grad(set_to_none=True)
        try:
            manager.peft_model.delete_adapter(evict_name)
        except Exception:  # noqa: BLE001
            pass
    manager.peft_model.add_adapter(name, manager.peft_config)
    manager.peft_model.set_adapter(name)
    for n, p in manager.peft_model.named_parameters():
        if f".{name}." in n and ("lora_A" in n or "lora_B" in n):
            p.requires_grad = True
    _zero_init_adapter(manager.peft_model, name)
    if source_dir is not None:
        _load_adapter_weights_from_dir(manager.peft_model, name, source_dir)
    params = _adapter_params(manager.peft_model, name)
    if not params:
        raise RuntimeError(f"no trainable params for adapter '{name}'")
    manager.optimizers[name] = torch.optim.AdamW(
        params, lr=manager.lr, weight_decay=manager.wd,
    )


def _save_sync(manager: AdapterManager, name: str, out_dir: str) -> None:
    """Synchronous variant of AdapterManager.save (no event loop)."""
    import torch.distributed as dist
    os.makedirs(out_dir, exist_ok=True)
    if manager.world > 1:
        from .fsdp_save import save_adapter_fsdp
        save_adapter_fsdp(manager.peft_model, name, out_dir, manager.peft_config, rank=manager.rank)
        if dist.is_initialized():
            dist.barrier()
        return
    if manager.rank == 0:
        tmp = out_dir + ".tmp"
        if os.path.isdir(tmp):
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        manager.peft_model.save_pretrained(tmp, selected_adapters=[name])
        if os.path.isdir(out_dir):
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
        os.rename(tmp, out_dir)
    if manager.world > 1 and dist.is_initialized():
        dist.barrier()


def _train_base_handler(model, tokenizer, optimizer, grpo_cfg, device):
    """Collective handler for /train_base.

    Runs on EVERY rank. Payload: rollouts (list of dicts) + group_ids.
    Returns the metrics dict (rank 0 only). No adapter args — the base
    model itself is the policy.
    """
    def handler(payload: dict) -> dict:
        rollouts = [
            RolloutT(prompt=r["prompt"], completion=r["completion"], reward=float(r["reward"]))
            for r in payload["rollouts"]
        ]
        group_ids = payload.get("group_ids")
        params = [p for p in model.parameters() if p.requires_grad]
        metrics = grpo_step(
            peft_model=None,
            model_call=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            adapter_name=None,
            adapter_params=params,
            rollouts=rollouts,
            cfg=grpo_cfg,
            device=device,
            group_ids=group_ids,
        )
        return metrics
    return handler


def build_app(
    config_path: str,
    max_concurrent: int,
    max_resident_adapters: int,
    use_fsdp: bool = False,
    full_model: bool = False,
) -> tuple[FastAPI, Dispatcher | None]:
    raw = yaml.safe_load(open(config_path))
    model_cfg = _build_model_cfg(raw)
    grpo_cfg = _build_grpo_cfg(raw)
    g = raw["grpo"]
    learning_rate = float(g.get("learning_rate", 1.0e-5))
    weight_decay = float(g.get("weight_decay", 0.0))

    rank, world, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    log.info("trainer init: rank=%d world=%d device=%s fsdp=%s full_model=%s",
             rank, world, device, use_fsdp, full_model)

    # ---- model construction ------------------------------------------------
    # Three modes:
    #   A) full_model + world>1: FSDP-wrapped bare causal LM (multitask path).
    #   B) PEFT + use_fsdp + world>1: per-problem LoRA path under FSDP.
    #   C) PEFT, single GPU.
    base_model = None      # full-model path
    peft_model = None      # PEFT path
    optimizer = None       # full-model path
    manager: Optional[AdapterManager] = None  # PEFT path

    tokenizer = build_tokenizer(model_cfg)

    if full_model:
        from .full_model import build_fsdp_full_model
        if world == 1:
            # Allow single-GPU full-model for smoke tests; build_fsdp_full_model
            # still wraps but with a 1-rank mesh which is a no-op shard.
            log.warning("full-model path with world=1: FSDP wrap is a no-op")
        base_model = build_fsdp_full_model(model_cfg, rank, world, local_rank)
        optimizer = torch.optim.AdamW(
            [p for p in base_model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif use_fsdp and world > 1:
        from .fsdp_model import build_fsdp_peft_model
        peft_model = build_fsdp_peft_model(model_cfg, rank, world, local_rank)
    else:
        peft_model = build_peft_model(model_cfg, device)

    if not full_model:
        assert peft_model is not None
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

    # ---- dispatcher (collective broadcast for FSDP) ------------------------
    dispatcher: Dispatcher | None = None
    if world > 1:
        handlers: dict[str, object] = {}
        if full_model:
            handlers["train_base"] = _train_base_handler(
                base_model, tokenizer, optimizer, grpo_cfg, device,
            )
        else:
            handlers["train"] = _train_step_handler(manager, tokenizer, grpo_cfg, device)
        dispatcher = Dispatcher(rank, world, handlers=handlers)

    sem = asyncio.Semaphore(max_concurrent)
    state = {"in_flight": 0}
    # Lazily created on first /init_broadcast call. Trainer only.
    broadcaster_holder: dict = {"broadcaster": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN001
        if dispatcher is not None:
            dispatcher.bind_loop(asyncio.get_running_loop())
            dispatcher.start()
        try:
            yield
        finally:
            if dispatcher is not None:
                dispatcher.stop_blocking()
            b = broadcaster_holder.get("broadcaster")
            if b is not None:
                await b.aclose()

    app = FastAPI(title="ttt-kernel trainer pool", lifespan=lifespan)
    if manager is not None:
        app.state.manager = manager
    app.state.counters = state
    app.state.full_model = full_model

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
        if full_model:
            raise HTTPException(400, "trainer is in --full-model mode; use /train_base")
        if not req.rollouts:
            raise HTTPException(400, "empty rollouts")
        await sem.acquire()
        state["in_flight"] += 1
        try:
            if dispatcher is not None:
                # FSDP path: serialize via the collective. All ranks execute
                # the same train step in lockstep; only rank 0 returns metrics.
                fut = dispatcher.submit("train", {
                    "adapter_in_path": req.adapter_in_path,
                    "adapter_in_name": req.adapter_in_name,
                    "adapter_out_path": req.adapter_out_path,
                    "adapter_out_name": req.adapter_out_name,
                    "rollouts": [r.model_dump() for r in req.rollouts],
                    "group_ids": req.group_ids,
                })
                metrics = await fut
            else:
                # Single-GPU path: run inline (with per-adapter serialization).
                await manager.ensure_resident(req.adapter_in_name, req.adapter_in_path)
                if req.adapter_in_name != req.adapter_out_name:
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

    # ---- full-model / multitask endpoints --------------------------------
    if full_model:
        from .weight_broadcast import BroadcastCfg, WeightBroadcaster
        from .disk_sync import DiskBroadcaster, DiskSyncCfg

        def _build_broadcaster(payload: dict):
            mode = payload.get("weight_sync", "disk")
            if mode == "disk":
                return DiskBroadcaster(
                    DiskSyncCfg(
                        sglang_base_url=payload["sglang_base_url"],
                        ckpt_root=payload["ckpt_root"],
                    ),
                    rank=rank, world=world,
                )
            elif mode == "nccl":
                return WeightBroadcaster(
                    BroadcastCfg(
                        sglang_base_url=payload["sglang_base_url"],
                        sglang_tp=int(payload["sglang_tp"]),
                        master_address=payload["master_address"],
                        master_port=int(payload["master_port"]),
                        group_name=payload.get("group_name", "ttt_weight_update"),
                    ),
                    rank=rank, world=world,
                )
            else:
                raise ValueError(f"unknown weight_sync mode: {mode!r}")

        @app.post("/init_broadcast")
        async def init_broadcast(req: _InitBroadcastReq):  # noqa: ANN001
            if broadcaster_holder["broadcaster"] is not None:
                return {"ok": True, "already_initialized": True}
            payload = req.model_dump()
            bc = _build_broadcaster(payload)
            broadcaster_holder["broadcaster"] = bc

            async def _do_init_collective():
                if dispatcher is not None:
                    fut = dispatcher.submit("init_broadcast", payload)
                    await fut
                else:
                    await bc.init_group_async()
            await _do_init_collective()
            return {"ok": True, "weight_sync": req.weight_sync}

        @app.post("/train_base", response_model=TrainBaseResponse)
        async def train_base(req: TrainBaseRequest):
            if not req.rollouts:
                raise HTTPException(400, "empty rollouts")
            await sem.acquire()
            state["in_flight"] += 1
            try:
                if dispatcher is not None:
                    fut = dispatcher.submit("train_base", {
                        "step": req.step,
                        "rollouts": [r.model_dump() for r in req.rollouts],
                        "group_ids": req.group_ids,
                    })
                    metrics = await fut
                else:
                    metrics = await asyncio.to_thread(
                        grpo_step,
                        peft_model=None,
                        model_call=base_model,
                        optimizer=optimizer,
                        tokenizer=tokenizer,
                        adapter_name=None,
                        adapter_params=[p for p in base_model.parameters() if p.requires_grad],
                        rollouts=[
                            RolloutT(prompt=r.prompt, completion=r.completion, reward=r.reward)
                            for r in req.rollouts
                        ],
                        cfg=grpo_cfg,
                        device=device,
                        group_ids=req.group_ids,
                    )

                sync_ms = 0.0
                bc = broadcaster_holder["broadcaster"]
                if bc is not None:
                    if dispatcher is not None:
                        bfut = dispatcher.submit("broadcast_weights", {})
                        sync_ms = float(await bfut)
                    else:
                        if isinstance(bc, DiskBroadcaster):
                            sync_ms = await bc.broadcast_model(base_model)
                        else:
                            sync_ms = await bc.broadcast_named_params(
                                base_model.named_parameters()
                            )

                return TrainBaseResponse(
                    loss=metrics["loss"],
                    pg=metrics["pg"],
                    kl=metrics["kl"],
                    grad_norm=metrics["grad_norm"],
                    reward_mean=metrics["reward_mean"],
                    reward_std=metrics["reward_std"],
                    advantage_mean=metrics["advantage_mean"],
                    weight_sync_ms=sync_ms,
                )
            finally:
                state["in_flight"] -= 1
                sem.release()

        # Register the two collective handlers (init + broadcast) so non-rank-0
        # ranks join the work. These are no-ops on world=1.
        if dispatcher is not None:
            def _init_broadcast_handler(payload: dict) -> dict:
                bc = broadcaster_holder.get("broadcaster")
                if bc is None:
                    bc = _build_broadcaster(payload)
                    broadcaster_holder["broadcaster"] = bc
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(bc.init_group_async())
                finally:
                    loop.close()
                return {"ok": True}

            def _broadcast_weights_handler(payload: dict) -> dict:
                bc = broadcaster_holder.get("broadcaster")
                if bc is None:
                    raise RuntimeError("broadcaster not initialized")
                loop = asyncio.new_event_loop()
                try:
                    if isinstance(bc, DiskBroadcaster):
                        ms = loop.run_until_complete(bc.broadcast_model(base_model))
                    else:
                        ms = loop.run_until_complete(
                            bc.broadcast_named_params(base_model.named_parameters())
                        )
                finally:
                    loop.close()
                return {"weight_sync_ms": float(ms)}

            dispatcher.handlers["init_broadcast"] = _init_broadcast_handler
            dispatcher.handlers["broadcast_weights"] = _broadcast_weights_handler

    return app, dispatcher


def _non_rank0_loop(dispatcher: Dispatcher) -> None:
    """Non-rank-0 ranks: just run the dispatcher in the main thread.
    They never call uvicorn — only rank 0 serves HTTP."""
    dispatcher.run_forever()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--max-concurrent", type=int, required=True,
                   help="Concurrent /train calls (on different adapters).")
    p.add_argument("--max-resident-adapters", type=int, default=8,
                   help="LRU cap on adapters held in GPU memory.")
    p.add_argument("--fsdp", action="store_true",
                   help="Wrap base model in FSDP2 fully_shard (per transformer block). "
                        "Only meaningful under torchrun (WORLD_SIZE>1).")
    p.add_argument("--full-model", action="store_true",
                   help="Train the base model directly (no PEFT/LoRA). Implies "
                        "--fsdp under torchrun. Exposes /train_base + /init_broadcast.")
    p.add_argument("--run-root", default=None,
                   help="If set, write a RegistryEntry to <run-root>/registry/trainer/<idx>.json on startup.")
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--advertise-host", default=None)
    p.add_argument("--log-level", default="info")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app, dispatcher = build_app(
        args.config, args.max_concurrent, args.max_resident_adapters,
        use_fsdp=args.fsdp,
        full_model=args.full_model,
    )
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        # Non-rank-0 ranks block on the collective dispatcher loop. They never
        # touch FastAPI; they just answer broadcasts initiated by rank 0.
        assert dispatcher is not None, "non-rank-0 reached without dispatcher (world>1 required)"
        _non_rank0_loop(dispatcher)
        return

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


if __name__ == "__main__":
    main()
