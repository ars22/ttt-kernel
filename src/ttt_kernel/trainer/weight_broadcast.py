"""Trainer → SGLang weight broadcast (multitask / full-model path).

Mirrors SGLang's own `init_custom_process_group` helper on the trainer side
so the trainer's torch.distributed group lines up exactly with SGLang's. The
combined ProcessGroup contains the trainer's `world` ranks at positions
[0..world) and SGLang's `tp` ranks at [world..world+tp).

Flow:

  bootstrap (once):
    rank 0  →  POST /init_weights_update_group  (rank_offset=world, world=combined)
    all trainer ranks → init_custom_process_group(rank=this_rank, world=combined)

  per step:
    for each (name, param) in model.named_parameters():
        all FSDP ranks call DTensor.full_tensor() → fully-replicated tensor
        rank 0 → POST /update_weights_from_distributed {[name], [dtype], [shape]}
        all trainer ranks call dist.broadcast(full, src=0, group=update_pg)

We send one tensor per HTTP call: simpler, and pipeline overhead is small
relative to the actual NCCL transfer at ~50 GB/s on H100s.

The named-group helper here is copied from `sglang/srt/model_executor/model_runner.py`
so both sides go through the SAME code path (TCPStore via PrefixStore).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Optional

import httpx
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
)
from torch.distributed.rendezvous import rendezvous
from torch.distributed.tensor import DTensor

log = logging.getLogger("ttt_kernel.trainer.weight_broadcast")


@dataclass
class BroadcastCfg:
    sglang_base_url: str       # e.g. http://node0:30100
    sglang_tp: int             # SGLang's --tp-size
    master_address: str        # trainer node hostname/IP visible to SGLang
    master_port: int           # free port on trainer node, e.g. 29600
    group_name: str = "ttt_weight_update"


_TORCH_TO_SGLANG_DTYPE = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def _dtype_to_str(dt: torch.dtype) -> str:
    s = _TORCH_TO_SGLANG_DTYPE.get(dt)
    if s is None:
        raise ValueError(f"unsupported dtype for SGLang weight broadcast: {dt}")
    return s


def _init_custom_process_group(
    *,
    init_method: str,
    world_size: int,
    rank: int,
    group_name: str,
    backend: str = "nccl",
    timeout: Optional[timedelta] = None,
    device_id: Optional[torch.device] = None,
) -> dist.ProcessGroup:
    """Trainer-side mirror of sglang's `init_custom_process_group`.

    Creates a secondary NCCL group alongside the trainer's existing FSDP
    process group. The store is a PrefixStore so the keys don't collide
    with anything else SGLang/torch uses on the same master address.

    `device_id` is REQUIRED for NCCL to know which CUDA device to bind the
    secondary communicator to — without it the rendezvous can hang
    indefinitely. SGLang passes it on its side; we must do the same.
    """
    if timeout is None:
        timeout = default_pg_timeout
    rendezvous_iter = rendezvous(init_method, rank, world_size, timeout=timeout)
    store, _rank, _world_size = next(rendezvous_iter)
    store.set_timeout(timeout)
    store = PrefixStore(group_name, store)

    backend_obj = Backend(backend)
    # torch 2.6+ renamed pg_options → backend_options. We pass whichever is
    # accepted by this torch version.
    import inspect
    helper_sig = inspect.signature(_new_process_group_helper)
    kwargs = {}
    if "backend_options" in helper_sig.parameters:
        kwargs["backend_options"] = None
    elif "pg_options" in helper_sig.parameters:
        kwargs["pg_options"] = None
    if "device_id" in helper_sig.parameters and device_id is not None:
        kwargs["device_id"] = device_id

    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],                       # global_ranks_in_group; [] = new global mapping
        backend_obj,
        store,
        group_name=group_name,
        timeout=timeout,
        **kwargs,
    )
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg


class WeightBroadcaster:
    """Owns the trainer↔SGLang update group and pushes weights after each step."""

    def __init__(
        self,
        cfg: BroadcastCfg,
        *,
        rank: int,
        world: int,
    ) -> None:
        self.cfg = cfg
        self.rank = rank
        self.world = world
        self.combined_world = world + cfg.sglang_tp
        self.update_pg: Optional[dist.ProcessGroup] = None
        self._client = httpx.AsyncClient(timeout=600.0)
        self._initialized = False

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- bootstrap -------------------------------------------------------

    async def init_group_async(self) -> None:
        if self._initialized:
            return

        init_method = f"tcp://{self.cfg.master_address}:{self.cfg.master_port}"

        async def _http_init() -> None:
            # Only rank 0 (the HTTP host) POSTs. Other ranks can't anyway —
            # they have no httpx client and they live in the dispatcher thread.
            if self.rank != 0:
                return
            log.info("POST /init_weights_update_group rank_offset=%d world=%d",
                     self.world, self.combined_world)
            r = await self._client.post(
                f"{self.cfg.sglang_base_url.rstrip('/')}/init_weights_update_group",
                json={
                    "master_address": self.cfg.master_address,
                    "master_port": self.cfg.master_port,
                    "rank_offset": self.world,
                    "world_size": self.combined_world,
                    "group_name": self.cfg.group_name,
                    "backend": "nccl",
                },
            )
            r.raise_for_status()

        def _torch_init() -> None:
            import os
            import torch
            # Pin THIS thread to the rank's CUDA device. asyncio.to_thread'd
            # workers don't inherit cuda.set_device from the spawning thread,
            # and NCCL secondary-group init needs the right device active to
            # avoid hanging the rendezvous.
            local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                dev = torch.device(f"cuda:{local_rank}")
            else:
                dev = None
            log.info(
                "creating secondary NCCL group rank=%d world=%d device=%s",
                self.rank, self.combined_world, dev,
            )
            self.update_pg = _init_custom_process_group(
                init_method=init_method,
                world_size=self.combined_world,
                rank=self.rank,
                group_name=self.cfg.group_name,
                device_id=dev,
            )
            log.info("rank %d: secondary NCCL group ready", self.rank)

        await asyncio.gather(
            _http_init(),
            asyncio.to_thread(_torch_init),
        )
        self._initialized = True
        log.info("update group ready (trainer rank %d/%d in combined world=%d)",
                 self.rank, self.world, self.combined_world)

    # ---- per-step push ---------------------------------------------------

    @staticmethod
    def _materialize_full(t: torch.Tensor) -> torch.Tensor:
        if isinstance(t, DTensor):
            return t.full_tensor()
        return t

    async def broadcast_named_params(
        self,
        named_params: Iterable[tuple[str, torch.Tensor]],
    ) -> float:
        """Push every (name, tensor) over the update group. Returns wall ms."""
        if not self._initialized:
            raise RuntimeError("WeightBroadcaster.init_group_async must be called first")
        assert self.update_pg is not None
        t0 = time.monotonic()
        for name, t in named_params:
            full = self._materialize_full(t)
            if not full.is_contiguous():
                full = full.contiguous()
            if self.rank == 0:
                http_task = asyncio.create_task(self._client.post(
                    f"{self.cfg.sglang_base_url.rstrip('/')}/update_weights_from_distributed",
                    json={
                        "names": [name],
                        "dtypes": [_dtype_to_str(full.dtype)],
                        "shapes": [list(full.shape)],
                        "group_name": self.cfg.group_name,
                        "flush_cache": False,
                    },
                ))
                await asyncio.to_thread(
                    dist.broadcast, full, src=0, group=self.update_pg,
                )
                resp = await http_task
                resp.raise_for_status()
            else:
                await asyncio.to_thread(
                    dist.broadcast, full, src=0, group=self.update_pg,
                )
            del full
        if self.rank == 0:
            try:
                r = await self._client.post(
                    f"{self.cfg.sglang_base_url.rstrip('/')}/flush_cache", json={},
                )
                r.raise_for_status()
            except httpx.HTTPError as e:
                log.warning("flush_cache POST failed (continuing): %s", e)
        return (time.monotonic() - t0) * 1000.0
