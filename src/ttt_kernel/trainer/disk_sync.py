"""Disk-based trainer→SGLang weight sync (fallback when NCCL broadcast is unusable).

Per step the trainer:
  1. Gathers the full unsharded state dict on rank 0 via FSDP2's collective
     `get_model_state_dict(full_state_dict=True, cpu_offload=True)`.
  2. Writes it as a HuggingFace-compatible directory (config.json + a single
     model.safetensors) to `<ckpt_root>/step_<N>/`.
  3. POSTs `/update_weights_from_disk` to SGLang with that path.
  4. Deletes the previous step's directory to keep disk usage bounded.

Cost: ~6s write + ~6s SGLang read + ~5s reload = ~17s per step on Weka.

The state-dict gather is itself a COLLECTIVE that all ranks must participate
in — hence this lives in the dispatcher's broadcast-handler path on the
trainer, exactly like the NCCL broadcaster.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import Optional

import httpx
import torch

log = logging.getLogger("ttt_kernel.trainer.disk_sync")


@dataclass
class DiskSyncCfg:
    sglang_base_url: str        # e.g. http://node0:30100
    ckpt_root: str              # writable dir; trainer creates step_<N>/ subdirs
    keep_last: int = 1          # delete older checkpoints; SGLang only needs the latest


def _save_hf_dir(model: torch.nn.Module, output_dir: str) -> None:
    """Save model.config + state_dict to a HF-compatible directory.

    Called ONLY on rank 0, after the state_dict has been gathered via the
    FSDP2 full-state-dict collective. The state dict is expected to already
    be on CPU (cpu_offload=True in the gather options).
    """
    from safetensors.torch import save_file
    os.makedirs(output_dir, exist_ok=True)
    # SGLang loads with from_pretrained; config + tokenizer + weights expected.
    # Tokenizer isn't strictly needed for /update_weights_from_disk (SGLang
    # keeps its tokenizer), but harmless to skip — only save what we must.
    if hasattr(model, "config"):
        model.config.save_pretrained(output_dir)
    # Strip module prefixes ("_fsdp_wrapped_module." etc) that FSDP2 doesn't
    # add but other wrappers might; safetensors needs plain param names.
    sd = {k.replace("module.", ""): v for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(output_dir, "model.safetensors"))


def _gather_full_state_dict(model: torch.nn.Module) -> dict:
    """All ranks call; only rank 0 receives the full unsharded dict.

    NOTE: We deliberately do NOT pass cpu_offload=True. On this cluster the
    per-tensor GPU→host streaming inside get_model_state_dict was hanging
    indefinitely (rank 0 spinning at 100% util on a single NCCL collective).
    Without cpu_offload, the all_gather completes as one bulk collective and
    rank 0 holds the full state on GPU. We then do a single rank-0 move-to-CPU
    immediately afterward so GPU memory is freed before downstream work.

    Memory check: 32B bf16 = 64 GB on rank 0 GPU on top of ~8 GB FSDP shard
    ≈ 72 GB. Fits in 80 GB H100. For 4B, it's ~10 GB total — trivial.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict,
    )
    opts = StateDictOptions(full_state_dict=True, cpu_offload=False)
    state_dict = get_model_state_dict(model, options=opts)
    # Bulk move-to-CPU on rank 0 only. Other ranks got an empty sentinel.
    if torch.distributed.get_rank() == 0:
        cpu_state = {k: v.detach().to("cpu", non_blocking=False) for k, v in state_dict.items()}
        # Free the GPU copies eagerly.
        del state_dict
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return cpu_state
    return state_dict


class DiskBroadcaster:
    """Owns the trainer↔SGLang disk-sync state."""

    def __init__(self, cfg: DiskSyncCfg, *, rank: int, world: int) -> None:
        self.cfg = cfg
        self.rank = rank
        self.world = world
        self._client = httpx.AsyncClient(timeout=1800.0)
        self._step_counter = 0
        self._last_path: Optional[str] = None
        self._initialized = False

    async def aclose(self) -> None:
        await self._client.aclose()

    async def init_group_async(self) -> None:
        """No-op for disk sync — there's no rendezvous needed. Just mark ready."""
        if self.rank == 0:
            os.makedirs(self.cfg.ckpt_root, exist_ok=True)
        self._initialized = True
        log.info("disk-sync ready (rank %d/%d, ckpt_root=%s, sglang=%s)",
                 self.rank, self.world, self.cfg.ckpt_root, self.cfg.sglang_base_url)

    async def broadcast_model(self, model: torch.nn.Module) -> float:
        """Save full model to disk, POST to SGLang, drop old checkpoint. Returns wall ms."""
        if not self._initialized:
            raise RuntimeError("DiskBroadcaster.init_group_async must be called first")
        t0 = time.monotonic()
        step = self._step_counter
        self._step_counter += 1
        step_dir = os.path.join(self.cfg.ckpt_root, f"step_{step:05d}")

        # All ranks participate in the gather collective.
        log.info("rank %d: gathering full state dict for step %d", self.rank, step)
        state_dict = await asyncio.to_thread(_gather_full_state_dict, model)

        if self.rank == 0:
            log.info("rank 0: writing %d tensors to %s", len(state_dict), step_dir)
            # We do the save in a thread so the asyncio loop stays alive (the
            # httpx clients owned by FastAPI may have pending tasks).
            await asyncio.to_thread(_save_to_disk_rank0, model, state_dict, step_dir)

        if torch.distributed.is_initialized():
            await asyncio.to_thread(torch.distributed.barrier)

        sglang_ms = 0.0
        if self.rank == 0:
            t_post = time.monotonic()
            log.info("rank 0: POST /update_weights_from_disk path=%s", step_dir)
            r = await self._client.post(
                f"{self.cfg.sglang_base_url.rstrip('/')}/update_weights_from_disk",
                json={"model_path": step_dir},
            )
            r.raise_for_status()
            sglang_ms = (time.monotonic() - t_post) * 1000.0
            log.info("rank 0: SGLang reload OK in %.1fs", sglang_ms / 1000.0)

            # Best-effort cleanup of older checkpoint(s).
            if self._last_path is not None and self.cfg.keep_last == 1:
                old = self._last_path
                self._last_path = step_dir
                try:
                    shutil.rmtree(old, ignore_errors=True)
                except Exception as e:  # noqa: BLE001
                    log.warning("failed to remove old checkpoint %s: %s", old, e)
            else:
                self._last_path = step_dir

        # Free CPU memory from the state dict on every rank.
        del state_dict

        return (time.monotonic() - t0) * 1000.0


def _save_to_disk_rank0(
    model: torch.nn.Module,
    state_dict: dict,
    output_dir: str,
) -> None:
    """rank 0 only: write the gathered state dict + config to output_dir."""
    from safetensors.torch import save_file
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(model, "config"):
        model.config.save_pretrained(output_dir)
    # Strip any wrapper prefixes (defensive; FSDP2 typically doesn't add any).
    clean = {k.replace("module.", ""): v for k, v in state_dict.items()}
    save_file(clean, os.path.join(output_dir, "model.safetensors"))
