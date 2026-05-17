"""FSDP2-aware adapter save.

PEFT's `save_pretrained` calls `state_dict()` which under FSDP2 returns
DTensors (one shard per rank, with metadata for reassembly). Writing those
to safetensors directly would fail; we need to gather full tensors on rank
0 and write `adapter_model.safetensors` + `adapter_config.json` by hand.
"""
from __future__ import annotations

import json
import logging
import os
import shutil

import torch
import torch.distributed as dist
from peft import PeftModel
from safetensors.torch import save_file as save_safetensors

log = logging.getLogger("ttt_kernel.trainer.fsdp_save")


def _to_full_tensor(t: torch.Tensor) -> torch.Tensor:
    """If `t` is a DTensor, gather to a full tensor on rank 0. Otherwise return as-is."""
    try:
        from torch.distributed.tensor import DTensor  # FSDP2 distributed tensor type
    except ImportError:
        DTensor = None  # noqa: N806
    if DTensor is not None and isinstance(t, DTensor):
        # full_tensor() does an all-gather and returns a plain tensor on every rank.
        # We only write on rank 0 anyway; the extra all-gather is the price of FSDP2.
        return t.full_tensor()
    return t


def save_adapter_fsdp(
    peft_model: PeftModel,
    adapter_name: str,
    out_dir: str,
    peft_config,
    rank: int,
) -> str:
    """Gather and write `out_dir/adapter_{config.json,model.safetensors}`.

    Atomic via tmp+rename so a partial write is never visible.
    """
    # Walk all params; keep just the adapter's lora_A / lora_B.
    tag = f".{adapter_name}."
    full_sd: dict[str, torch.Tensor] = {}
    for name, p in peft_model.named_parameters():
        if tag not in name:
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue
        # All ranks participate in the gather; only rank 0 keeps the result.
        gathered = _to_full_tensor(p.detach()).to("cpu")
        if rank == 0:
            # Strip the leading `base_model.model.` from the key to match PEFT's
            # on-disk format (PEFT saves keys relative to the peft module root).
            key = name
            if key.startswith("base_model.model."):
                key = "base_model.model." + key[len("base_model.model."):]
            full_sd[key] = gathered.contiguous().to(p.dtype)

    if rank != 0:
        return out_dir

    tmp = out_dir + ".tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    save_safetensors(full_sd, os.path.join(tmp, "adapter_model.safetensors"))

    # Write adapter_config.json from the in-memory PEFT config so SGLang can
    # load this adapter the same way it loads the seed.
    cfg_dict = peft_config.to_dict()
    # PEFT stores some non-JSON-serializable fields (e.g. set for target_modules);
    # normalise.
    for k, v in list(cfg_dict.items()):
        if isinstance(v, set):
            cfg_dict[k] = sorted(v)
    with open(os.path.join(tmp, "adapter_config.json"), "w") as fp:
        json.dump(cfg_dict, fp, indent=2)

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.rename(tmp, out_dir)
    log.info("saved FSDP-gathered adapter '%s' to %s (%d tensors)",
             adapter_name, out_dir, len(full_sd))
    return out_dir
