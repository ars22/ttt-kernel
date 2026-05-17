"""FSDP2 wrap for a PEFT-LoRA base model.

`fully_shard()` (the FSDP2 per-module API) plays nicely with PEFT's adapter
routing wrappers. The pattern:

    1. Build base AutoModelForCausalLM on meta or CPU (avoid duplicating
       full base model across ranks before sharding).
    2. Wrap in PEFT (LoRA adds A/B side-modules onto the linear projections).
    3. Apply fully_shard() to each transformer block. This is the
       checkpoint-granularity unit that FSDP2 gathers per forward step.
    4. Apply fully_shard() to the root model.

Concurrent updates on *different* adapters are allowed: PEFT routes per
active adapter, and only the active adapter's A/B + AdamW participate.
"""
from __future__ import annotations

import logging
import os
from typing import List

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM

from .model import ModelInitCfg, build_peft_lora_cfg, dtype_from_str

log = logging.getLogger("ttt_kernel.trainer.fsdp_model")


def _find_transformer_blocks(peft_model: PeftModel) -> List[torch.nn.Module]:
    """Walk down to the transformer block list.

    Standard HF causal-LM layout:
        peft_model.base_model.model.model.layers[i]
    `base_model` is PEFT's wrapper; the inner `.model` is the HF root;
    `.model` again is the LM body; `.layers` is the ModuleList of blocks.
    """
    inner = peft_model.base_model.model
    body = getattr(inner, "model", inner)  # decoder-only causal LMs nest one deeper
    layers = getattr(body, "layers", None)
    if layers is None:
        # Fall back: scan for the first ModuleList that's long enough to be the blocks.
        for _name, mod in inner.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) > 4:
                layers = mod
                break
    if layers is None:
        raise RuntimeError(
            "could not locate transformer block ModuleList; FSDP2 wrap needs "
            "to be applied per-block but the model architecture wasn't recognized"
        )
    return list(layers)


def build_fsdp_peft_model(
    cfg: ModelInitCfg,
    rank: int,
    world: int,
    local_rank: int,
) -> PeftModel:
    """Build the base model, wrap in PEFT, and fully_shard per transformer block.

    All ranks must call this in lockstep — FSDP2 init is a collective.
    """
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    log.info("rank %d/%d building base model %s", rank, world, cfg.name)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        torch_dtype=dtype_from_str(cfg.dtype),
        trust_remote_code=cfg.trust_remote_code,
    )
    peft_model: PeftModel = get_peft_model(base, build_peft_lora_cfg(cfg))
    peft_model.to(device)
    peft_model.train()

    blocks = _find_transformer_blocks(peft_model)
    log.info("rank %d wrapping %d transformer blocks with fully_shard", rank, len(blocks))
    mp = MixedPrecisionPolicy(
        param_dtype=dtype_from_str(cfg.dtype),
        reduce_dtype=torch.float32,
    )
    for blk in blocks:
        fully_shard(blk, mp_policy=mp)
    fully_shard(peft_model, mp_policy=mp)

    if dist.is_initialized():
        dist.barrier()
    return peft_model
