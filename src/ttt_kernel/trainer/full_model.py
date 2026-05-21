"""Full-model FSDP2 factory (no PEFT).

Used by the multitask trainer mode: trains the base model in place via
REINFORCE on a flat batch of (prompt, completion, reward) rollouts. The
model is wrapped per transformer block with `fully_shard()` exactly like
the PEFT path, but `model_call` is the bare causal-LM (no adapter routing).

Activation checkpointing is enabled per block — full-model training of
Qwen3-32B at 16k context on 8×H100 needs it.
"""
from __future__ import annotations

import logging
from typing import List

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

from .model import ModelInitCfg, dtype_from_str

log = logging.getLogger("ttt_kernel.trainer.full_model")


def _find_blocks(model: torch.nn.Module) -> List[torch.nn.Module]:
    body = getattr(model, "model", model)
    layers = getattr(body, "layers", None)
    if layers is None:
        for _name, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) > 4:
                layers = mod
                break
    if layers is None:
        raise RuntimeError("could not locate transformer block ModuleList")
    return list(layers)


def _enable_grad_checkpointing(blocks: List[torch.nn.Module]) -> None:
    """Wrap each block's forward so its activations are recomputed on backward.

    Saves ~80% of activation memory at ~30% step-time cost. Mandatory for
    Qwen3-32B full-model training at 16k seqlen on H100s.
    """
    for blk in blocks:
        orig_forward = blk.forward

        def make_ckpt(_orig):
            def ckpt_forward(*args, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(_orig, *args, use_reentrant=False, **kwargs)
                return _orig(*args, **kwargs)
            return ckpt_forward

        blk.forward = make_ckpt(orig_forward)


def build_fsdp_full_model(
    cfg: ModelInitCfg,
    rank: int,
    world: int,
    local_rank: int,
    *,
    grad_checkpoint: bool = True,
) -> torch.nn.Module:
    """Build the base causal-LM, enable grad checkpointing, fully_shard per block."""
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    log.info("rank %d/%d building full model %s (no PEFT)", rank, world, cfg.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        torch_dtype=dtype_from_str(cfg.dtype),
        trust_remote_code=cfg.trust_remote_code,
    )
    model.to(device)
    model.train()
    # HF's model.config.use_cache=True is incompatible with grad checkpointing.
    if hasattr(model, "config"):
        model.config.use_cache = False

    blocks = _find_blocks(model)
    log.info("rank %d wrapping %d transformer blocks with fully_shard", rank, len(blocks))
    if grad_checkpoint:
        _enable_grad_checkpointing(blocks)
    mp = MixedPrecisionPolicy(
        param_dtype=dtype_from_str(cfg.dtype),
        reduce_dtype=torch.float32,
    )
    for blk in blocks:
        fully_shard(blk, mp_policy=mp)
    fully_shard(model, mp_policy=mp)

    if dist.is_initialized():
        dist.barrier()
    return model
