"""Base-model factory + PEFT wrapper.

Single-GPU path here (task #5). The FSDP2 wrap lands in `fsdp_model.py` in
task #6 and the server picks the factory based on world size.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


@dataclass(frozen=True)
class ModelInitCfg:
    name: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    # LoRA
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    bias: str = "none"


def dtype_from_str(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


def init_distributed() -> tuple[int, int, int]:
    """If launched under torchrun, init NCCL and return (rank, world, local_rank).
    Otherwise (rank=0, world=1, local_rank=0). FSDP2 wrap is decided by caller
    based on world > 1."""
    import torch.distributed as dist
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def build_tokenizer(cfg: ModelInitCfg) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def build_peft_lora_cfg(cfg: ModelInitCfg) -> LoraConfig:
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias=cfg.bias,
        task_type="CAUSAL_LM",
    )


def build_peft_model(cfg: ModelInitCfg, device: torch.device) -> PeftModel:
    """Single-GPU build. The base model is moved to `device` and wrapped in PEFT.
    A `default` adapter exists immediately (PEFT requires one); training uses
    per-problem adapters added via `add_adapter`."""
    base = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        torch_dtype=dtype_from_str(cfg.dtype),
        trust_remote_code=cfg.trust_remote_code,
    )
    peft_model: PeftModel = get_peft_model(base, build_peft_lora_cfg(cfg))
    peft_model.to(device)
    peft_model.train()
    return peft_model
