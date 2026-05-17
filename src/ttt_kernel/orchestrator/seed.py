"""Bootstrap the v000 seed adapter for each problem on disk.

The sampler can only load adapters that have `adapter_config.json` +
`adapter_model.safetensors` on disk. The trainer also expects to find the
v_in directory before /train (it cold-loads from there). For v000 we need
to materialize a fresh zero-effect LoRA: kaiming-init A, zero-init B.

Done once per run, on disk; the per-problem state machine then refers to
the same path via AdapterRef.path.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from ..shared.adapter_paths import seed

log = logging.getLogger("ttt_kernel.orchestrator.seed")


def materialize_seeds(
    adapters_root: str | Path,
    problem_ids: Iterable[int],
    *,
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    target_modules: list[str],
    bias: str = "none",
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
) -> None:
    """Write v000 for each problem id under `adapters_root`. Skips problems
    whose seed directory already exists."""
    adapters_root = Path(adapters_root)
    adapters_root.mkdir(parents=True, exist_ok=True)
    needed = [
        pid for pid in problem_ids
        if not (seed(adapters_root, pid).path / "adapter_config.json").exists()
    ]
    if not needed:
        log.info("all seed adapters already present under %s", adapters_root)
        return

    log.info("materializing %d seed adapters under %s", len(needed), adapters_root)
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    # Load the base model ONCE (these are big) and save a fresh adapter per problem.
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code,
    )
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
        target_modules=target_modules, bias=bias, task_type="CAUSAL_LM",
    )
    peft = get_peft_model(base, lora_cfg)
    for pid in needed:
        out_dir = seed(adapters_root, pid).path
        os.makedirs(out_dir, exist_ok=True)
        # Zero-effect: PEFT's default A is kaiming-init, B is zero-init. Save as-is.
        peft.save_pretrained(str(out_dir))
        # Strip tokenizer files that sneak in — SGLang's LoRA manager rejects
        # adapter dirs containing added_tokens.json / tokenizer*.json.
        for junk in (
            "added_tokens.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.json", "merges.txt",
            "chat_template.jinja",
        ):
            p = Path(out_dir) / junk
            if p.exists():
                p.unlink()
        log.info("seeded %s", out_dir)
    del peft, base
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
