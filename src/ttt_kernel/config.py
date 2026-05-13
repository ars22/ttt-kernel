"""Typed config loader. YAML on disk → pydantic models in memory.

Override any field on the CLI with dotted syntax:
    python scripts/inference.py --config configs/default.yaml \
        rollout.num_samples=16 grpo.learning_rate=5e-6
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field


class ModelCfg(BaseModel):
    name: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = True


class LoraCfg(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"


class SGLangCfg(BaseModel):
    base_url: str = "http://127.0.0.1:30000"
    adapter_name: str = "ttt"
    adapter_out_dir: str = "./adapters/ttt"
    update_weights_endpoint: str = "/update_weights_from_disk"


class RolloutCfg(BaseModel):
    num_samples: int = 8
    max_tokens: int = 16384
    temperature: float = 1.0
    top_p: float = 0.95


class KernelBenchCfg(BaseModel):
    repo_path: str
    dataset_src: str = "huggingface"
    dataset_name: str = "ScalingIntelligence/KernelBench"
    level: int = 1
    problem_ids: Optional[List[int]] = None
    backend: str = "cuda"
    precision: str = "fp32"
    gpu_arch: str = "Blackwell"
    prompt_option: str = "one_shot"
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    timing_method: str = "cuda_event"
    eval_timeout_s: int = 120


class RewardCfg(BaseModel):
    speedup_log_scale: bool = True
    error_penalty: float = -1.0
    incorrect_penalty: float = -1.0
    clip: float = 2.0


class GRPOCfg(BaseModel):
    beta_kl: float = 0.04
    epsilon_clip: float = 0.2
    group_advantage_norm: bool = True
    update_epochs: int = 1
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    micro_batch_size: int = 1
    max_seq_len: int = 8192


class LoopCfg(BaseModel):
    num_turns: int = 5
    persist_adapter_across_problems: bool = False
    seed: int = 0


class WandbCfg(BaseModel):
    enabled: bool = False
    project: str = "ttt-kernel"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    mode: str = "online"  # online | offline | disabled
    tags: List[str] = Field(default_factory=list)


class LoggingCfg(BaseModel):
    out_dir: str = "./runs"
    run_name: Optional[str] = None
    log_every: int = 1
    save_adapter_every_turn: bool = True
    wandb: WandbCfg = WandbCfg()


class Config(BaseModel):
    model: ModelCfg
    lora: LoraCfg = LoraCfg()
    sglang: SGLangCfg = SGLangCfg()
    rollout: RolloutCfg = RolloutCfg()
    kernelbench: KernelBenchCfg
    reward: RewardCfg = RewardCfg()
    grpo: GRPOCfg = GRPOCfg()
    loop: LoopCfg = LoopCfg()
    logging: LoggingCfg = LoggingCfg()


def _set_nested(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    # naive type coercion: try int → float → bool → keep str
    if isinstance(value, str):
        v = value
        for caster in (int, float):
            try:
                value = caster(v); break
            except ValueError:
                pass
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            value = value.lower() == "true"
        if isinstance(value, str) and value.lower() == "null":
            value = None
    cur[keys[-1]] = value


def load_config(path: str | Path, overrides: List[str] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    # env-var overlay for paths that are commonly per-host
    if (env_kb := os.environ.get("KERNELBENCH_PATH")):
        raw.setdefault("kernelbench", {})["repo_path"] = env_kb
    if (env_sg := os.environ.get("SGLANG_BASE_URL")):
        raw.setdefault("sglang", {})["base_url"] = env_sg
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Bad override (need key=value): {ov}")
        k, v = ov.split("=", 1)
        _set_nested(raw, k.strip(), v.strip())
    return Config(**raw)
