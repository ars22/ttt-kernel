"""Wire types for the three-pool architecture.

These Pydantic models are the contract between the orchestrator and each
pool service. Anything that travels over HTTP must round-trip through one
of these.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---- sampler --------------------------------------------------------------


class SampleRequest(BaseModel):
    """Orchestrator → sampler. Adapter is referenced by Weka path; the sampler
    is responsible for load_lora_adapter (with LRU eviction) before sampling."""
    problem_id: int
    turn: int
    prompt: str
    adapter_path: str        # absolute Weka path under <adapters_root>/p.../v...
    adapter_name: str        # stable handle, == AdapterRef.name
    n: int = Field(ge=1, description="K rollouts to draw for this prompt")
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 16384


class Generation(BaseModel):
    text: str
    finish_reason: str
    completion_tokens: int


class SampleResponse(BaseModel):
    completions: List[Generation]


# ---- env ------------------------------------------------------------------


ErrorKind = Literal["ok", "parse", "compile", "incorrect", "harness", "timeout"]


class EvaluateRequest(BaseModel):
    """Orchestrator → env. Per-rollout: one kernel source, one slot."""
    problem_id: int
    turn: int
    sample: int              # 0..K-1, just for logging
    completion: str          # raw model output; env extracts the ```python``` block
    # KernelBench config is read from the env service's startup, not the wire.


class EvaluateResponse(BaseModel):
    raw_completion: str
    kernel_src: Optional[str]
    compiled: bool
    correct: bool
    speedup: float
    runtime_ms: float
    ref_runtime_ms: float
    reward: float
    feedback: str
    error_kind: ErrorKind


# ---- trainer --------------------------------------------------------------


class Rollout(BaseModel):
    """One (prompt, completion, reward) triple for the GRPO step."""
    prompt: str
    completion: str
    reward: float


class TrainRequest(BaseModel):
    """Orchestrator → trainer.

    The trainer loads `adapter_in_path` (cold from disk, or hot-cached),
    runs one GRPO step over `rollouts`, and writes the updated weights to
    `adapter_out_path`. `group_ids` is for per-group advantage normalization
    when training jointly on multiple problems (default: one group).
    """
    problem_id: int
    turn: int
    adapter_in_path: str
    adapter_in_name: str
    adapter_out_path: str
    adapter_out_name: str
    rollouts: List[Rollout]
    group_ids: Optional[List[int]] = None


class TrainResponse(BaseModel):
    loss: float
    pg: float
    kl: float
    grad_norm: float
    reward_mean: float
    reward_std: float
    advantage_mean: float


# ---- capacity & registry --------------------------------------------------


PoolKind = Literal["sampler", "env", "trainer"]


class Capacity(BaseModel):
    """Self-reported capacity, used by the orchestrator's scheduler."""
    pool: PoolKind
    max_concurrent: int      # decode requests / envs / adapter updates
    in_flight: int = 0


class RegistryEntry(BaseModel):
    """Written by each service to runs/<run>/registry/{pool}/{idx}.json on
    startup; read by the orchestrator. `state` flips to "down" on graceful
    shutdown so the orchestrator drops it from rotation."""
    pool: PoolKind
    idx: int                 # SLURM array index, or 0 for single-node
    host: str
    port: int
    capacity: int
    state: Literal["up", "down"] = "up"
