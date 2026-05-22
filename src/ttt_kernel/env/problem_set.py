"""KernelBench dataset wrapper.

The env service enumerates problems at startup so callers reference them
only by `problem_id` over the wire. The orchestrator can also call this
to materialize prompts to feed the sampler — that's the only reason
`get_prompt` is exposed.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional


def _add_kernelbench_to_path(repo_path: str) -> None:
    src = os.path.join(repo_path, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _apply_gpu_arch(gpu_arch: str) -> None:
    """Replicate kernel_env.py's gpu-arch handling. Named archs go through
    KernelBench's `set_gpu_arch`; numeric archs bypass it and set
    `TORCH_CUDA_ARCH_LIST` directly so nvcc only builds one variant."""
    from kernelbench.utils import set_gpu_arch, NVIDIA_ARCHS  # noqa: WPS433
    if gpu_arch in NVIDIA_ARCHS:
        set_gpu_arch([gpu_arch])
        return
    tcl = gpu_arch.replace("sm_", "").replace("_", ".")
    if tcl and tcl[0].isdigit() and "." not in tcl:
        tcl = tcl[:-1] + "." + tcl[-1] if not tcl[-1].isalpha() else tcl[:-2] + "." + tcl[-2:]
    os.environ["TORCH_CUDA_ARCH_LIST"] = tcl


@dataclass
class Problem:
    level: int
    problem_id: int
    name: str
    ref_src: str
    prompt: str


@dataclass(frozen=True)
class ProblemSetCfg:
    repo_path: str
    dataset_src: str = "huggingface"
    dataset_name: str = "ScalingIntelligence/KernelBench"
    level: int = 1
    problem_ids: Optional[List[int]] = None
    backend: str = "cuda"
    precision: str = "fp32"
    gpu_arch: str = "Blackwell"
    prompt_option: str = "one_shot"


class ProblemSet:
    def __init__(self, cfg: ProblemSetCfg):
        self.cfg = cfg
        _add_kernelbench_to_path(cfg.repo_path)
        _apply_gpu_arch(cfg.gpu_arch)
        from kernelbench import dataset as kb_dataset
        from kernelbench import prompt_constructor_toml as kb_prompts
        self._get_prompt = kb_prompts.get_prompt_for_backend
        self._dataset = kb_dataset.construct_kernelbench_dataset(
            level=cfg.level,
            source=cfg.dataset_src,
            dataset_name=cfg.dataset_name,
        )

    def list_problem_ids(self) -> List[int]:
        if self.cfg.problem_ids:
            return list(self.cfg.problem_ids)
        return [p.problem_id for p in self._dataset]

    def get_problem(self, problem_id: int) -> Problem:
        prob = self._dataset.get_problem_by_id(problem_id)
        prompt = self._get_prompt(
            prob.code,
            self.cfg.backend,
            option=self.cfg.prompt_option,
            precision=self.cfg.precision,
            include_hardware=False,
            gpu_name=None,
        )
        return Problem(
            level=self.cfg.level,
            problem_id=problem_id,
            name=prob.name,
            ref_src=prob.code,
            prompt=prompt,
        )
