"""Adapter path scheme — the ONLY cross-pool state.

Each problem owns a sequence of adapter versions on the shared Weka
filesystem; the orchestrator drives version bumps and every pool service
reads/writes via these paths.

Layout under a run's adapters root:

    <adapters_root>/p{pid:03d}/v{turn:03d}/
        adapter_config.json
        adapter_model.safetensors

`v000` is the seed adapter (zero-effect LoRA). Turn `t`'s training output is
written to `v{t+1}` so turn `t+1`'s sample call loads from there.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_PROBLEM_RE = re.compile(r"^p(\d{3})$")
_VERSION_RE = re.compile(r"^v(\d{3})$")


@dataclass(frozen=True)
class AdapterRef:
    """Identifies one adapter version on disk."""
    root: Path        # the run-wide adapters_root
    problem_id: int
    version: int

    @property
    def path(self) -> Path:
        return self.root / f"p{self.problem_id:03d}" / f"v{self.version:03d}"

    @property
    def name(self) -> str:
        # Stable name for SGLang's load_lora_adapter — must be unique per
        # (problem, version) so successive turns don't collide in the LoRA
        # manager's slot table.
        return f"p{self.problem_id:03d}_v{self.version:03d}"

    def next(self) -> "AdapterRef":
        return AdapterRef(self.root, self.problem_id, self.version + 1)


def seed(root: str | Path, problem_id: int) -> AdapterRef:
    return AdapterRef(Path(root), problem_id, 0)


def parse(path: str | Path, root: str | Path) -> AdapterRef:
    """Inverse of AdapterRef.path; raises ValueError if the layout doesn't match."""
    p = Path(path).resolve()
    r = Path(root).resolve()
    rel = p.relative_to(r)
    parts = rel.parts
    if len(parts) != 2:
        raise ValueError(f"adapter path {path} not under {root}/pNNN/vNNN")
    pm = _PROBLEM_RE.match(parts[0])
    vm = _VERSION_RE.match(parts[1])
    if not pm or not vm:
        raise ValueError(f"adapter path {path} doesn't match pNNN/vNNN scheme")
    return AdapterRef(Path(root), int(pm.group(1)), int(vm.group(1)))
