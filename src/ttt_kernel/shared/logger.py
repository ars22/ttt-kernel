"""JSONL + (optional) wandb logger.

One run → one events.jsonl on disk, and if wandb is enabled, scalar fields are
also pushed there. JSONL is always the ground truth; wandb is a mirror for the
metrics the user wants to watch live.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


class JsonlLogger:
    def __init__(
        self,
        out_dir: str,
        run_name: Optional[str],
        wandb_cfg=None,
        full_config: Optional[dict] = None,
    ):
        run_name = run_name or f"ttt_{int(time.time())}"
        self.run_name = run_name
        self.run_dir = Path(out_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"
        self._fp = open(self.path, "a", buffering=1)  # line-buffered

        self._wandb = None
        self._step = 0
        # Track which problem ids we've already declared per-problem
        # step-metrics for in wandb, so per-problem panels use `turn` as x-axis.
        self._problem_metrics_defined: set[int] = set()
        if wandb_cfg is not None and getattr(wandb_cfg, "enabled", False):
            import wandb  # imported lazily so the dep is optional

            self._wandb = wandb.init(
                project=wandb_cfg.project,
                entity=wandb_cfg.entity,
                name=wandb_cfg.run_name or run_name,
                mode=wandb_cfg.mode,
                tags=list(wandb_cfg.tags or []),
                config=full_config or {},
                dir=str(self.run_dir),
                resume="allow",
            )

    # --- per-problem metric scoping ----------------------------------------
    def _ensure_problem_metrics(self, problem_id: int) -> None:
        """Declare a per-problem step-metric so wandb plots turn on the x-axis.

        Per-problem keys live under `problem_{pid:03d}/*` and use
        `problem_{pid:03d}/turn` as their step metric. Without this, wandb
        would plot them against the global monotonic step and turn-curves
        from different problems would interleave on the same x-axis.
        """
        if self._wandb is None or problem_id in self._problem_metrics_defined:
            return
        ns = f"problem_{problem_id:03d}"
        self._wandb.define_metric(f"{ns}/turn")
        self._wandb.define_metric(f"{ns}/*", step_metric=f"{ns}/turn")
        self._problem_metrics_defined.add(problem_id)

    def log(self, event: str, **fields: Any) -> None:
        rec = {"t": time.time(), "event": event, **fields}
        self._fp.write(json.dumps(rec, default=str) + "\n")
        if self._wandb is None:
            return

        # Per-turn events get logged under a per-problem namespace so each
        # problem gets its own set of turn-indexed plots in wandb.
        if event == "turn" and "problem_id" in fields and "turn" in fields:
            pid = int(fields["problem_id"])
            turn = int(fields["turn"])
            self._ensure_problem_metrics(pid)
            ns = f"problem_{pid:03d}"
            skip = {"problem_id", "turn", "kind", "problem_name", "pair"}
            payload: dict[str, Any] = {f"{ns}/turn": turn}
            for k, v in fields.items():
                if k in skip:
                    continue
                if isinstance(v, (int, float, bool)):
                    payload[f"{ns}/{k}"] = v
            self._wandb.log(payload)
            return

        scalars = {
            f"{event}/{k}": v
            for k, v in fields.items()
            if isinstance(v, (int, float, bool))
        }
        if scalars:
            # No explicit step: wandb auto-increments. Mixing explicit step=
            # with per-problem logs (which omit step to honor their declared
            # step_metric) risks monotonicity violations and silent drops.
            self._wandb.log(scalars)

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()
        if self._wandb is not None:
            self._wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
