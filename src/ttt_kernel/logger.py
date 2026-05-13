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

    def log(self, event: str, **fields: Any) -> None:
        rec = {"t": time.time(), "event": event, **fields}
        self._fp.write(json.dumps(rec, default=str) + "\n")
        if self._wandb is not None:
            scalars = {
                f"{event}/{k}": v
                for k, v in fields.items()
                if isinstance(v, (int, float, bool))
            }
            if scalars:
                self._step += 1
                self._wandb.log(scalars, step=self._step)

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()
        if self._wandb is not None:
            self._wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
