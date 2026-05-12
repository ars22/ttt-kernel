"""Tiny JSONL logger — one file per run, one line per turn-rollout."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, out_dir: str, run_name: str | None):
        run_name = run_name or f"ttt_{int(time.time())}"
        self.run_dir = Path(out_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"
        self._fp = open(self.path, "a", buffering=1)  # line-buffered

    def log(self, event: str, **fields: Any) -> None:
        rec = {"t": time.time(), "event": event, **fields}
        self._fp.write(json.dumps(rec, default=str) + "\n")

    def close(self) -> None:
        self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
