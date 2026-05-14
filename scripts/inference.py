#!/usr/bin/env python
"""ttt-kernel entry point — pool-based online-RL orchestrator.

Launches N (sampler, trainer) pairs and dispatches problems to them.
Anything in the YAML can be overridden on the CLI with `dotted.key=value`.

Usage:
    python scripts/inference.py --config configs/default.yaml \
        pool.num_pairs=4 rollout.num_samples=8 rollout.max_tokens=16384

The orchestrator itself uses asyncio; per-pair workers are subprocesses.
"""
from __future__ import annotations

import argparse
import os
import sys

from ttt_kernel.config import load_config
from ttt_kernel.orchestrator import run as run_pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides=overrides)
    print(
        f"[ttt-kernel] loaded config; launching pool: "
        f"num_pairs={cfg.pool.num_pairs} "
        f"sampler(dp={cfg.pool.sampler.dp},tp={cfg.pool.sampler.tp}) "
        f"trainer(dp={cfg.pool.trainer.dp},tp={cfg.pool.trainer.tp})",
        file=sys.stderr,
    )
    run_pool(cfg, os.path.abspath(args.config), overrides)


if __name__ == "__main__":
    main()
