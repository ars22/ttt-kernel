#!/usr/bin/env python
"""ttt-kernel entry point — the online-RL inference loop.

Usage:
    python scripts/inference.py --config configs/default.yaml \
        kernelbench.level=1 kernelbench.problem_ids=[1] loop.num_turns=5 \
        rollout.num_samples=8 lora.r=16

Anything in the YAML can be overridden on the CLI with `dotted.key=value`.
"""
from __future__ import annotations

import argparse
import sys

from ttt_kernel.config import load_config
from ttt_kernel.loop import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides=overrides)
    print(f"[ttt-kernel] loaded config; starting run", file=sys.stderr)
    run(cfg)


if __name__ == "__main__":
    main()
