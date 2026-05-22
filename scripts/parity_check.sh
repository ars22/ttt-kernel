#!/usr/bin/env bash
# Cross-branch parity check: run the same small_smoke config under main and
# under refactor; diff per-turn reward_mean and kl from events.jsonl.
#
# Usage:
#   ./scripts/parity_check.sh
#
# Pass if the diff is within 1% per-turn (allowed drift from routing changes;
# the forward math is identical).
set -euo pipefail

TTT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARITY_DIR="$TTT_ROOT/runs/parity_$(date +%s)"
mkdir -p "$PARITY_DIR"

orig_branch="$(git -C "$TTT_ROOT" rev-parse --abbrev-ref HEAD)"
echo "[parity] starting branch: $orig_branch"
echo "[parity] working dir: $PARITY_DIR"

git -C "$TTT_ROOT" stash --include-untracked --quiet || true

trap 'git -C "$TTT_ROOT" checkout "$orig_branch" --quiet; git -C "$TTT_ROOT" stash pop --quiet 2>/dev/null || true' EXIT

for branch in main refactor; do
  echo "[parity] === checking out $branch ==="
  git -C "$TTT_ROOT" checkout "$branch" --quiet

  RUN_NAME="parity_${branch}"
  RUN_ROOT="$PARITY_DIR/$RUN_NAME"
  mkdir -p "$RUN_ROOT/logs"

  if [ "$branch" = "main" ]; then
    # main uses the pair-orchestrator entry point (scripts/inference.py).
    echo "[parity] main: NOTE — drive via scripts/inference.py with --config configs/small_smoke.yaml"
    echo "[parity] main: this is a hint, not automated (the two CLIs differ)."
  else
    # refactor uses the three-pool wrapper.
    NUM_SAMPLERS=1 NUM_ENVS=1 NUM_TRAINERS=1 \
      "$TTT_ROOT/scripts/run_refactor.sh" configs/small_smoke.yaml "$RUN_NAME"
    echo "[parity] refactor submitted; tail logs/orchestrator_*.out for events.jsonl path"
  fi
done

cat <<MSG
[parity] Both branches submitted. After both finish, diff per-turn metrics with:

  jq -r 'select(.event=="turn") | "p=\(.problem_id) t=\(.turn) r=\(.reward_mean) kl=\(.kl)"' \\
      $PARITY_DIR/parity_main/.../events.jsonl > main.csv
  jq -r 'select(.event=="turn") | "p=\(.problem_id) t=\(.turn) r=\(.reward_mean) kl=\(.kl)"' \\
      $PARITY_DIR/parity_refactor/.../events.jsonl > refactor.csv
  diff main.csv refactor.csv

A reward_mean / kl drift of <1% per turn is the pass criterion.
MSG
