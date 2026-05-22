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
        inference_only: bool = False,
    ):
        run_name = run_name or f"ttt_{int(time.time())}"
        self.run_name = run_name
        self.run_dir = Path(out_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"
        self._fp = open(self.path, "a", buffering=1)  # line-buffered

        self._wandb = None
        self._step = 0
        self._inference_only = inference_only
        # Track which problem ids we've already declared per-problem
        # step-metrics for in wandb, so per-problem panels use `turn` as x-axis.
        self._problem_metrics_defined: set[int] = set()
        # Running aggregates for inference-only mode (no per-problem panels).
        # Each `turn` event adds to these and we re-emit the agg/* scalars.
        self._agg = {
            "n_problems": 0,
            "n_failed": 0,
            "sum_reward_mean": 0.0,
            "sum_reward_max": 0.0,
            "sum_n_correct": 0,
            "sum_n_compiled": 0,
            "sum_samples": 0,            # total completions evaluated
            "n_pass_at_k": 0,            # problems with any-correct
            "n_compiled_any": 0,         # problems with any-compile
            "sum_completion_tokens": 0,
            "n_sample_events": 0,
        }
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
            if self._inference_only:
                # One step-axis for all aggregate panels: problems completed.
                self._wandb.define_metric("agg/n_problems")
                self._wandb.define_metric("agg/*", step_metric="agg/n_problems")

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

        # Per-turn events: in training, log under a per-problem namespace so each
        # problem gets its own turn-indexed panel. In inference-only, skip those
        # 100-channel panels and update running aggregates instead.
        if event == "turn" and "problem_id" in fields and "turn" in fields:
            if self._inference_only:
                self._update_agg_from_turn(fields)
                return
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

        if self._inference_only and event == "problem_failed":
            self._agg["n_failed"] += 1
            self._wandb.log({"agg/n_failed": self._agg["n_failed"]})
            return

        if self._inference_only and event == "sample_done":
            ct = fields.get("completion_tokens")
            if isinstance(ct, (int, float)):
                self._agg["sum_completion_tokens"] += int(ct)
                self._agg["n_sample_events"] += 1
            # Don't emit per-event scalars — covered by the running aggregate.
            return

        if self._inference_only and event in {"problem_start", "sample_start",
                                              "problem_done", "evaluate_done"}:
            # These are bookkeeping events; skip wandb to avoid noise. The
            # JSONL still has the ground truth on disk.
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

    def _update_agg_from_turn(self, fields: dict) -> None:
        """Update running aggregates from a `turn` event and emit agg/* scalars."""
        a = self._agg
        a["n_problems"] += 1
        rm = float(fields.get("reward_mean", 0.0))
        rmax = float(fields.get("reward_max", 0.0))
        nc = int(fields.get("n_correct", 0))
        ncomp = int(fields.get("n_compiled", 0))
        # Total samples per problem isn't in the event; infer it from the K
        # config or default to len(rewards) if present. We assume `K=4` matches
        # n_compiled+wrongs; fall back to 1 to avoid divide-by-zero.
        a["sum_reward_mean"] += rm
        a["sum_reward_max"] += rmax
        a["sum_n_correct"] += nc
        a["sum_n_compiled"] += ncomp
        if nc > 0:
            a["n_pass_at_k"] += 1
        if ncomp > 0:
            a["n_compiled_any"] += 1
        n = max(a["n_problems"], 1)
        avg_tok = (a["sum_completion_tokens"] / a["n_sample_events"]
                   if a["n_sample_events"] else 0)
        self._wandb.log({
            "agg/n_problems":       a["n_problems"],
            "agg/n_failed":         a["n_failed"],
            "agg/reward_mean":      a["sum_reward_mean"] / n,
            "agg/reward_max_mean":  a["sum_reward_max"]  / n,
            "agg/pass_at_k_rate":   a["n_pass_at_k"]     / n,
            "agg/compile_any_rate": a["n_compiled_any"]  / n,
            "agg/sum_n_correct":    a["sum_n_correct"],
            "agg/sum_n_compiled":   a["sum_n_compiled"],
            "agg/mean_completion_tokens": avg_tok,
        })

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()
        if self._wandb is not None:
            self._wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
