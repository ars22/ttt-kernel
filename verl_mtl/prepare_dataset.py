"""Generate a verl-compatible parquet dataset from KernelBench.

Each row corresponds to one KernelBench problem. verl's GRPO trainer treats
each row as one "task" and draws K rollouts per row per step (multitask =
N rows sampled per step from this dataset).

Schema (verl convention):
  data_source : str          ("kernelbench")
  prompt      : list[dict]   ([{"role":"user","content": ... }])
  ability     : str          ("code")
  reward_model: dict         ({"style":"rule","ground_truth": <problem_id>})
  extra_info  : dict         ({"level": L, "problem_id": pid, "name": N, "index": i})

Run (one-shot):
  ./.venv-verl/bin/python verl_mtl/prepare_dataset.py \
      --kernelbench-repo /weka/.../KernelBench \
      --out-dir verl_mtl/data \
      --level 1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def _add_kernelbench_to_path(repo_path: str) -> None:
    src = os.path.join(repo_path, "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kernelbench-repo", required=True,
                   help="Local KernelBench checkout (its src/ is added to sys.path).")
    p.add_argument("--out-dir", required=True, help="Where to write train.parquet / val.parquet.")
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--dataset-src", default="huggingface")
    p.add_argument("--dataset-name", default="ScalingIntelligence/KernelBench")
    p.add_argument("--backend", default="cuda")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--prompt-option", default="one_shot")
    p.add_argument("--problem-ids", default=None,
                   help="Optional comma-separated subset of problem_ids.")
    args = p.parse_args()

    _add_kernelbench_to_path(args.kernelbench_repo)
    from kernelbench import dataset as kb_dataset
    from kernelbench import prompt_constructor_toml as kb_prompts

    ds = kb_dataset.construct_kernelbench_dataset(
        level=args.level,
        source=args.dataset_src,
        dataset_name=args.dataset_name,
    )

    pids = (
        [int(x) for x in args.problem_ids.split(",") if x.strip()]
        if args.problem_ids
        else [p.problem_id for p in ds]
    )

    rows = []
    for idx, pid in enumerate(pids):
        prob = ds.get_problem_by_id(pid)
        prompt_text = kb_prompts.get_prompt_for_backend(
            prob.code,
            args.backend,
            option=args.prompt_option,
            precision=args.precision,
            include_hardware=False,
            gpu_name=None,
        )
        rows.append({
            "data_source": "kernelbench",
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": str(pid)},
            "extra_info": {
                "level": args.level,
                "problem_id": pid,
                "name": prob.name,
                "index": idx,
            },
        })

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    df.to_parquet(train_path, index=False)
    # We have no held-out problems for KernelBench level-1 (100 problems is small);
    # write the same 100 to val.parquet so verl's eval loop has something to chew on
    # (val is sampled with rollout temp=0; useful for tracking deterministic-pass-rate).
    df.to_parquet(val_path, index=False)
    print(f"wrote {len(df)} rows → {train_path}")
    print(f"wrote {len(df)} rows → {val_path}")


if __name__ == "__main__":
    main()
