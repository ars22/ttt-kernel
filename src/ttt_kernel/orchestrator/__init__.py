"""Central orchestrator — drives per-problem state machines across the three pools.

Layout:
- registry.py     Service discovery via runs/<run>/registry/{pool}/<idx>.json.
- scheduler.py    Capacity-aware pick across pool members; one Pool[T] per kind.
- problem_sm.py   Per-problem coroutine: SAMPLING → EVALUATING → TRAINING → ...
- main.py         Entrypoint: load config, build pools, fan out problems, log.
"""
