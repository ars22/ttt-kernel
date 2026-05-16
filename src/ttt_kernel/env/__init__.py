"""Env pool — KernelBench eval-as-a-service.

One process per env node owns M subprocess sandbox slots (M =
`max_concurrent`). Each slot is a long-lived `env.eval_worker` subprocess
owning its own CUDA context — so a kernel that poisons CUDA only kills
its own slot, not the whole service.

Layout:

- `eval_worker.py` — the sandbox subprocess (newline-delimited JSON protocol)
- `problem_set.py` — KernelBench dataset wrapper: prompts + ref_src by id
- `scoring.py`     — reward shaping (ports `_score` from old `kernel_env.py`)
- `pool.py`        — async pool of M sandbox slots + capacity tracking
- `server.py`      — FastAPI: /evaluate /healthz /capacity /problems
- `client.py`      — orchestrator-side async HTTP client
"""
