"""Cross-pool shared types, paths, and logger.

This package holds anything that the orchestrator AND one or more pool
services (sampler, env, trainer) need to agree on:

- wire types (Pydantic models that travel over HTTP)
- adapter path scheme (the only cross-pool state, on Weka)
- the JSONL/wandb logger (orchestrator-owned but reused by tests)
"""
