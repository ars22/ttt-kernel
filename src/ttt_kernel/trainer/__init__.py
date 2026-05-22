"""Trainer pool — HF + PEFT base model with a dictionary of resident LoRA adapters.

Each trainer service owns one copy of the base model (single GPU in task #5;
FSDP2-sharded in task #6) plus an LRU of adapters, each with its own AdamW
state. /train runs one GRPO update on the named adapter; concurrent updates
on *different* adapters share the base forward and run in parallel.

Layout:
- grpo.py            GRPO step (ported from grpo_trainer.py).
- adapter_manager.py LRU + per-adapter locks + save_adapter.
- model.py           PEFT-wrapped base model factory (single GPU here; FSDP2 later).
- server.py          FastAPI: /train, /healthz, /capacity.
- client.py          httpx async client.
"""
