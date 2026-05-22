"""Sampler pool — thin FastAPI shim around an SGLang server.

Layout:
- client.py    HTTP client used by the orchestrator (httpx.AsyncClient).
- server.py    FastAPI service: /sample, /load_lora_adapter, /healthz, /capacity.

The SGLang server is launched separately (scripts/launch_sglang.sh); the
shim adds capacity bookkeeping + auto-LRU adapter management so the
orchestrator can hand it a Weka adapter path and forget.
"""
