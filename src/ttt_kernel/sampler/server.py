"""FastAPI sampler shim.

Sits in front of an SGLang server already running on this node. Adds:
- capacity bookkeeping (in_flight decode requests)
- LRU adapter management keyed by (name, path)
- a uniform /sample contract for the orchestrator

Endpoints:
- POST /sample              SampleRequest → SampleResponse
- POST /load_lora_adapter   {name, path} → {ok: true}     (manual preload)
- POST /unload_lora_adapter {name}       → {ok: true}     (manual evict)
- GET  /healthz             {ok: true}
- GET  /capacity            Capacity
- GET  /adapters            {loaded: [name, ...]}

Launch:
    python -m ttt_kernel.sampler.server \
        --sglang-url http://127.0.0.1:30000 \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --port 8002 \
        --max-concurrent 64 \
        --max-loaded-adapters 16
"""
from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..shared.types import Capacity, Generation, SampleRequest, SampleResponse
from .adapter_lru import AdapterLRU
from .sglang import SGLangWire

log = logging.getLogger("ttt_kernel.sampler.server")


class _LoadReq(BaseModel):
    name: str
    path: str


class _UnloadReq(BaseModel):
    name: str


def build_app(
    sglang_url: str,
    model_name: str,
    max_concurrent: int,
    max_loaded_adapters: int,
    wait_ready_s: float = 120.0,
) -> FastAPI:
    wire = SGLangWire(sglang_url, model_name)
    lru = AdapterLRU(wire, max_loaded=max_loaded_adapters)
    state = {"in_flight": 0}

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN001
        try:
            await wire.wait_ready(timeout_s=wait_ready_s)
        except Exception as e:  # noqa: BLE001
            log.warning("SGLang readiness probe failed: %s (continuing; /healthz will reflect)", e)
        try:
            yield
        finally:
            await wire.aclose()

    app = FastAPI(title="ttt-kernel sampler pool", lifespan=lifespan)
    app.state.wire = wire
    app.state.lru = lru
    app.state.counters = state

    @app.get("/healthz")
    async def healthz():
        try:
            await wire.wait_ready(timeout_s=2.0)
            return {"ok": True}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"sglang unready: {e}")

    @app.get("/capacity", response_model=Capacity)
    async def capacity():
        return Capacity(
            pool="sampler",
            max_concurrent=max_concurrent,
            in_flight=state["in_flight"],
        )

    @app.get("/adapters")
    async def adapters():
        return {"loaded": lru.loaded()}

    @app.post("/load_lora_adapter")
    async def load_adapter(req: _LoadReq):
        await lru.ensure_loaded(req.name, req.path)
        return {"ok": True}

    @app.post("/unload_lora_adapter")
    async def unload_adapter(req: _UnloadReq):
        await lru.unload(req.name)
        return {"ok": True}

    @app.post("/sample", response_model=SampleResponse)
    async def sample(req: SampleRequest):
        # Empty adapter handle = base-model inference. Skip the load_lora_adapter
        # call entirely and let SGLangWire fall back to self.model_name.
        use_adapter = bool(req.adapter_name) and bool(req.adapter_path)
        if use_adapter:
            await lru.ensure_loaded(req.adapter_name, req.adapter_path)
        state["in_flight"] += 1
        try:
            results = await wire.sample(
                req.prompt,
                n=req.n,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens,
                adapter_name=req.adapter_name if use_adapter else None,
                reasoning_effort=req.reasoning_effort,
            )
        finally:
            state["in_flight"] -= 1
        return SampleResponse(
            completions=[
                Generation(
                    text=r.text,
                    finish_reason=r.finish_reason,
                    completion_tokens=r.completion_tokens,
                )
                for r in results
            ]
        )

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sglang-url", required=True,
                   help="Base URL of the SGLang server on this node.")
    p.add_argument("--model", required=True,
                   help="Base model name (must match --model-path the SGLang server was launched with).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--max-concurrent", type=int, required=True,
                   help="Capacity = max in-flight /sample requests.")
    p.add_argument("--max-loaded-adapters", type=int, required=True,
                   help="LRU size for resident LoRA adapters in the SGLang server.")
    p.add_argument("--wait-ready-s", type=float, default=120.0)
    p.add_argument("--run-root", default=None,
                   help="If set, write a RegistryEntry to <run-root>/registry/sampler/<idx>.json on startup.")
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--advertise-host", default=None,
                   help="Hostname to advertise in the registry (default: socket.gethostname()).")
    p.add_argument("--log-level", default="info")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app = build_app(
        sglang_url=args.sglang_url,
        model_name=args.model,
        max_concurrent=args.max_concurrent,
        max_loaded_adapters=args.max_loaded_adapters,
        wait_ready_s=args.wait_ready_s,
    )
    if args.run_root:
        import socket
        from ..orchestrator.registry import write_entry, mark_down
        from ..shared.types import RegistryEntry
        host = args.advertise_host or socket.gethostname()
        entry = RegistryEntry(
            pool="sampler", idx=args.idx, host=host, port=args.port,
            capacity=args.max_concurrent,
        )
        write_entry(args.run_root, entry)
        try:
            uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
        finally:
            mark_down(args.run_root, "sampler", args.idx)
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
