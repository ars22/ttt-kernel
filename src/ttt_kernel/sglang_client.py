"""Thin client around an SGLang HTTP server.

Two things we need:
1. Generate K parallel completions for one prompt (the rollouts for a turn).
2. Hot-swap the LoRA adapter weights between turns without restarting the server.

SGLang exposes an OpenAI-compatible /v1/chat/completions endpoint *and*
admin endpoints for LoRA adapter management. We use httpx so we can fan out
the K samples concurrently in one event loop.

httpx.AsyncClient instances are bound to the event loop they're created in.
The sync wrappers below call `asyncio.run()` which spins up a fresh loop each
time, so the client is created INSIDE each coroutine and closed before the
loop exits — that keeps socket cleanup on the right loop.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import List, Optional

import httpx


@dataclass
class GenerationResult:
    text: str
    finish_reason: str
    completion_tokens: int


class SGLangClient:
    def __init__(self, cfg, model_name: str):
        self.base_url = cfg.base_url.rstrip("/")
        self.adapter_name = cfg.adapter_name
        self.adapter_out_dir = os.path.abspath(cfg.adapter_out_dir)
        self.update_endpoint = cfg.update_weights_endpoint
        self.model_name = model_name
        # Big budget: with K=16 and max_tokens=64k on dp>=1, a single rollout
        # can take 30+ minutes when many requests are queued per replica.
        self._timeout = httpx.Timeout(3600.0)

    # ---- generation --------------------------------------------------------

    async def _one_completion(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        adapter_name: Optional[str] = None,
    ) -> GenerationResult:
        # /v1/chat/completions so SGLang applies the model's chat template
        # automatically. Instruct-style models degenerate badly on raw
        # /v1/completions because no special tokens (assistant marker etc.)
        # frame the response. KernelBench's prompt is a single self-contained
        # user turn, so wrapping it as one `user` message is faithful.
        # adapter_name: which loaded LoRA to use; None = base model.
        payload = {
            "model": adapter_name if adapter_name is not None else self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
        }
        r = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        text = choice.get("message", {}).get("content") or choice.get("text", "")
        return GenerationResult(
            text=text,
            finish_reason=choice.get("finish_reason", "stop"),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

    async def sample_async(
        self,
        prompt: str,
        *,
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        adapter_name: Optional[str] = None,
    ) -> List[GenerationResult]:
        """Async fan-out of n rollouts for one prompt against the given adapter."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            tasks = [
                self._one_completion(
                    client, prompt,
                    temperature=temperature, top_p=top_p,
                    max_tokens=max_tokens, adapter_name=adapter_name,
                )
                for _ in range(n)
            ]
            return await asyncio.gather(*tasks)

    def sample(
        self,
        prompt: str,
        *,
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        adapter_name: Optional[str] = None,
    ) -> List[GenerationResult]:
        """Sync wrapper for sample_async (spins its own event loop)."""
        return asyncio.run(self.sample_async(
            prompt, n=n, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, adapter_name=adapter_name,
        ))

    # ---- dynamic adapter management (async) -------------------------------

    async def load_adapter_async(self, name: str, path: str) -> None:
        """Register a LoRA adapter with the server (or refresh if already loaded).

        On failure we still raise — the caller can decide whether to retry.
        Use reload_adapter_async if you want unload-then-load semantics for
        a hot-swap.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(
                f"{self.base_url}/load_lora_adapter",
                json={"lora_name": name, "lora_path": os.path.abspath(path)},
            )
            r.raise_for_status()

    async def unload_adapter_async(self, name: str) -> None:
        """Drop a LoRA adapter slot. Best-effort: 4xx is swallowed (not loaded)."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                await client.post(
                    f"{self.base_url}/unload_lora_adapter",
                    json={"lora_name": name},
                )
            except httpx.HTTPError:
                pass

    async def reload_adapter_async(self, name: str, path: str) -> None:
        """Unload then re-load the adapter so SGLang re-reads weights from disk."""
        await self.unload_adapter_async(name)
        try:
            await self.load_adapter_async(name, path)
        except httpx.HTTPError as e:
            # dp>1 servers reject dynamic LoRA — swallow so training continues.
            print(f"[sglang] hot-swap of '{name}' failed ({e}); continuing")

    # ---- LoRA hot-swap (legacy default-adapter shim) ----------------------

    def reload_adapter(self) -> None:
        """Compatibility shim: refresh the default `adapter_name` from disk."""
        asyncio.run(self.reload_adapter_async(self.adapter_name, self.adapter_out_dir))

    # ---- health check ------------------------------------------------------

    async def wait_ready_async(self, timeout_s: float = 120.0) -> None:
        """Poll /v1/models until the server responds 200 or timeout elapses."""
        import time

        deadline = time.time() + timeout_s
        last_err = None
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            while time.time() < deadline:
                try:
                    r = await client.get(f"{self.base_url}/v1/models")
                    if r.status_code == 200:
                        return
                except Exception as e:  # noqa: BLE001
                    last_err = e
                await asyncio.sleep(2.0)
        raise RuntimeError(
            f"SGLang at {self.base_url} not ready in {timeout_s}s: {last_err}"
        )

    def wait_ready(self, timeout_s: float = 120.0) -> None:
        """Sync wrapper. Caller must NOT already be inside a running event loop."""
        asyncio.run(self.wait_ready_async(timeout_s))
