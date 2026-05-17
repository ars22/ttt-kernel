"""Async wire to the SGLang HTTP server.

Lifted from `sglang_client.py` and made event-loop-resident: one shared
AsyncClient owned by the shim's lifespan, instead of spinning one per call.
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


class SGLangWire:
    """One persistent client against an SGLang server.

    All adapter-management calls (load/unload) are serialized under a single
    asyncio.Lock — SGLang's manager rejects concurrent POSTs to
    /load_lora_adapter with HTTP 400.
    """

    def __init__(self, base_url: str, model_name: str, timeout_s: float = 3600.0):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))
        self._adapter_op_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- generation -------------------------------------------------------

    async def one_completion(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        adapter_name: Optional[str] = None,
    ) -> GenerationResult:
        payload = {
            "model": adapter_name if adapter_name is not None else self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
        }
        r = await self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
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

    async def sample(
        self,
        prompt: str,
        *,
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        adapter_name: Optional[str] = None,
    ) -> List[GenerationResult]:
        tasks = [
            self.one_completion(
                prompt,
                temperature=temperature, top_p=top_p,
                max_tokens=max_tokens, adapter_name=adapter_name,
            )
            for _ in range(n)
        ]
        return await asyncio.gather(*tasks)

    # ---- adapter management ----------------------------------------------

    async def load_adapter(self, name: str, path: str) -> None:
        async with self._adapter_op_lock:
            r = await self._client.post(
                f"{self.base_url}/load_lora_adapter",
                json={"lora_name": name, "lora_path": os.path.abspath(path)},
            )
            r.raise_for_status()

    async def unload_adapter(self, name: str) -> None:
        async with self._adapter_op_lock:
            try:
                await self._client.post(
                    f"{self.base_url}/unload_lora_adapter",
                    json={"lora_name": name},
                )
            except httpx.HTTPError:
                pass

    async def reload_adapter(self, name: str, path: str) -> None:
        """Unload then re-load so SGLang re-reads weights from disk."""
        async with self._adapter_op_lock:
            try:
                await self._client.post(
                    f"{self.base_url}/unload_lora_adapter",
                    json={"lora_name": name},
                )
            except httpx.HTTPError:
                pass
            r = await self._client.post(
                f"{self.base_url}/load_lora_adapter",
                json={"lora_name": name, "lora_path": os.path.abspath(path)},
            )
            r.raise_for_status()

    async def wait_ready(self, timeout_s: float = 120.0) -> None:
        import time
        deadline = time.time() + timeout_s
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                r = await self._client.get(f"{self.base_url}/v1/models", timeout=5.0)
                if r.status_code == 200:
                    return
            except Exception as e:  # noqa: BLE001
                last_err = e
            await asyncio.sleep(2.0)
        raise RuntimeError(
            f"SGLang at {self.base_url} not ready in {timeout_s}s: {last_err}"
        )
