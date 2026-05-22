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
        reasoning_effort: Optional[str] = None,
    ) -> GenerationResult:
        payload = {
            "model": adapter_name if adapter_name is not None else self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
        }
        if reasoning_effort is not None:
            # gpt-oss harmony knob: low|medium|high. SGLang forwards this into
            # the chat template so the model sees the correct reasoning budget.
            payload["reasoning_effort"] = reasoning_effort
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
        reasoning_effort: Optional[str] = None,
    ) -> List[GenerationResult]:
        tasks = [
            self.one_completion(
                prompt,
                temperature=temperature, top_p=top_p,
                max_tokens=max_tokens, adapter_name=adapter_name,
                reasoning_effort=reasoning_effort,
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

    # ---- full-model weight broadcast (multitask path) -------------------

    async def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str = "ttt_weight_update",
        backend: str = "nccl",
    ) -> None:
        """Make SGLang join a torch.distributed group with the trainer.

        SGLang's TP ranks join the group at [rank_offset, rank_offset+tp). The
        trainer occupies the first ranks. After this call the trainer's
        rank-0 can broadcast individual parameter tensors via NCCL and SGLang
        will receive + load them in `update_weights_from_distributed`.
        """
        r = await self._client.post(
            f"{self.base_url}/init_weights_update_group",
            json={
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
            timeout=600.0,
        )
        r.raise_for_status()

    async def update_weights_from_distributed(
        self,
        *,
        name: str,
        dtype: str,
        shape: list[int],
        flush_cache: bool = False,
    ) -> None:
        """Tell SGLang to receive one parameter tensor over the update group.

        The trainer's rank 0 must call torch.distributed.broadcast for the
        same tensor concurrently with this HTTP request — SGLang blocks on
        the recv until the broadcast arrives.
        """
        r = await self._client.post(
            f"{self.base_url}/update_weights_from_distributed",
            json={
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "flush_cache": flush_cache,
            },
            timeout=600.0,
        )
        r.raise_for_status()

    async def flush_cache(self) -> None:
        try:
            r = await self._client.post(f"{self.base_url}/flush_cache", json={})
            r.raise_for_status()
        except httpx.HTTPError:
            # /flush_cache exists on most SGLang versions but tolerate absence.
            pass

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
