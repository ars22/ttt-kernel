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
from typing import List

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
        use_adapter: bool,
    ) -> GenerationResult:
        # /v1/chat/completions so SGLang applies the model's chat template
        # automatically. Instruct-style models degenerate badly on raw
        # /v1/completions because no special tokens (assistant marker etc.)
        # frame the response. KernelBench's prompt is a single self-contained
        # user turn, so wrapping it as one `user` message is faithful.
        payload = {
            "model": self.adapter_name if use_adapter else self.model_name,
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

    async def _sample_async(
        self,
        prompt: str,
        *,
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        use_adapter: bool,
    ) -> List[GenerationResult]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            tasks = [
                self._one_completion(
                    client, prompt,
                    temperature=temperature, top_p=top_p,
                    max_tokens=max_tokens, use_adapter=use_adapter,
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
        use_adapter: bool = True,
    ) -> List[GenerationResult]:
        """Fan out n parallel rollouts for the same prompt. Sync entry point."""
        return asyncio.run(self._sample_async(
            prompt, n=n, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, use_adapter=use_adapter,
        ))

    async def _sample_many_async(
        self,
        prompts: List[str],
        *,
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        use_adapter: bool,
    ) -> List[List[GenerationResult]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            tasks = []
            for prompt in prompts:
                for _ in range(n):
                    tasks.append(self._one_completion(
                        client, prompt,
                        temperature=temperature, top_p=top_p,
                        max_tokens=max_tokens, use_adapter=use_adapter,
                    ))
            flat = await asyncio.gather(*tasks)
        out: List[List[GenerationResult]] = []
        for b in range(len(prompts)):
            out.append(flat[b * n : (b + 1) * n])
        return out

    def sample_many(
        self,
        prompts: List[str],
        *,
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        use_adapter: bool = True,
    ) -> List[List[GenerationResult]]:
        """Fan out len(prompts)*n parallel rollouts (n per prompt) in one loop.

        Lets SGLang's scheduler interleave requests across multiple problems
        instead of running them sequentially. Returns a list of length
        len(prompts); element b is a list of n GenerationResults.
        """
        return asyncio.run(self._sample_many_async(
            prompts, n=n, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, use_adapter=use_adapter,
        ))

    # ---- LoRA hot-swap -----------------------------------------------------

    async def _reload_adapter_async(self) -> None:
        # SGLang 0.5+ LoRA hot-swap = unload then re-load. There's no in-place
        # update; we have to drop the slot and re-register the path. /unload
        # may 4xx if the adapter isn't currently registered (e.g. very first
        # turn after a server bounce) — that's fine, swallow it.
        #
        # When the server was launched with dp_size>1, SGLang rejects dynamic
        # LoRA load/unload entirely. We swallow that too so the training loop
        # can continue; the trade-off is the server keeps serving the initial
        # adapter (rollouts won't reflect trainer updates).
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                await client.post(
                    f"{self.base_url}/unload_lora_adapter",
                    json={"lora_name": self.adapter_name},
                )
            except httpx.HTTPError:
                pass
            try:
                r = await client.post(
                    f"{self.base_url}/load_lora_adapter",
                    json={
                        "lora_name": self.adapter_name,
                        "lora_path": self.adapter_out_dir,
                    },
                )
                r.raise_for_status()
            except httpx.HTTPError as e:
                print(f"[sglang] LoRA hot-swap failed ({e}); continuing with stale adapter")

    def reload_adapter(self) -> None:
        """Tell SGLang to re-read the adapter from disk.

        Requires SGLang to have been started with
        `--lora-paths <adapter_name>=<adapter_out_dir> --enable-lora`.
        """
        asyncio.run(self._reload_adapter_async())

    # ---- health check ------------------------------------------------------

    async def _wait_ready_async(self, timeout_s: float) -> None:
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
        asyncio.run(self._wait_ready_async(timeout_s))
