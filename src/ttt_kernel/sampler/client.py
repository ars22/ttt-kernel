"""Async HTTP client for the sampler service.

Used by the orchestrator. One client per sampler node.
"""
from __future__ import annotations

import httpx

from ..shared.types import Capacity, SampleRequest, SampleResponse


class SamplerClient:
    def __init__(self, base_url: str, timeout_s: float = 3600.0):
        # Sampling for K=16 at 16k tokens can take 30+ minutes under queue load.
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "SamplerClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def healthz(self) -> bool:
        try:
            r = await self._client.get("/healthz", timeout=5.0)
            return r.status_code == 200 and bool(r.json().get("ok"))
        except Exception:  # noqa: BLE001
            return False

    async def capacity(self) -> Capacity:
        r = await self._client.get("/capacity", timeout=5.0)
        r.raise_for_status()
        return Capacity.model_validate(r.json())

    async def load_adapter(self, name: str, path: str) -> None:
        r = await self._client.post(
            "/load_lora_adapter",
            json={"name": name, "path": path},
            timeout=120.0,
        )
        r.raise_for_status()

    async def unload_adapter(self, name: str) -> None:
        r = await self._client.post(
            "/unload_lora_adapter",
            json={"name": name},
            timeout=60.0,
        )
        r.raise_for_status()

    async def sample(self, req: SampleRequest) -> SampleResponse:
        r = await self._client.post("/sample", json=req.model_dump())
        r.raise_for_status()
        return SampleResponse.model_validate(r.json())
