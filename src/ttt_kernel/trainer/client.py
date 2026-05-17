"""Async HTTP client for the trainer service."""
from __future__ import annotations

import httpx

from ..shared.types import Capacity, TrainRequest, TrainResponse


class TrainerClient:
    def __init__(self, base_url: str, timeout_s: float = 1800.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "TrainerClient":
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

    async def train(self, req: TrainRequest) -> TrainResponse:
        r = await self._client.post("/train", json=req.model_dump())
        r.raise_for_status()
        return TrainResponse.model_validate(r.json())
