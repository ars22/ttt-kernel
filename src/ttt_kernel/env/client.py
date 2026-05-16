"""Async HTTP client for the env service.

Used by the orchestrator. One client per env-pool node; the orchestrator
picks which node to send each /evaluate to based on Capacity reports.
"""
from __future__ import annotations

from typing import List

import httpx

from ..shared.types import Capacity, EvaluateRequest, EvaluateResponse


class EnvClient:
    def __init__(self, base_url: str, timeout_s: float = 600.0):
        # Eval can be slow (kernelbench compile + perf trials); generous default.
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "EnvClient":
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

    async def list_problems(self) -> List[int]:
        r = await self._client.get("/problems", timeout=10.0)
        r.raise_for_status()
        return r.json()["problem_ids"]

    async def get_prompt(self, problem_id: int) -> str:
        r = await self._client.get(f"/problems/{problem_id}", timeout=10.0)
        r.raise_for_status()
        return r.json()["prompt"]

    async def evaluate(self, req: EvaluateRequest) -> EvaluateResponse:
        r = await self._client.post("/evaluate", json=req.model_dump())
        r.raise_for_status()
        return EvaluateResponse.model_validate(r.json())
