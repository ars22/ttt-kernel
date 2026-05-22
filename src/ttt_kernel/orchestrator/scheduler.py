"""Capacity-aware pool routing.

One `Pool[T]` per kind (sampler/env/trainer). Each pool wraps N typed
clients (one per registered node) with `max_concurrent` slots per node and
an asyncio.Event that fires whenever any slot frees. Callers use the
`acquire(...)` async context manager to reserve a slot on the least-loaded
node with capacity.

The pool tracks `in_flight` LOCALLY for each node. The remote /capacity
endpoint is the source of truth used at startup; once running, we trust
our own counter (the orchestrator is the only client).
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Generic, List, TypeVar

from ..shared.types import RegistryEntry

log = logging.getLogger("ttt_kernel.orchestrator.scheduler")

T = TypeVar("T")


@dataclass
class _PoolMember(Generic[T]):
    entry: RegistryEntry
    client: T
    in_flight: int = 0

    def free(self) -> int:
        return self.entry.capacity - self.in_flight


class Pool(Generic[T]):
    """Capacity-tracked round-robin / least-loaded pool over typed clients."""

    def __init__(self, members: List[_PoolMember[T]]):
        if not members:
            raise ValueError("Pool needs at least one member")
        self._members = members
        # Fires on each release so a waiter wakes when capacity returns.
        self._slot_event = asyncio.Event()

    @property
    def members(self) -> List[_PoolMember[T]]:
        return self._members

    def total_capacity(self) -> int:
        return sum(m.entry.capacity for m in self._members)

    def total_in_flight(self) -> int:
        return sum(m.in_flight for m in self._members)

    async def _pick_free(self) -> _PoolMember[T]:
        while True:
            self._slot_event.clear()
            free = [m for m in self._members if m.free() > 0]
            if free:
                # Least-loaded; ties broken by entry idx for reproducibility.
                free.sort(key=lambda m: (m.in_flight, m.entry.idx))
                return free[0]
            # All full → wait for someone to release.
            await self._slot_event.wait()

    @asynccontextmanager
    async def acquire(self):
        member = await self._pick_free()
        member.in_flight += 1
        try:
            yield member.client, member.entry
        finally:
            member.in_flight -= 1
            self._slot_event.set()

    async def close(self) -> None:
        for m in self._members:
            close = getattr(m.client, "aclose", None)
            if close is None:
                continue
            try:
                await close()
            except Exception as e:  # noqa: BLE001
                log.warning("close failed for %s/%d: %s", m.entry.pool, m.entry.idx, e)


def build_pool(entries: List[RegistryEntry], client_factory) -> Pool:
    """Construct a Pool from registry entries; `client_factory(base_url)` makes
    one client per entry."""
    from .registry import base_url as _bu
    members = [
        _PoolMember(entry=e, client=client_factory(_bu(e)))
        for e in entries
    ]
    return Pool(members)
