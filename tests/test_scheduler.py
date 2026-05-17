"""Capacity contention in the orchestrator's Pool."""
from __future__ import annotations

import asyncio

import pytest

from ttt_kernel.orchestrator.scheduler import Pool, _PoolMember
from ttt_kernel.shared.types import RegistryEntry


class _FakeClient:
    def __init__(self, idx: int):
        self.idx = idx

    async def aclose(self) -> None:
        pass


def _build_pool(capacities: list[int]) -> Pool:
    members = [
        _PoolMember(
            entry=RegistryEntry(pool="env", idx=i, host="h", port=8000 + i, capacity=c),
            client=_FakeClient(i),
        )
        for i, c in enumerate(capacities)
    ]
    return Pool(members)


@pytest.mark.asyncio
async def test_pool_round_robins_under_concurrent_load():
    pool = _build_pool([2, 2])  # total capacity 4

    seen_in_flight_max = 0

    async def task(i):
        nonlocal seen_in_flight_max
        async with pool.acquire() as (_c, e):
            seen_in_flight_max = max(seen_in_flight_max, pool.total_in_flight())
            await asyncio.sleep(0.01)
            return e.idx

    results = await asyncio.gather(*[task(i) for i in range(8)])
    assert sorted(results) == sorted([0, 1, 0, 1, 0, 1, 0, 1])
    # Hard cap = 4; we never exceeded it.
    assert seen_in_flight_max <= 4
    # And after everyone returns, in_flight returns to 0.
    assert pool.total_in_flight() == 0
    await pool.close()


@pytest.mark.asyncio
async def test_pool_blocks_when_at_capacity_then_releases():
    pool = _build_pool([1])  # only one slot total

    order: list[str] = []

    async def slow():
        async with pool.acquire():
            order.append("slow-acquired")
            await asyncio.sleep(0.05)
            order.append("slow-released")

    async def fast():
        # Give slow() time to acquire first.
        await asyncio.sleep(0.005)
        order.append("fast-attempting")
        async with pool.acquire():
            order.append("fast-acquired")

    await asyncio.gather(slow(), fast())
    # fast must have logged 'fast-attempting' BEFORE slow-released, and
    # 'fast-acquired' AFTER. That proves the wait-then-resume path.
    i_attempt = order.index("fast-attempting")
    i_release = order.index("slow-released")
    i_acquire = order.index("fast-acquired")
    assert i_attempt < i_release < i_acquire
    await pool.close()


@pytest.mark.asyncio
async def test_pool_picks_least_loaded_first():
    pool = _build_pool([2, 2])

    # Hold a long-running slot on node 0.
    started = asyncio.Event()
    release = asyncio.Event()

    async def hold():
        async with pool.acquire() as (_c, e):
            assert e.idx == 0  # first acquire goes to idx 0 (tie-break by idx)
            started.set()
            await release.wait()

    async def follower():
        await started.wait()
        async with pool.acquire() as (_c, e):
            return e.idx

    hold_task = asyncio.create_task(hold())
    pick = await follower()
    assert pick == 1  # least-loaded was node 1 (idx 0 has 1 in-flight)
    release.set()
    await hold_task
    await pool.close()
