"""LRU of resident adapters in the SGLang server.

The orchestrator hands the sampler a (name, path) per /sample call. If the
adapter is already loaded we reuse it; otherwise we load it, evicting the
least-recently-used adapter first if we are at `max_loaded`.

`name` must be globally unique per (problem_id, version) — the orchestrator
takes care of that via AdapterRef.name.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Optional

from .sglang import SGLangWire


class AdapterLRU:
    def __init__(self, wire: SGLangWire, max_loaded: int):
        if max_loaded < 1:
            raise ValueError(f"max_loaded must be >=1, got {max_loaded}")
        self.wire = wire
        self.max_loaded = max_loaded
        # name -> path (path is informational; the order in this OrderedDict
        # is the LRU order: most-recently-used at the end).
        self._loaded: OrderedDict[str, str] = OrderedDict()
        # Serialize touch+load+evict; SGLang adapter-mgmt is already serialized
        # internally but we also want the LRU bookkeeping to be atomic.
        self._lock = asyncio.Lock()

    async def ensure_loaded(self, name: str, path: str) -> None:
        async with self._lock:
            if name in self._loaded:
                self._loaded.move_to_end(name)
                # If the path changed but name didn't, refresh weights.
                if self._loaded[name] != path:
                    await self.wire.reload_adapter(name, path)
                    self._loaded[name] = path
                return
            while len(self._loaded) >= self.max_loaded:
                evict_name, _ = self._loaded.popitem(last=False)
                await self.wire.unload_adapter(evict_name)
            await self.wire.load_adapter(name, path)
            self._loaded[name] = path

    async def unload(self, name: str) -> None:
        async with self._lock:
            if name in self._loaded:
                self._loaded.pop(name)
                await self.wire.unload_adapter(name)

    def loaded(self) -> list[str]:
        return list(self._loaded.keys())
