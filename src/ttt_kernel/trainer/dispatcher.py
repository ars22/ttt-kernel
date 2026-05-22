"""Rank-0-driven collective dispatcher.

Under torchrun every rank must participate in every FSDP collective. The
HTTP server only lives on rank 0, so we need a way for rank 0 to broadcast
"do this step" to all other ranks before each forward/backward/save.

Design:
- Each rank runs a `dispatch_loop()` in a dedicated thread.
- Rank 0 enqueues `(command_str, payload_dict)`; the loop pops it.
- All ranks then call `broadcast_object_list([cmd, payload])` — the
  payload is JSON-safe so pickling works across hosts.
- All ranks execute the handler for that cmd in lockstep.
- Rank 0 stores the result in a Future the HTTP handler is awaiting.

Commands: 'train', 'save', 'add_adapter', 'exit'.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch.distributed as dist

log = logging.getLogger("ttt_kernel.trainer.dispatcher")


@dataclass
class _Job:
    cmd: str
    payload: dict
    future: Optional["asyncio.Future"]   # only set on rank 0


class Dispatcher:
    """Broadcasts work from rank 0 to all ranks and runs handlers in lockstep."""

    def __init__(self, rank: int, world: int, handlers: dict[str, Callable[[dict], dict]]):
        self.rank = rank
        self.world = world
        self.handlers = handlers
        self._inbox: "queue.Queue[_Job]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = False

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the asyncio loop used to resolve futures (rank 0 only)."""
        self._loop = loop

    def submit(self, cmd: str, payload: dict) -> "asyncio.Future":
        """Rank-0 entry point: enqueue a job, return a future for the result."""
        if self.rank != 0:
            raise RuntimeError("Dispatcher.submit is rank-0 only")
        if self._loop is None:
            raise RuntimeError("Dispatcher.bind_loop must be called before submit")
        fut: asyncio.Future = self._loop.create_future()
        self._inbox.put(_Job(cmd=cmd, payload=payload, future=fut))
        return fut

    def run_forever(self) -> None:
        """Worker thread: loop until 'exit' is broadcast.

        Rank 0 MUST enter the broadcast every iteration even when its inbox
        is empty — otherwise non-rank-0 ranks (which unconditionally call
        broadcast_object_list) block on the collective and NCCL's 10-minute
        watchdog kills them. When the inbox is empty rank 0 broadcasts a
        'noop' command that other ranks recognize and skip.
        """
        while not self._stop:
            job: Optional[_Job] = None
            if self.rank == 0:
                try:
                    job = self._inbox.get(timeout=0.1)
                    obj = [job.cmd, job.payload]
                except queue.Empty:
                    obj = ["noop", {}]
            else:
                obj = [None, None]
            if self.world > 1 and dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)
            cmd, payload = obj[0], obj[1]
            if cmd == "noop":
                continue
            if cmd == "exit":
                self._stop = True
                if self.rank == 0 and job is not None and job.future is not None and not job.future.done():
                    self._loop.call_soon_threadsafe(job.future.set_result, {"ok": True})
                break
            try:
                handler = self.handlers[cmd]
            except KeyError:
                err = f"unknown cmd '{cmd}'"
                log.error(err)
                if self.rank == 0 and job is not None:
                    self._loop.call_soon_threadsafe(job.future.set_exception, RuntimeError(err))
                continue
            try:
                result = handler(payload or {})
            except BaseException as e:  # noqa: BLE001
                log.exception("handler '%s' raised", cmd)
                if self.rank == 0 and job is not None and job.future is not None and not job.future.done():
                    self._loop.call_soon_threadsafe(job.future.set_exception, e)
                continue
            if self.rank == 0 and job is not None and job.future is not None and not job.future.done():
                self._loop.call_soon_threadsafe(job.future.set_result, result or {})

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run_forever, name="trainer-dispatcher", daemon=True)
        self._thread.start()

    def stop_blocking(self) -> None:
        if self.rank == 0:
            # Issue an exit broadcast so non-rank ranks unblock from
            # broadcast_object_list and exit their loop.
            self._inbox.put(_Job(cmd="exit", payload={}, future=None))
        if self._thread is not None:
            self._thread.join(timeout=10)
