"""Filesystem service discovery.

Each pool process writes `runs/<run>/registry/{pool}/{idx}.json` at startup
with `{host, port, capacity}` (RegistryEntry). The orchestrator polls these
directories until expected counts are seen, then connects.

This is intentionally the simplest possible discovery — no etcd, no
zookeeper. Weka is the cross-node shared FS so a write here is immediately
visible to the orchestrator anywhere in the SLURM job.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import List

from ..shared.types import PoolKind, RegistryEntry

log = logging.getLogger("ttt_kernel.orchestrator.registry")


def registry_dir(run_root: str | Path, pool: PoolKind) -> Path:
    p = Path(run_root) / "registry" / pool
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_entry(
    run_root: str | Path,
    entry: RegistryEntry,
) -> Path:
    """Atomic write so partial files are never visible."""
    d = registry_dir(run_root, entry.pool)
    final = d / f"{entry.idx:03d}.json"
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(entry.model_dump_json())
    tmp.replace(final)
    log.info("registered %s/%03d at %s:%d cap=%d",
             entry.pool, entry.idx, entry.host, entry.port, entry.capacity)
    return final


def mark_down(
    run_root: str | Path,
    pool: PoolKind,
    idx: int,
) -> None:
    d = registry_dir(run_root, pool)
    final = d / f"{idx:03d}.json"
    if not final.exists():
        return
    try:
        entry = RegistryEntry.model_validate_json(final.read_text())
        entry = entry.model_copy(update={"state": "down"})
        final.write_text(entry.model_dump_json())
    except Exception as e:  # noqa: BLE001
        log.warning("mark_down(%s/%d) failed: %s", pool, idx, e)


def read_entries(run_root: str | Path, pool: PoolKind) -> List[RegistryEntry]:
    d = registry_dir(run_root, pool)
    out: list[RegistryEntry] = []
    for f in sorted(d.glob("*.json")):
        try:
            e = RegistryEntry.model_validate_json(f.read_text())
            if e.state == "up":
                out.append(e)
        except Exception as e:  # noqa: BLE001
            log.warning("could not parse %s: %s", f, e)
    return out


def wait_for_pool(
    run_root: str | Path,
    pool: PoolKind,
    expected_count: int,
    timeout_s: float = 600.0,
    poll_s: float = 2.0,
) -> List[RegistryEntry]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        entries = read_entries(run_root, pool)
        if len(entries) >= expected_count:
            return entries
        time.sleep(poll_s)
    raise TimeoutError(
        f"pool {pool} did not register {expected_count} entries within {timeout_s}s; "
        f"saw {len(read_entries(run_root, pool))}"
    )


def base_url(entry: RegistryEntry) -> str:
    return f"http://{entry.host}:{entry.port}"
