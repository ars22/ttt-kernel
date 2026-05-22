"""KernelBench reward function for verl.

Loaded by verl via `reward.custom_reward_function.path = verl_mtl/reward.py`.
Per-rollout flow: verl gives us (data_source, solution_str, ground_truth, ...),
we POST {problem_id=ground_truth, completion=solution_str} to one of the env
services running on this node (reused from ttt-kernel's env/server.py), and
return the env's reward (already speedup-scaled with the same shaping the
prior ttt-kernel runs used).

Multiple env services are load-balanced round-robin; the URL list is taken
from the env var TTT_ENV_URLS (comma-separated). With the launcher's default
layout that's "http://127.0.0.1:8100,http://127.0.0.1:8101".

Circuit breaker: after N consecutive failures in this worker, future calls
short-circuit to the parse penalty for `cooldown_s` seconds instead of
hanging on dead env services. Prevents the trainer stalling forever when all
envs go down (which is what killed verl_mtl_20260521_124405 at step 10).
"""
from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from typing import Any

import requests

log = logging.getLogger(__name__)


def _env_urls() -> list[str]:
    raw = os.environ.get(
        "TTT_ENV_URLS",
        "http://127.0.0.1:8100,http://127.0.0.1:8101",
    )
    return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]


_URL_CYCLE_LOCK = threading.Lock()
_URL_CYCLE: "itertools.cycle[str] | None" = None


def _pick_url() -> str:
    global _URL_CYCLE
    with _URL_CYCLE_LOCK:
        if _URL_CYCLE is None:
            _URL_CYCLE = itertools.cycle(_env_urls())
        return next(_URL_CYCLE)


_TIMEOUT_S = float(os.environ.get("TTT_ENV_TIMEOUT_S", "300"))
_PARSE_PENALTY = float(os.environ.get("TTT_PARSE_PENALTY", "-3.0"))
_BREAKER_OPEN_AFTER = int(os.environ.get("TTT_ENV_FAIL_FAST_AFTER", "8"))
_BREAKER_COOLDOWN_S = float(os.environ.get("TTT_ENV_BREAKER_COOLDOWN_S", "60"))

_BREAKER_LOCK = threading.Lock()
_consecutive_failures = 0
_open_until = 0.0


def _record_success() -> None:
    global _consecutive_failures, _open_until
    with _BREAKER_LOCK:
        _consecutive_failures = 0
        _open_until = 0.0


def _record_failure() -> None:
    global _consecutive_failures, _open_until
    with _BREAKER_LOCK:
        _consecutive_failures += 1
        if _consecutive_failures >= _BREAKER_OPEN_AFTER:
            _open_until = time.monotonic() + _BREAKER_COOLDOWN_S
            log.warning(
                "env circuit breaker OPEN: %d consecutive failures, "
                "short-circuiting calls for %.0fs",
                _consecutive_failures, _BREAKER_COOLDOWN_S,
            )


def _breaker_open() -> bool:
    with _BREAKER_LOCK:
        return time.monotonic() < _open_until


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    if data_source != "kernelbench":
        raise ValueError(f"verl_mtl/reward.compute_score: unexpected data_source {data_source!r}")

    try:
        problem_id = int(ground_truth)
    except (TypeError, ValueError) as e:
        raise ValueError(f"ground_truth must coerce to int problem_id, got {ground_truth!r}") from e

    if _breaker_open():
        return _PARSE_PENALTY

    url = _pick_url() + "/evaluate"
    body = {
        "problem_id": problem_id,
        "turn": 0,
        "sample": int((extra_info or {}).get("sample_idx", 0)),
        "completion": solution_str,
    }

    try:
        r = requests.post(url, json=body, timeout=_TIMEOUT_S)
        r.raise_for_status()
        payload = r.json()
        reward = float(payload.get("reward", _PARSE_PENALTY))
        _record_success()
    except requests.RequestException as e:
        log.warning("env eval failed for problem %d at %s: %s", problem_id, url, e)
        _record_failure()
        reward = _PARSE_PENALTY

    return reward
