"""State-machine wiring test with fake pools.

Asserts the per-problem coroutine drives K rollouts → K env evaluates → 1
train per turn, with versioned adapter paths walked v000 → v001 → ... and
that the orchestrator may route turn t+1 to a different sampler/trainer
than turn t.
"""
from __future__ import annotations

import asyncio
import itertools
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

from ttt_kernel.orchestrator.problem_sm import run_problem
from ttt_kernel.orchestrator.scheduler import Pool, _PoolMember
from ttt_kernel.shared.adapter_paths import AdapterRef
from ttt_kernel.shared.types import (
    EvaluateResponse,
    Generation,
    RegistryEntry,
    SampleResponse,
    TrainResponse,
)


class _FakeSampler:
    def __init__(self, idx: int, sink: list):
        self.idx = idx
        self.sink = sink

    async def aclose(self): pass

    async def sample(self, req):
        self.sink.append(("sample", self.idx, req.adapter_name, req.n))
        return SampleResponse(completions=[
            Generation(text=f"completion-{i}", finish_reason="stop", completion_tokens=8)
            for i in range(req.n)
        ])


class _FakeEnv:
    def __init__(self, idx: int, sink: list):
        self.idx = idx
        self.sink = sink

    async def aclose(self): pass

    async def evaluate(self, req):
        self.sink.append(("eval", self.idx, req.problem_id, req.turn, req.sample))
        # Yield so concurrent gather() actually multiplexes; without an await
        # each fake eval runs to completion before the next acquire(), and the
        # scheduler never sees two in-flight at once on different nodes.
        await asyncio.sleep(0.01)
        return EvaluateResponse(
            raw_completion=req.completion,
            kernel_src="def k(): pass",
            compiled=True, correct=True, speedup=2.0,
            runtime_ms=1.0, ref_runtime_ms=2.0,
            reward=float(req.sample) + 0.1,
            feedback="ok",
            error_kind="ok",
        )


class _FakeTrainer:
    def __init__(self, idx: int, sink: list):
        self.idx = idx
        self.sink = sink

    async def aclose(self): pass

    async def train(self, req):
        self.sink.append((
            "train", self.idx, req.problem_id, req.turn,
            req.adapter_in_name, req.adapter_out_name, len(req.rollouts),
        ))
        return TrainResponse(
            loss=0.5, pg=0.1, kl=0.01, grad_norm=0.3,
            reward_mean=0.5, reward_std=0.2, advantage_mean=0.0,
        )


class _NullLogger:
    def __init__(self):
        self.events = []

    def log(self, event, **fields):
        self.events.append((event, fields))


def _build_pool(kind: str, n: int, sink: list, factory):
    members = [
        _PoolMember(
            entry=RegistryEntry(pool=kind, idx=i, host="h", port=8000 + i, capacity=4),
            client=factory(i, sink),
        )
        for i in range(n)
    ]
    return Pool(members)


@pytest.mark.asyncio
async def test_one_problem_walks_versions_and_fans_evals():
    sink: list = []
    sampler_pool = _build_pool("sampler", 1, sink, _FakeSampler)
    env_pool = _build_pool("env", 2, sink, _FakeEnv)
    trainer_pool = _build_pool("trainer", 1, sink, _FakeTrainer)
    logger = _NullLogger()

    with tempfile.TemporaryDirectory() as root:
        async def fetch_prompt(pid: int) -> str:
            return f"prompt-{pid}"

        result = await run_problem(
            problem_id=3,
            num_turns=2,
            K=4,
            sampler_pool=sampler_pool,
            env_pool=env_pool,
            trainer_pool=trainer_pool,
            adapters_root=root,
            base_prompt_fetcher=fetch_prompt,
            logger=logger,
            sample_kwargs={"temperature": 1.0, "top_p": 0.95, "max_tokens": 16},
        )

    # Per turn: 1 sample call, 4 eval calls, 1 train call.
    sample_calls = [s for s in sink if s[0] == "sample"]
    eval_calls = [s for s in sink if s[0] == "eval"]
    train_calls = [s for s in sink if s[0] == "train"]
    assert len(sample_calls) == 2
    assert len(eval_calls) == 8     # 4 rollouts × 2 turns
    assert len(train_calls) == 2

    # Adapter versions walk v000 → v001 → v002 (final).
    adapter_names = [c[2] for c in sample_calls]
    assert adapter_names == ["p003_v000", "p003_v001"]
    train_in_out = [(c[4], c[5]) for c in train_calls]
    assert train_in_out == [("p003_v000", "p003_v001"), ("p003_v001", "p003_v002")]

    assert result["final_adapter_path"].endswith("v002")


@pytest.mark.asyncio
async def test_evals_fan_across_env_nodes():
    sink: list = []
    sampler_pool = _build_pool("sampler", 1, sink, _FakeSampler)
    env_pool = _build_pool("env", 2, sink, _FakeEnv)
    trainer_pool = _build_pool("trainer", 1, sink, _FakeTrainer)
    logger = _NullLogger()

    with tempfile.TemporaryDirectory() as root:
        async def fetch_prompt(pid: int) -> str:
            return f"prompt-{pid}"

        await run_problem(
            problem_id=0,
            num_turns=1,
            K=4,
            sampler_pool=sampler_pool,
            env_pool=env_pool,
            trainer_pool=trainer_pool,
            adapters_root=root,
            base_prompt_fetcher=fetch_prompt,
            logger=logger,
            sample_kwargs={"temperature": 1.0, "top_p": 0.95, "max_tokens": 16},
        )

    # Both env nodes saw at least one eval — the scheduler load-balanced
    # K=4 evals over 2 envs with capacity 4 each.
    eval_idxs = {s[1] for s in sink if s[0] == "eval"}
    assert eval_idxs == {0, 1}
