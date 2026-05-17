"""Filesystem registry round-trip."""
from __future__ import annotations

import tempfile

from ttt_kernel.orchestrator.registry import mark_down, read_entries, write_entry
from ttt_kernel.shared.types import RegistryEntry


def test_registry_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        a = RegistryEntry(pool="env", idx=0, host="ha", port=8001, capacity=4)
        b = RegistryEntry(pool="env", idx=1, host="hb", port=8002, capacity=2)
        write_entry(d, a)
        write_entry(d, b)
        seen = read_entries(d, "env")
        assert sorted((e.idx, e.host, e.port, e.capacity) for e in seen) == [
            (0, "ha", 8001, 4),
            (1, "hb", 8002, 2),
        ]


def test_mark_down_filters_out_of_active_list():
    with tempfile.TemporaryDirectory() as d:
        a = RegistryEntry(pool="trainer", idx=0, host="ha", port=8101, capacity=2)
        b = RegistryEntry(pool="trainer", idx=1, host="hb", port=8102, capacity=2)
        write_entry(d, a)
        write_entry(d, b)
        mark_down(d, "trainer", 0)
        seen = read_entries(d, "trainer")
        assert [e.idx for e in seen] == [1]


def test_pools_are_namespaced_by_dir():
    with tempfile.TemporaryDirectory() as d:
        write_entry(d, RegistryEntry(pool="env", idx=0, host="ha", port=1, capacity=1))
        write_entry(d, RegistryEntry(pool="sampler", idx=0, host="hb", port=2, capacity=1))
        write_entry(d, RegistryEntry(pool="trainer", idx=0, host="hc", port=3, capacity=1))
        assert len(read_entries(d, "env")) == 1
        assert len(read_entries(d, "sampler")) == 1
        assert len(read_entries(d, "trainer")) == 1
