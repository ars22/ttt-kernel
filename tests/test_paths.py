"""Version-scheme round-trip for AdapterRef."""
from __future__ import annotations

import os
import tempfile

import pytest

from ttt_kernel.shared.adapter_paths import AdapterRef, parse, seed


def test_seed_layout():
    with tempfile.TemporaryDirectory() as root:
        ref = seed(root, 7)
        assert ref.version == 0
        assert ref.problem_id == 7
        assert ref.path.parts[-2:] == ("p007", "v000")
        assert ref.name == "p007_v000"


def test_next_bumps_version():
    with tempfile.TemporaryDirectory() as root:
        ref = seed(root, 12).next().next()
        assert ref.version == 2
        assert ref.name == "p012_v002"
        assert ref.path.parts[-2:] == ("p012", "v002")


def test_parse_inverse_of_path():
    with tempfile.TemporaryDirectory() as root:
        ref = AdapterRef(root=__import__("pathlib").Path(root), problem_id=4, version=9)
        # Materialize the dir so parse's relative_to resolution works.
        os.makedirs(ref.path, exist_ok=True)
        round_trip = parse(ref.path, root)
        assert round_trip == ref


def test_parse_rejects_bad_layout():
    with tempfile.TemporaryDirectory() as root:
        bad = os.path.join(root, "not-a-problem", "v000")
        os.makedirs(bad, exist_ok=True)
        with pytest.raises(ValueError):
            parse(bad, root)
