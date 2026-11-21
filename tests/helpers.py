from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from memdiag.models import Sample, SampleSet
from memdiag.plan import load_plan, load_samples


ROOT = Path(__file__).resolve().parents[1]


def fixture_plan():
    return load_plan(ROOT / "data" / "plan.json")


def fixture_samples():
    return load_samples(ROOT / "data" / "samples.json")


def tiny_plan_dict(plan_version: int = 1, budget_mib: int = 4):
    return {
        "format_version": 1,
        "plan_version": plan_version,
        "seed": 1729,
        "block_size_kb": 16,
        "memory_budget_mib": budget_mib,
        "max_batch_size": 2,
        "layers": [
            {
                "id": "a",
                "input_dim": 4,
                "output_dim": 5,
                "param_count": 512,
                "precision": "float32",
                "scale": 0,
            },
            {
                "id": "b",
                "input_dim": 5,
                "output_dim": 4,
                "param_count": 1024,
                "precision": "fixed",
                "scale": 4,
            },
            {
                "id": "c",
                "input_dim": 4,
                "output_dim": 3,
                "param_count": 768,
                "precision": "fixed",
                "scale": 5,
            },
        ],
        "consumers": {"a": ["b"], "b": ["c"], "c": []},
        "memory_derived_from": {"a": [], "b": ["a"], "c": ["b"]},
    }


def tiny_samples(count: int = 6) -> SampleSet:
    return SampleSet(
        format_version=1,
        seed=1729,
        dim=4,
        samples=tuple(
            Sample(
                sample_id=f"s{index:04d}",
                values=(
                    float(index) / 7.0,
                    (float(index) - 2.0) / 5.0,
                    0.125 * index,
                    -0.2 + 0.03 * index,
                ),
            )
            for index in range(count)
        ),
    )


def read_fixture_json(name: str) -> dict[str, Any]:
    with (ROOT / "data" / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cleanup_tree(path: str | Path) -> None:
    root = Path(path)
    if not root.exists():
        return
    for current, directories, files in os.walk(root, topdown=False):
        for name in files + directories:
            candidate = Path(current) / name
            try:
                os.chmod(candidate, 0o700)
            except OSError:
                pass
    shutil.rmtree(root)


class TemporaryDirectoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="memdiag-test-")
        self.temp_dir = Path(self._temporary.name)

    def tearDown(self) -> None:
        cleanup_tree(self.temp_dir)
        self._temporary.cleanup()
