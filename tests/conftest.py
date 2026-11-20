from __future__ import annotations

from pathlib import Path

import pytest

from imputation_workbench.engine import WorkspacePaths
from imputation_workbench.samplegen import generate_sample
from imputation_workbench.snapshot import SnapshotStore


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "sample"
    generate_sample(path, seed=20261121, rows=24)
    return path


def make_paths(workspace: Path, state_name: str = "work", injections: bool = True) -> WorkspacePaths:
    injection_path = workspace / "injections.json" if injections else workspace / "__none__.json"
    return WorkspacePaths(
        workspace=workspace,
        state=workspace / state_name,
        config=workspace / "imputation.yaml",
        data=workspace / "raw_data.jsonl",
        group_map=workspace / "group_map.json",
        plan=workspace / "edit_plan.json",
        injections=injection_path,
    )


def load_state(paths: WorkspacePaths):
    loaded = SnapshotStore(paths.state).load_state()
    assert loaded is not None
    return loaded

