from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from acoustic_workbench.engine import IncrementalEngine


@pytest.fixture(scope="session")
def baseline_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("baseline-workspace")
    IncrementalEngine.open(root)
    return root


@pytest.fixture
def workspace(tmp_path: Path, baseline_workspace: Path) -> Path:
    target = tmp_path / "workspace"
    shutil.copytree(baseline_workspace, target)
    return target
