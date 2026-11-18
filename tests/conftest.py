"""Shared test fixtures: project path and deterministic sample inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_app.generator import generate  # noqa: E402
from rag_app.replay import load_inputs  # noqa: E402
from rag_app.snapshot import SnapshotStore  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture()
def sample_dir(tmp_path: Path) -> Path:
    target = tmp_path / "sample"
    generate(target)
    return target


@pytest.fixture()
def snapshot_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "snapshots"
    SnapshotStore(directory).ensure_dirs()
    return directory


@pytest.fixture()
def loaded(sample_dir: Path):
    return load_inputs(sample_dir)
