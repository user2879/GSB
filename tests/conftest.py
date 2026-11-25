from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gsb_index.canonical import write_json
from gsb_index.config import load_project
from gsb_index.models import Project
from gsb_index.sample import generate_sample


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("deterministic-sample")
    generate_sample(directory, seed=17, port_start=19300)
    return directory


@pytest.fixture(scope="session")
def sample_config(sample_dir: Path) -> Path:
    return sample_dir / "project.json"


@pytest.fixture(scope="session")
def sample_project(sample_config: Path) -> Project:
    project, issues = load_project(sample_config)
    assert issues == []
    return project


@pytest.fixture(scope="session")
def sample_edits(sample_dir: Path) -> list[dict[str, Any]]:
    return json.loads((sample_dir / "edits.json").read_text(encoding="utf-8"))["edits"]


@pytest.fixture
def write_config(tmp_path: Path):
    def _write(raw: dict[str, Any], name: str = "project.json") -> Path:
        path = tmp_path / name
        write_json(path, raw)
        return path

    return _write

