from __future__ import annotations

import json
from pathlib import Path

import pytest

from remix_workbench.config import load_config
from remix_workbench.generate import generate_workspace


@pytest.fixture(scope="session")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("remix_workspace")
    generate_workspace(root, seed=20261118)
    return root


@pytest.fixture(scope="session")
def samples_root(workspace: Path) -> Path:
    return workspace / "samples"


@pytest.fixture()
def config(workspace: Path) -> dict:
    return load_config(workspace / "configs" / "remix.json")


@pytest.fixture()
def injections_path(workspace: Path) -> Path:
    return workspace / "configs" / "fault_injections.json"


@pytest.fixture()
def edits_path(workspace: Path) -> Path:
    return workspace / "configs" / "edit_sequence.json"


@pytest.fixture(scope="session")
def sample_manifest(samples_root: Path) -> dict:
    return json.loads((samples_root / "manifest.json").read_text(encoding="utf-8"))

