from __future__ import annotations

import shutil
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPO_ROOT / "samples"


@pytest.fixture()
def sample_dir(tmp_path: Path) -> Path:
    target = tmp_path / "samples"
    shutil.copytree(SAMPLE_ROOT, target)
    return target
