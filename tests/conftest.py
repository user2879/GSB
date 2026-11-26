from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmfm.workspace import Workspace, load_workspace  # noqa: E402


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def workspace() -> Workspace:
    return load_workspace(ROOT / "configs" / "render.json", ROOT / "samples" / "conditions.json")
