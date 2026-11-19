from __future__ import annotations

from pathlib import Path

import pytest

from spatial_diag.config import load_config
from spatial_diag.service import run_workbench


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def committed_workspace(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("committed-workspace")
    report = run_workbench(root, render_view=True)
    return Path(root), report

