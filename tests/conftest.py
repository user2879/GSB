from __future__ import annotations

from pathlib import Path

import pytest

from mesh_repair.pipeline import RunResult, run_fixed_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def default_run(tmp_path_factory: pytest.TempPathFactory) -> RunResult:
    output = tmp_path_factory.mktemp("default_run")
    return run_fixed_pipeline(PROJECT_ROOT / "configs" / "default.json", output)
