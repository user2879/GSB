from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_samples import write_samples  # noqa: E402
from rsseg.runner import build_inputs, run_full_rebuild, run_incremental  # noqa: E402


@pytest.fixture(scope="session")
def sample_set(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("samples") / "sample_set"
    write_samples(path)
    return path


@pytest.fixture()
def work_dir(tmp_path) -> Path:
    path = tmp_path / "work"
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def second_work_dir(tmp_path) -> Path:
    return tmp_path / "work_full"
