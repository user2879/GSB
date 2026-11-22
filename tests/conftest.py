from __future__ import annotations

from pathlib import Path

import pytest

from iidx.config import load_config
from iidx.demo import generate_sample
from iidx.loader import discover_inputs


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    data = tmp_path / "sample"
    generate_sample(data, force=True)
    return data


@pytest.fixture
def sample_config(sample_dir: Path):
    paths = discover_inputs(sample_dir)
    return load_config(paths.config)
