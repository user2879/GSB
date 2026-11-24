from __future__ import annotations

from pathlib import Path

import pytest

from quant_pipeline.samples import generate_samples


@pytest.fixture()
def sample_dirs(tmp_path: Path) -> tuple[Path, Path]:
    sample_dir = tmp_path / "samples"
    config_dir = tmp_path / "configs"
    generate_samples(sample_dir, config_dir)
    return sample_dir, config_dir


@pytest.fixture()
def sample_dir(sample_dirs: tuple[Path, Path]) -> Path:
    return sample_dirs[0]


@pytest.fixture()
def config_dir(sample_dirs: tuple[Path, Path]) -> Path:
    return sample_dirs[1]

