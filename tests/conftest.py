from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ancient_vlayout.samples import generate_samples


@pytest.fixture(scope="session")
def generated_samples(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("generated") / "samples"
    generate_samples(root)
    return root


@pytest.fixture
def samples_copy(tmp_path: Path, generated_samples: Path) -> Path:
    target = tmp_path / "samples"
    shutil.copytree(generated_samples, target)
    return target
