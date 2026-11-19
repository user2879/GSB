from __future__ import annotations

from pathlib import Path

import pytest

from layout_workbench.config import RecognitionConfig
from layout_workbench.samplegen import generate_samples


@pytest.fixture(scope="session")
def sample_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("layout-samples")
    generate_samples(root)
    return root


@pytest.fixture()
def config() -> RecognitionConfig:
    return RecognitionConfig()

