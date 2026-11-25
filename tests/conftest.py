from pathlib import Path

import pytest

from lease_middleware.spec import SpecBundle, load_sample_dir


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture()
def bundle() -> SpecBundle:
    return load_sample_dir(ROOT / "samples")
