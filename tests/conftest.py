import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depedit.samplegen import generate_samples  # noqa: E402


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("sample")
    generate_samples(path, seed=20241120, n_sentences=8)
    return path
