import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from samplegen.generate_samples import generate  # noqa: E402


@pytest.fixture(scope="session")
def samples_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("data")
    source = base / "samples"
    generate(str(source))
    return str(source)


@pytest.fixture()
def work_dir(tmp_path):
    return str(tmp_path / "work")
