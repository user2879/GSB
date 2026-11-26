import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

SAMPLE_CONFIG = os.path.join(ROOT, "samples", "config")


@pytest.fixture()
def tmp_workdir():
    with tempfile.TemporaryDirectory(prefix="edge-cache-test-") as path:
        yield path


@pytest.fixture()
def sample_config_dir():
    return SAMPLE_CONFIG
