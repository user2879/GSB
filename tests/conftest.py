import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from generate_samples import generate  # noqa: E402
from workbench.cli import load_dataset  # noqa: E402


@pytest.fixture(scope="session")
def samples_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("samples") / "samples"
    generate(str(out))
    return str(out)


@pytest.fixture()
def dataset(samples_dir):
    return load_dataset(samples_dir)
