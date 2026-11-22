import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from mlimgedit.config import load_config_dict
from mlimgedit.datagen import generate_dataset
from mlimgedit.snapshot import SnapshotStore
from scripts.bootstrap_demo import build_config, build_injections


@pytest.fixture()
def config_dict():
    return build_config(threads=3, budget=50000)


@pytest.fixture()
def config(config_dict):
    return load_config_dict(config_dict)


@pytest.fixture()
def sample_dir(tmp_path, config):
    directory = tmp_path / "samples"
    generate_dataset(directory, config, 10)
    return directory


@pytest.fixture()
def injections_path(tmp_path):
    path = tmp_path / "injections.json"
    path.write_text(__import__("json").dumps(build_injections()), encoding="utf-8")
    return path


@pytest.fixture()
def store(tmp_path):
    return SnapshotStore(tmp_path / "store")
