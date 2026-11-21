import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_monitor.config import load_config


@pytest.fixture()
def config_path():
    return ROOT / "configs" / "default_monitor.yaml"


@pytest.fixture()
def config(config_path):
    return load_config(config_path)
