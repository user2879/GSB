"""Shared fixtures and path setup for the test suite."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dialog_state_pipeline.config import load_config, load_fsm
from dialog_state_pipeline.sample_data import write_sample_dir


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("samples")
    write_sample_dir(path)
    return path


@pytest.fixture
def fsm(sample_dir):
    return load_fsm(sample_dir / "fsm.yaml")


@pytest.fixture
def config(sample_dir, tmp_path):
    cfg = load_config(sample_dir / "pipeline_config.json")
    cfg.snapshot_dir = str(tmp_path / "snapshot")
    return cfg

