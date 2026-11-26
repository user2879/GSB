from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibstream.blocks import parse_block_stream
from vibstream.config import load_config
from vibstream.engine import MonitorEngine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_config():
    return load_config(ROOT / "configs" / "app.json")


@pytest.fixture(scope="session")
def sample_blocks(sample_config):
    data = json.loads((ROOT / "samples" / "data_blocks.json").read_text(encoding="utf-8"))
    return parse_block_stream(data, sample_config)


@pytest.fixture(scope="session")
def sample_commands():
    data = json.loads((ROOT / "samples" / "commands.json").read_text(encoding="utf-8"))
    return data["commands"]


@pytest.fixture
def engine(sample_config, sample_blocks):
    return MonitorEngine(sample_config, sample_blocks)
