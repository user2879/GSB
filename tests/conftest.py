from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from candidate_diagnostics.config import load_config
from candidate_diagnostics.loader import load_requests_jsonl


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture()
def config():
    return load_config(ROOT / "samples" / "config.json")


@pytest.fixture()
def requests(config):
    requests, failures = load_requests_jsonl(ROOT / "samples" / "candidates.jsonl", config)
    assert failures == []
    return requests
