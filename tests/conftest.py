import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from rerank_core.config import load_config
from rerank_core.loader import load_candidates

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "sample_input"


@pytest.fixture(scope="session")
def config():
    return load_config(SAMPLE / "rerank_config.json", SAMPLE / "feature_groups.json")


@pytest.fixture(scope="session")
def loaded(config):
    return load_candidates(SAMPLE / "candidates.jsonl", config)
