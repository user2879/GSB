from __future__ import annotations

import json
from pathlib import Path

import pytest

from columnar_pipeline.model import RunConfig


ROOT = Path(__file__).resolve().parents[1]
SMALL_DATA = ROOT / "data" / "small.json"
SKEW_DATA = ROOT / "data" / "skew_amplified.json"
PARAMS = ROOT / "data" / "params"


@pytest.fixture
def small_data() -> Path:
    return SMALL_DATA


@pytest.fixture
def skew_data() -> Path:
    return SKEW_DATA


@pytest.fixture
def params() -> Path:
    return PARAMS


@pytest.fixture
def base_config() -> RunConfig:
    return RunConfig(
        parallelism=2,
        memory_budget_bytes=4096,
        budget_version=1,
        split_threshold=24,
        split_parts=2,
        row_group_size=16,
        ledger_batch_size=2,
        skew_safety_limit=100000,
    )


def load_params(path: Path) -> RunConfig:
    return RunConfig.from_json_dict(json.loads(path.read_text(encoding="utf-8")))
