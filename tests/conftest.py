from __future__ import annotations

import json
from pathlib import Path

import pytest

from slowprof.models import Interval, ProfilePolicy

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture
def samples_dir() -> Path:
    return SAMPLES


@pytest.fixture
def policy() -> ProfilePolicy:
    return ProfilePolicy()


@pytest.fixture
def small_interval() -> Interval:
    return Interval(1_700_000_000_000, 1_700_000_020_000, "obs-v1")


@pytest.fixture
def changed_interval() -> Interval:
    return Interval(1_700_000_004_000, 1_700_000_020_000, "obs-narrow-v2")


@pytest.fixture
def large_interval() -> Interval:
    return Interval(1_710_000_000_000, 1_710_001_000_000, "obs-v1")


def write_ndjson(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path

