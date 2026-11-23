from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from joinlab import load_workload
from joinlab.loader import rebuild_workload


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "demo_workload.json"
WHITESPACE_SAMPLE = ROOT / "samples" / "demo_workload_whitespace.json"


@pytest.fixture()
def workload():
    return load_workload(SAMPLE)


def with_tuning(workload, **updates):
    return rebuild_workload(workload.tables, workload.query, replace(workload.tuning, **updates))

