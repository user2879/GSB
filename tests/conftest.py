from __future__ import annotations

import copy
from typing import Any

import pytest

from reco_shard_diag.fixtures import build_demo_config


@pytest.fixture
def demo_raw() -> dict[str, Any]:
    return build_demo_config(include_events=True)


@pytest.fixture
def demo_raw_no_events() -> dict[str, Any]:
    return build_demo_config(include_events=False)


@pytest.fixture
def raw_copy(demo_raw: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(demo_raw)
