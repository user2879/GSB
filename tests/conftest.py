from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from batch_diagnostic.spec import BudgetSpec, RequestSpec, SystemSpec, WorkerSpec  # noqa: E402


@pytest.fixture
def requests_spec():
    return tuple(
        RequestSpec(
            request_id=f"req-{index:03d}",
            arrival_seq=index,
            input_dim=4,
            seed=100 + index,
        )
        for index in range(9)
    )


@pytest.fixture
def workers_spec():
    return (
        WorkerSpec("w-a", "127.0.0.1", 6101, 7, 1, 4, 3, 2, 1, 2),
        WorkerSpec("w-b", "127.0.0.1", 6102, 8, 1, 4, 3, 3, 2, 2),
    )


@pytest.fixture
def budget_spec():
    return BudgetSpec(global_max_in_flight=2, retry_limit=2, cost_budget_ms=10000)


@pytest.fixture
def system_spec(requests_spec, workers_spec, budget_spec):
    return SystemSpec(requests=requests_spec, workers=workers_spec, budget=budget_spec)
