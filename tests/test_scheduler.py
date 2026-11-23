from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from quant_workbench.scheduler import (
    build_budget_plan,
    execute_batch,
    point_budget_pending_issues,
)


def test_budget_plan_uses_stable_lexicographic_order() -> None:
    plan = build_budget_plan({"z": 10, "a": 60, "b": 10}, 64)
    assert plan.batches == [["a"], ["b", "z"]]
    assert [entry["layer"] for entry in plan.ledger] == ["a", "b", "z"]
    assert [entry["remaining_after"] for entry in plan.ledger] == [4, 54, 44]


def test_oversized_layer_remains_pending_but_earlier_batch_is_kept() -> None:
    plan = build_budget_plan({"a": 10, "b": 100, "c": 20}, 64)
    assert plan.batches == [["a"]]
    assert plan.pending_layers == ["b", "c"]
    issues = point_budget_pending_issues(plan)
    assert [issue.code for issue in issues] == [
        "E_POINT_BUDGET_PENDING",
        "E_POINT_BUDGET_PENDING",
    ]


def test_execute_batch_respects_pool_limit_and_is_serial_equivalent() -> None:
    active = 0
    maximum = 0
    lock = threading.Lock()

    def worker(layer, deadline, cancel_event):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"layer": layer, "value": int(layer[1:]) * 2}

    layers = [f"l{index}" for index in range(6)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        concurrent = execute_batch(
            layers=layers,
            worker=worker,
            executor=executor,
            deadline=None,
        )
    assert maximum <= 3
    assert list(concurrent.completed) == sorted(layers)
    assert not concurrent.failures
    assert not concurrent.pending_layers

    with ThreadPoolExecutor(max_workers=1) as executor:
        serial = execute_batch(
            layers=layers,
            worker=worker,
            executor=executor,
            deadline=None,
        )
    assert concurrent.completed == serial.completed
