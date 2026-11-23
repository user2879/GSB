from __future__ import annotations

import time

import pytest

from joinlab import LabError
from joinlab.scheduler import ConstrainedScheduler, TaskRunResult, TaskSpec


def runner(value: int, sleep: float = 0.01):
    def run(_dependencies):
        time.sleep(sleep)
        return TaskRunResult(value, value, value)

    return run


def test_per_table_quota_serializes_same_table_and_records_wait():
    scheduler = ConstrainedScheduler(
        global_concurrency=4,
        per_table_quota=1,
        memory_budget_bytes=1000,
    )
    tasks = [
        TaskSpec(f"t{index}", "scan", "scan:a", ("a",), 10, (), runner(index))
        for index in range(4)
    ]
    result = scheduler.run(tasks)
    assert result.max_global_running == 1
    assert result.max_per_table_running["a"] == 1
    assert result.constraints["table_quota_respected"] is True
    assert all(trace.wait_ms >= 0 for trace in result.traces)
    assert any(trace.wait_ms > 0 for trace in result.traces)


def test_global_concurrency_limit_is_respected():
    scheduler = ConstrainedScheduler(
        global_concurrency=2,
        per_table_quota=1,
        memory_budget_bytes=1000,
    )
    tasks = [
        TaskSpec(f"t{index}", "scan", f"scan:{index}", (f"table{index}",), 10, (), runner(index))
        for index in range(5)
    ]
    result = scheduler.run(tasks)
    assert result.max_global_running <= 2
    assert result.max_global_running == 2
    assert result.constraints["global_limit_respected"] is True


def test_memory_budget_prevents_simultaneous_tasks():
    scheduler = ConstrainedScheduler(
        global_concurrency=4,
        per_table_quota=1,
        memory_budget_bytes=100,
    )
    tasks = [
        TaskSpec(f"t{index}", "scan", f"scan:{index}", (f"table{index}",), 60, (), runner(index))
        for index in range(3)
    ]
    result = scheduler.run(tasks)
    assert result.max_memory_in_use_bytes <= 100
    assert result.max_global_running == 1
    assert result.constraints["memory_budget_respected"] is True


def test_dependency_order_and_task_peak_memory_are_reported():
    scheduler = ConstrainedScheduler(
        global_concurrency=2,
        per_table_quota=1,
        memory_budget_bytes=200,
    )
    tasks = [
        TaskSpec("left", "scan", "scan:left", ("left",), 20, (), runner(3)),
        TaskSpec("right", "scan", "scan:right", ("right",), 20, (), runner(4)),
        TaskSpec("join", "join", "join:0", ("left", "right"), 30, ("left", "right"), runner(7)),
    ]
    result = scheduler.run(tasks)
    assert result.values["join"] == 7
    traces = {trace.task_id: trace for trace in result.traces}
    assert traces["join"].start_ns >= traces["left"].end_ns
    assert traces["join"].start_ns >= traces["right"].end_ns
    assert traces["join"].completed_rows == 7
    assert traces["join"].peak_memory_bytes == 7


def test_impossible_single_task_reservation_is_rejected():
    scheduler = ConstrainedScheduler(
        global_concurrency=1,
        per_table_quota=1,
        memory_budget_bytes=10,
    )
    with pytest.raises(LabError) as caught:
        scheduler.run([TaskSpec("too-big", "scan", "scan:a", ("a",), 11, (), runner(1))])
    assert caught.value.code == "E_RESOURCE"

