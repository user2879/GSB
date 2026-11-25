from __future__ import annotations

import copy
import threading
import time

import pytest

from vscan.errors import BudgetError
from vscan.executor import execute_batch_for_query
from vscan.planner import compile_plan
from vscan.scheduler import SchedulerError, SchedulerTask, run_batches
from vscan.spec import parse_spec
from vscan.store import build_store


def _scheduler_fixture(tmp_path, spec_dict, *, budget: int = 300, concurrency: int = 3):
    data = copy.deepcopy(spec_dict)
    data["segments"][0]["rows"] = 40
    data["execution"]["batch_rows"] = 5
    data["execution"]["segment_decode_memory_bytes"] = budget
    data["execution"]["max_concurrency"] = concurrency
    spec = parse_spec(data)
    store = build_store(spec, tmp_path / "data")
    plan = compile_plan(spec)
    query_plan = plan.queries[0]
    tasks = [
        SchedulerTask(plan=batch, query=query_plan.query, fingerprint=f"fp-{batch.batch_no}")
        for batch in query_plan.batches
    ]
    return spec, store, tasks


def test_thread_pool_concurrency_and_memory_admission(tmp_path, spec_dict):
    spec, store, tasks = _scheduler_fixture(tmp_path, spec_dict)
    lock = threading.Lock()
    active = 0
    max_active = 0
    committed = []

    def slow_executor(task):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        computation = execute_batch_for_query(spec, store, tmp_path / "data", task.query, task.plan)
        with lock:
            active -= 1
        return computation

    results, stats = run_batches(
        spec,
        store,
        tmp_path / "data",
        tasks,
        on_commit=committed.append,
        executor_fn=slow_executor,
    )
    assert len(results) == 8
    assert len(committed) == 8
    assert stats.dispatched_count == 8
    assert stats.max_observed_inflight <= 3
    assert stats.max_observed_inflight <= 2  # each batch admits 135 bytes; 2 * 135 <= 300
    assert stats.max_observed_decoded_bytes <= 300
    assert max_active <= stats.max_observed_inflight
    assert all(result.worker_thread.startswith("vscan-batch") for result in results)
    assert all(result.wait_end_ms >= result.wait_start_ms for result in results)
    assert all(result.wait_ms == result.wait_end_ms - result.wait_start_ms for result in results)
    assert all(result.dispatch_count == 1 for result in results)


def test_single_batch_larger_than_budget_fails_cleanly(tmp_path, spec_dict):
    spec, store, tasks = _scheduler_fixture(tmp_path, spec_dict, budget=10)
    with pytest.raises(BudgetError) as caught:
        run_batches(spec, store, tmp_path / "data", tasks)
    assert caught.value.code == "E_BUDGET"
    assert caught.value.detail["budget_bytes"] == 10


def test_duplicate_batch_task_rejected(tmp_path, spec_dict):
    spec, store, tasks = _scheduler_fixture(tmp_path, spec_dict)
    with pytest.raises(SchedulerError) as caught:
        run_batches(spec, store, tmp_path / "data", [tasks[0], tasks[0]])
    assert caught.value.code == "E_SCHEDULE"


def test_unready_batch_is_not_dispatched(tmp_path, spec_dict):
    spec, store, tasks = _scheduler_fixture(tmp_path, spec_dict)
    blocked = SchedulerTask(
        plan=tasks[0].plan,
        query=tasks[0].query,
        fingerprint=tasks[0].fingerprint,
        ready=False,
    )
    with pytest.raises(SchedulerError) as caught:
        run_batches(spec, store, tmp_path / "data", [blocked])
    assert caught.value.detail["ready"] is False


def test_dependency_is_completed_before_dependent_batch_is_dispatched(tmp_path, spec_dict):
    spec, store, tasks = _scheduler_fixture(tmp_path, spec_dict)
    first, second = tasks[0], tasks[1]
    second = SchedulerTask(
        plan=second.plan,
        query=second.query,
        fingerprint=second.fingerprint,
        dependencies=(first.key,),
    )
    order = []

    def ordered_executor(task):
        order.append(task.key)
        if task.key == second.key:
            assert first.key in order
        return execute_batch_for_query(
            spec, store, tmp_path / "data", task.query, task.plan
        )

    results, _ = run_batches(
        spec,
        store,
        tmp_path / "data",
        [second, first],
        executor_fn=ordered_executor,
    )
    assert {result.key for result in results} == {first.key, second.key}
    assert order.index(first.key) < order.index(second.key)
