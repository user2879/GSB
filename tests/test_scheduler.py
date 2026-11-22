from __future__ import annotations

import threading
import time

from driftwatch.errors import DriftError
from driftwatch.models import CollectionTask, Observation
from driftwatch.scheduler import ConcurrencyScheduler

from .conftest import make_agent, make_criteria


def _observation(task: CollectionTask, value: int) -> Observation:
    return Observation(
        agent_id=task.agent.agent_id,
        observation_seq=task.observation_seq,
        host_id=task.host_id,
        metric_id=task.metric_id,
        value=value,
        aggregation_version=task.agent.aggregation.version,
        monotonic_ms=1,
    )


def test_global_and_per_agent_quota_with_wait_timestamps():
    agents = [make_agent("a1", 41001, hosts=("host-a",)), make_agent("a2", 41002, hosts=("host-a",))]
    tasks = [
        CollectionTask(agent, 1, "host-a", metric, 101)
        for metric in ("cpu_quota", "memory_pages", "processes")
        for agent in agents
    ]
    active = 0
    active_by_agent = {"a1": 0, "a2": 0}
    maximum = 0
    maximum_by_agent = {"a1": 0, "a2": 0}
    lock = threading.Lock()

    def collect(task: CollectionTask) -> Observation:
        nonlocal active, maximum
        with lock:
            active += 1
            active_by_agent[task.agent.agent_id] += 1
            maximum = max(maximum, active)
            maximum_by_agent[task.agent.agent_id] = max(
                maximum_by_agent[task.agent.agent_id], active_by_agent[task.agent.agent_id]
            )
        try:
            time.sleep(0.04)
            return _observation(task, 100)
        finally:
            with lock:
                active -= 1
                active_by_agent[task.agent.agent_id] -= 1

    criteria = make_criteria(global_concurrency=2, per_agent_concurrency=1)
    outcome = ConcurrencyScheduler(criteria).run(tasks, collect, batch_id="batch")
    assert maximum == 2
    assert maximum_by_agent == {"a1": 1, "a2": 1}
    assert outcome.max_observed_global == 2
    assert outcome.max_observed_per_agent == {"a1": 1, "a2": 1}
    assert len(outcome.attempts) == len(tasks)
    assert any(item.wait_end_ms > item.wait_start_ms for item in outcome.attempts)
    assert all(item.run_end_ms >= item.run_start_ms for item in outcome.attempts)


def test_retry_is_registered_separately_and_reuses_semantics():
    agent = make_agent("a1", 41011)
    task = CollectionTask(agent, 7, "host-a", "cpu_quota", 701)
    calls = 0
    lock = threading.Lock()

    def collect(work: CollectionTask) -> Observation:
        nonlocal calls
        with lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            raise DriftError.timeout("first attempt timed out", agent_id="a1")
        assert work is task
        return _observation(work, 123)

    outcome = ConcurrencyScheduler(make_criteria(max_retries=1, retry_backoff_ms=5)).run(
        [task], collect, batch_id="batch"
    )
    assert outcome.successes[task.task_key].value == 123
    assert len(outcome.attempts) == 2
    first, second = outcome.attempts
    assert first.attempt_no == 1 and not first.is_retry and first.status == "FAILED"
    assert first.error_code == "E_TIMEOUT"
    assert second.attempt_no == 2 and second.is_retry and second.status == "SUCCESS"
    assert first.observation_seq == second.observation_seq == 701
    assert second.wait_end_ms >= first.run_end_ms
