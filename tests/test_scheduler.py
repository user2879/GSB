from __future__ import annotations

import copy

from modeldash.scheduler import EXHAUSTED, Scheduler
from modeldash.spec import validate_spec
from modeldash.storage import Ledger

from conftest import FakeDispatcher, free_port


def run_scheduler(tmp_path, raw, dispatcher):
    spec = validate_spec(raw)
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    activation = ledger.activate_spec(spec)
    try:
        summary = Scheduler(spec, ledger, activation.session_id, dispatcher).run()
        return summary, ledger, activation.session_id
    finally:
        ledger.close()


def minimal_raw(*, nodes=1, quota=1, global_concurrency=1, shards=2, tasks=None, backoff=None, max_retries=None):
    ports = [free_port() for _ in range(nodes)]
    raw = {
        "format_version": 1,
        "spec_version": 1,
        "seed": 3,
        "input": {"size": 4, "scale": 7},
        "nodes": [
            {
                "id": f"n{index + 1}",
                "host": "127.0.0.1",
                "port": ports[index],
                "quota": quota,
                "shard_delay_ms": 0,
            }
            for index in range(nodes)
        ],
        "configs": [
            {
                "version": 1,
                "priority": 0,
                "target_nodes": [f"n{index + 1}" for index in range(nodes)],
                "shards": shards,
                "items": {"alpha": 1, "beta": 2, "gamma": 3},
            }
        ],
        "budget": {
            "global_concurrency": global_concurrency,
            "timeout_ms": 1000,
            "retry_backoff_ms": backoff or [1],
            "max_retries": len(backoff or [1]) if max_retries is None else max_retries,
        },
        "tasks": tasks
        or [{"id": "t1", "config_version": 1, "node": "n1", "priority": 0, "release_after_ms": 0}],
    }
    return raw


def test_global_and_per_node_quotas_are_never_exceeded(tmp_path):
    raw = minimal_raw(
        nodes=2,
        quota=1,
        global_concurrency=2,
        tasks=[
            {"id": "a", "config_version": 1, "node": "n1", "priority": 4},
            {"id": "b", "config_version": 1, "node": "n2", "priority": 3},
            {"id": "c", "config_version": 1, "node": "n1", "priority": 2},
            {"id": "d", "config_version": 1, "node": "n2", "priority": 1},
        ],
    )
    dispatcher = FakeDispatcher(delay_ms=15)
    summary, _, _ = run_scheduler(tmp_path, raw, dispatcher)
    assert dispatcher.max_active <= 2
    assert dispatcher.max_active_by_node["n1"] <= 1
    assert dispatcher.max_active_by_node["n2"] <= 1
    assert {task["task_id"]: task["state"] for task in summary.tasks} == {
        "a": "DONE",
        "b": "DONE",
        "c": "DONE",
        "d": "DONE",
    }
    assert any(task["wait_ms"] > 0 for task in summary.tasks)


def test_high_priority_preempts_at_shard_boundary_and_resumes(tmp_path):
    raw = minimal_raw(
        shards=4,
        tasks=[
            {"id": "low", "config_version": 1, "node": "n1", "priority": 1},
            {"id": "high", "config_version": 1, "node": "n1", "priority": 9, "release_after_ms": 35},
        ],
    )
    dispatcher = FakeDispatcher(delay_ms=20)
    summary, _, session_id = run_scheduler(tmp_path, raw, dispatcher)
    by_id = {task["task_id"]: task for task in summary.tasks}
    assert by_id["low"]["state"] == "DONE"
    assert by_id["high"]["state"] == "DONE"
    assert len(by_id["low"]["completed_shards"]) == 4
    assert len({item["shard_index"] for item in by_id["low"]["completed_shards"]}) == 4
    suspended = [item for item in summary.preemptions if item["task_id"] == "low"]
    assert suspended and suspended[0]["status"] == "SUSPENDED"
    assert 0 < suspended[0]["suspended_shard"] < 4
    assert summary.decisions[0]["verdict"] == "PREEMPTABLE"
    assert by_id["low"]["preempt_count"] >= 1


def test_retry_backoff_releases_quota_and_preserves_request(tmp_path):
    raw = minimal_raw(
        shards=2,
        backoff=[10],
        max_retries=1,
        tasks=[
            {"id": "flaky", "config_version": 1, "node": "n1", "priority": 2},
            {"id": "other", "config_version": 1, "node": "n1", "priority": 1},
        ],
    )
    dispatcher = FakeDispatcher(delay_ms=5, fail_shards={"flaky": 1})
    summary, ledger, session_id = run_scheduler(tmp_path, raw, dispatcher)
    by_id = {task["task_id"]: task for task in summary.tasks}
    assert by_id["flaky"]["state"] == "DONE"
    assert by_id["flaky"]["retries_total"] == 1
    assert by_id["other"]["state"] == "DONE"
    assert dispatcher.attempts[("flaky", 0)] == 2
    assert dispatcher.max_active <= 1


def test_retry_limit_marks_e_exhaust(tmp_path):
    raw = minimal_raw(shards=1, backoff=[1], max_retries=1)
    dispatcher = FakeDispatcher(delay_ms=1, always_fail=True)
    summary, _, _ = run_scheduler(tmp_path, raw, dispatcher)
    task = summary.tasks[0]
    assert task["state"] == EXHAUSTED
    assert task["error_code"] == "E_EXHAUST"
    assert task["attempts_total"] == 2


def test_high_priority_logs_backoff_wait_exemption(tmp_path):
    raw = minimal_raw(
        nodes=1,
        quota=2,
        global_concurrency=2,
        shards=4,
        backoff=[500],
        max_retries=1,
        tasks=[
            {"id": "back", "config_version": 1, "node": "n1", "priority": 2},
            {"id": "hold1", "config_version": 1, "node": "n1", "priority": 1},
            {"id": "hold2", "config_version": 1, "node": "n1", "priority": 1},
            {"id": "high", "config_version": 1, "node": "n1", "priority": 9, "release_after_ms": 180},
        ],
    )
    dispatcher = FakeDispatcher(delay_ms=80, fail_shards={"back": 1})
    summary, _, _ = run_scheduler(tmp_path, raw, dispatcher)
    decisions = [
        item
        for item in summary.decisions
        if item["high_task_id"] == "high" and item["victim_task_id"] == "back"
    ]
    assert decisions
    assert decisions[0]["verdict"] == "WAIT_EXEMPT"
    assert decisions[0]["reason"] == "BACKOFF_DOES_NOT_HOLD_QUOTA"
    assert decisions[0]["preemptable"] is False
