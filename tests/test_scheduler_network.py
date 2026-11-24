from __future__ import annotations

import threading
import time

import pytest

from batch_diagnostic.errors import TimeoutError_
from batch_diagnostic.network import (
    WorkerClient,
    WorkerService,
    check_port_bindable,
)
from batch_diagnostic.scheduler import Scheduler


def _echo(payload):
    return {"echo": payload["batch_id"], "batch_index": payload["batch_index"]}


def test_real_loopback_round_trip():
    service = WorkerService("127.0.0.1", 6211, _echo)
    service.start()
    try:
        client = WorkerClient("127.0.0.1", 6211, timeout_ms=2000)
        response, attempts = client.round_trip_with_retry(
            {"batch_id": "b-1", "batch_index": 0}, retries=1
        )
        assert response == {"echo": "b-1", "batch_index": 0}
        assert attempts == 1
    finally:
        service.stop()


def test_timeout_is_real_and_retried():
    state = {"calls": 0}

    def slow(payload):
        state["calls"] += 1
        time.sleep(0.3)
        return {"batch_index": payload["batch_index"]}

    service = WorkerService("127.0.0.1", 6212, slow)
    service.start()
    try:
        client = WorkerClient("127.0.0.1", 6212, timeout_ms=50)
        with pytest.raises(TimeoutError_) as exc:
            client.round_trip_with_retry(
                {"batch_id": "b", "batch_index": 0}, retries=3
            )
        assert exc.value.code == "E_TIMEOUT"
        assert state["calls"] == 3
    finally:
        service.stop()


def test_port_conflict_detected():
    service = WorkerService("127.0.0.1", 6213, _echo)
    service.start()
    try:
        with pytest.raises(OSError):
            check_port_bindable("127.0.0.1", 6213)
    finally:
        service.stop()


def test_scheduler_respects_global_and_worker_caps(system_spec):
    jobs = [
        {
            "batch_id": f"b-{index}",
            "batch_index": index,
            "worker_id": worker_id,
        }
        for index, worker_id in enumerate(["w-a", "w-b", "w-b", "w-a", "w-b"])
    ]
    lock = threading.Lock()
    active_global = 0
    peak_global = 0
    active_worker: dict[str, int] = {}
    peak_worker: dict[str, int] = {}

    def factory(host, port):
        class FakeClient:
            def round_trip_with_retry(self, payload, retries):
                nonlocal active_global, peak_global
                with lock:
                    active_global += 1
                    peak_global = max(peak_global, active_global)
                    worker_id = payload["worker_id"]
                    active_worker[worker_id] = active_worker.get(worker_id, 0) + 1
                    peak_worker[worker_id] = max(
                        peak_worker.get(worker_id, 0), active_worker[worker_id]
                    )
                time.sleep(0.05)
                with lock:
                    active_global -= 1
                    active_worker[payload["worker_id"]] -= 1
                return (
                    {"batch_id": payload["batch_id"], "batch_index": payload["batch_index"]},
                    1,
                )

        return FakeClient()

    scheduler = Scheduler(system_spec, factory)
    results = scheduler.run(jobs)
    assert peak_global <= system_spec.budget.global_max_in_flight
    assert peak_worker["w-a"] <= 1
    assert peak_worker["w-b"] <= 2
    assert sorted(result.batch_id for result in results) == sorted(
        job["batch_id"] for job in jobs
    )
    assert all(result.timing.wait_ended_ms >= result.timing.wait_started_ms for result in results)
