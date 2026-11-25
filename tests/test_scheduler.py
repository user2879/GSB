from __future__ import annotations

import threading
import time

from lease_arbiter.scheduler import QuotaScheduler


def test_thread_pool_quota_limits_and_wait_metrics():
    scheduler = QuotaScheduler(
        global_limit=3,
        node_limits={"alpha": 2, "beta": 1},
        max_workers=6,
    )
    observed_threads: list[str] = []
    lock = threading.Lock()

    def task() -> None:
        with lock:
            observed_threads.append(threading.current_thread().name)
        time.sleep(0.06)

    futures = []
    try:
        for index in range(6):
            node_id = "alpha" if index < 4 else "beta"
            future, _task = scheduler.submit(node_id, f"r{index}", task)
            futures.append(future)
        for future in futures:
            future.result(timeout=5)
        snapshot = scheduler.snapshot()
    finally:
        scheduler.shutdown()

    assert snapshot["max_inflight_total"] == 3
    assert snapshot["max_inflight_node"]["alpha"] == 2
    assert snapshot["max_inflight_node"]["beta"] == 1
    assert snapshot["request_count"] == 6
    assert any(item["wait_ms"] > 0 for item in snapshot["requests"])
    assert all(item["wait_started_ms"] <= item["wait_ended_ms"] for item in snapshot["requests"])
    assert observed_threads
    assert all(name.startswith("lease-request") for name in observed_threads)
