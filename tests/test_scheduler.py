from __future__ import annotations

import threading
import time
from pathlib import Path

from cachelease.scheduler import QueuePermit, RequestScheduler
from cachelease.spec import load_config

from .helpers import two_node_config


def test_scheduler_enforces_global_and_per_node_limits_and_records_waits(tmp_path: Path) -> None:
    _, path = two_node_config(tmp_path)
    config = load_config(path)
    scheduler = RequestScheduler(config)
    release_gate = threading.Event()
    permits: list[QueuePermit] = []
    permits_lock = threading.Lock()
    active = 0
    active_lock = threading.Lock()
    two_acquired = threading.Event()

    def choose(node_in_flight: dict[str, int]):
        for node in config.nodes:
            if node_in_flight[node.node_id] < node.request_concurrency:
                chosen = node.node_id
                candidates = [
                    {
                        "node_id": chosen,
                        "artifact_key": "core",
                        "selected": True,
                        "eligible": True,
                        "reason": "selected",
                    }
                ]
                return chosen, candidates
        return None, []

    def worker() -> None:
        nonlocal active
        permit = scheduler.acquire("build", choose)
        with active_lock:
            active += 1
            if active == 2:
                two_acquired.set()
        release_gate.wait(timeout=2.0)
        with active_lock:
            active -= 1
        with permits_lock:
            permits.append(permit)
        scheduler.release(permit)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    assert two_acquired.wait(timeout=2.0)
    time.sleep(0.06)
    release_gate.set()
    for thread in threads:
        thread.join(timeout=3.0)
        assert not thread.is_alive()

    assert len(permits) == 4
    assert scheduler.max_global_observed == 2
    assert all(value <= 1 for value in scheduler.max_node_observed.values())
    assert any(permit.wait_ms >= 20 for permit in permits)
    for permit in permits:
        assert permit.wait_ms == permit.wait_end_ms - permit.wait_start_ms
        assert permit.global_in_flight <= 2
        assert permit.node_in_flight <= 1

