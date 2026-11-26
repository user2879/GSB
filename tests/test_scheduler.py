from __future__ import annotations

import threading
import time

from hopledger.scheduler import DispatchScheduler
from hopledger.specs import SchedulerConfig


def test_global_and_hop_quotas_are_never_exceeded():
    config = SchedulerConfig(max_in_flight=3, hop_quota=1, pool_workers=4)
    active_lock = threading.Lock()
    active_hop = {"a": 0, "b": 0}
    max_hop = {"a": 0, "b": 0}

    def invoke(request_id: str, node_id: str, prior):
        with active_lock:
            active_hop[node_id] += 1
            max_hop[node_id] = max(max_hop[node_id], active_hop[node_id])
        time.sleep(0.025)
        with active_lock:
            active_hop[node_id] -= 1
        return {"ok": True, "receipt": {"node_id": node_id, "request_id": request_id}}

    scheduler = DispatchScheduler(config, ["a", "b"], invoke)
    try:
        outcomes = scheduler.run_requests([f"req-{i}" for i in range(6)])
    finally:
        scheduler.shutdown()

    assert all(outcome.ok for outcome in outcomes)
    stats = scheduler.stats()
    assert stats["max_observed_global"] <= 3
    assert stats["max_observed_global"] == 3
    assert max_hop == {"a": 1, "b": 1}
    assert stats["max_observed_hop"] == {"a": 1, "b": 1}
    wait_events = [
        event
        for outcome in outcomes
        for event in outcome.events
        if event["scope"] == "hop" and event["node_id"] == "a"
    ]
    assert wait_events
    assert any(event["wait_ms"] >= 10 for event in wait_events)
    assert all(event["wait_end_monotonic_ms"] >= event["wait_start_monotonic_ms"]
               for event in wait_events)
