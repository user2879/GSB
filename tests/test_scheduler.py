import threading
import time

from lease_middleware.scheduler import QuotaScheduler


def test_global_and_node_quotas_are_not_exceeded():
    entered_a = threading.Event()
    entered_b = threading.Event()
    release = threading.Event()
    active_lock = threading.Lock()
    active = {
        "global": 0,
        "node-a": 0,
        "node-b": 0,
        "max_global": 0,
        "max_a": 0,
    }

    def job(node: str) -> str:
        with active_lock:
            active["global"] += 1
            active[node] += 1
            active["max_global"] = max(active["max_global"], active["global"])
            active["max_a"] = max(active["max_a"], active["node-a"])
        if node == "node-a":
            entered_a.set()
        else:
            entered_b.set()
        assert release.wait(timeout=5)
        with active_lock:
            active["global"] -= 1
            active[node] -= 1
        return node

    with QuotaScheduler(
        global_in_flight=2, node_in_flight={"a": 1, "b": 2}
    ) as scheduler:
        first = scheduler.submit("r1", "a", lambda: job("node-a"))
        second = scheduler.submit("r2", "b", lambda: job("node-b"))
        assert entered_a.wait(timeout=2)
        assert entered_b.wait(timeout=2)
        deadline = time.time() + 1
        while time.time() < deadline and active["global"] < 2:
            time.sleep(0.01)
        assert active["global"] == 2
        third = scheduler.submit("r3", "a", lambda: job("node-a"))
        try:
            deadline = time.time() + 2
            snapshot = scheduler.active_snapshot()
            while time.time() < deadline and not (
                snapshot["global_active"] == 2
                and snapshot["node_active"]["a"] == 1
                and snapshot["queued"] >= 1
            ):
                time.sleep(0.01)
                snapshot = scheduler.active_snapshot()
            assert snapshot["global_active"] == 2
            assert snapshot["node_active"]["a"] == 1
            assert snapshot["queued"] >= 1
        finally:
            release.set()
        assert first.result(timeout=2) == "node-a"
        assert second.result(timeout=2) == "node-b"
        assert third.result(timeout=2) == "node-a"
        final = scheduler.active_snapshot()
        assert final["global_max_seen"] == 2
        assert final["node_max_seen"]["a"] == 1
        assert len(scheduler.wait_records) == 3
        for record in scheduler.wait_records:
            assert record["wait_ended_monotonic_ms"] >= record["wait_started_monotonic_ms"]


def test_node_quota_allows_other_nodes():
    enter_a = threading.Event()
    release = threading.Event()

    def slow_a():
        enter_a.set()
        assert release.wait(timeout=5)

    with QuotaScheduler(global_in_flight=3, node_in_flight={"a": 1, "b": 1}) as scheduler:
        future_a = scheduler.submit("a1", "a", slow_a)
        assert enter_a.wait(timeout=2)
        future_b1 = scheduler.submit("b1", "b", lambda: release.wait(timeout=5))
        future_b2 = scheduler.submit("b2", "b", lambda: "done")
        deadline = time.time() + 1
        while time.time() < deadline:
            snap = scheduler.active_snapshot()
            if snap["node_active"]["b"] == 1 and snap["queued"] >= 1:
                break
            time.sleep(0.01)
        assert scheduler.active_snapshot()["node_active"]["b"] <= 1
        release.set()
        assert future_a.result(timeout=2) is None
        assert future_b1.result(timeout=2) is True
        assert future_b2.result(timeout=2) == "done"
