import threading
import time
import unittest

from arcore.clock import MonotonicClock
from arcore.lease import LeaseManager
from arcore.model import Alert, Topology
from arcore.scheduler import CorrelationScheduler, build_batches

TOPO = Topology.build([("svc-a", "svc-b")])


def alert(alert_id, ts, service):
    return Alert(alert_id=alert_id, source_id="s", timestamp_ms=ts,
                 service_id=service, level="P1", fingerprint=f"fp-{alert_id}",
                 labels=(), status="OPEN", seq=0)


class SchedulerTest(unittest.TestCase):
    def test_batches_split_on_window_boundary(self):
        alerts = [
            alert("a1", 0, "svc-a"), alert("a2", 50, "svc-a"),
            alert("a3", 200, "svc-a"), alert("a4", 260, "svc-a"),
            alert("b1", 10, "svc-b"),
        ]
        batches = build_batches(alerts, window_ms=100)
        by_service = {}
        for batch in batches:
            by_service.setdefault(batch.service_id, []).append(batch)
        self.assertEqual(len(by_service["svc-a"]), 2)
        self.assertEqual(len(by_service["svc-b"]), 1)

    def test_real_concurrency_never_exceeds_caps(self):
        alerts = [
            alert(f"a{i}", i * 5, "svc-a" if i % 2 else "svc-b")
            for i in range(16)
        ]
        clock = MonotonicClock()
        manager = LeaseManager(
            global_capacity=3, service_capacity=2, ttl_ms=10_000,
            clock=clock)
        lock = threading.Lock()
        observed = {"global": 0, "peak": 0, "svc": 0, "svc_peak": 0}

        def hook(batch, lease):
            with lock:
                observed["global"] = manager.active_count()
                observed["peak"] = max(observed["peak"], observed["global"])
                observed["svc"] = max(
                    manager.service_active_count("svc-a"),
                    manager.service_active_count("svc-b"))
                observed["svc_peak"] = max(observed["svc_peak"],
                                           observed["svc"])
            time.sleep(0.01)

        scheduler = CorrelationScheduler(
            manager, TOPO, window_ms=100, global_capacity=3,
            service_capacity=2, clock=clock, work_hook=hook)
        result = scheduler.execute(
            build_batches(alerts, 100), alerts, use_approximate=False)
        self.assertLessEqual(observed["peak"], 3)
        self.assertLessEqual(observed["svc_peak"], 2)
        self.assertGreater(observed["peak"], 1)
        self.assertTrue(all(report.wait_ms >= 0 for report in result.reports))
        self.assertEqual(manager.active_count(), 0)


if __name__ == "__main__":
    unittest.main()
