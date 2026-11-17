from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from tunnelvpn.budget import BudgetManager, ManualClock

from tests.helpers import config


class BudgetTests(unittest.TestCase):
    def test_manual_clock_queue_and_dual_token_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg, _ = config(
                Path(directory),
                peer_behaviors=("normal",),
                route_bandwidth_kbps=1,
                global_bandwidth_kbps=2,
            )
            clock = ManualClock(0)
            manager = BudgetManager(
                cfg.global_limits,
                cfg.routes,
                clock_ns=clock.now_ns,
                sleep_fn=clock.sleep,
            )
            events: list[tuple[str, int]] = []
            for _ in range(3):
                manager.acquire_bandwidth(
                    "alpha", 1000, observer=lambda event, ms: events.append((event, ms))
                )
            self.assertEqual(events[0][0], "start")
            self.assertEqual(events[-1][0], "end")
            self.assertGreaterEqual(clock.now_ns(), 1_800_000_000)
            state = manager.snapshot().to_dict()
            self.assertEqual(state["routes"]["alpha"]["used_bytes"], 3000)
            self.assertEqual(state["global"]["used_bytes"], 3000)
            self.assertLessEqual(state["global"]["used_bytes"], 6000)

    def test_global_budget_serializes_independent_tunnels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg, _ = config(
                Path(directory),
                peer_behaviors=("normal", "digest-error"),
                route_bandwidth_kbps=2,
                global_bandwidth_kbps=1,
                global_max_sessions=4,
            )
            clock = ManualClock(0)
            manager = BudgetManager(
                cfg.global_limits,
                cfg.routes,
                clock_ns=clock.now_ns,
                sleep_fn=clock.sleep,
            )
            manager.acquire_bandwidth("alpha", 1024)
            manager.acquire_bandwidth("corrupt", 1024)
            self.assertGreaterEqual(clock.now_ns(), 1_000_000_000)
            snapshot = manager.snapshot()
            self.assertEqual(snapshot.global_used_bytes, 2048)

    def test_concurrency_never_exceeds_global_or_route_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg, _ = config(
                Path(directory),
                peer_behaviors=("normal", "digest-error"),
                route_max_sessions=1,
                global_max_sessions=2,
            )
            manager = BudgetManager(cfg.global_limits, cfg.routes)
            max_global = 0
            max_route = {"alpha": 0, "corrupt": 0}
            observed = [0]
            lock = threading.Lock()

            def worker(route_id: str) -> None:
                nonlocal max_global
                manager.acquire_session(route_id)
                try:
                    snap = manager.snapshot()
                    with lock:
                        observed[0] += 1
                        max_global = max(max_global, snap.global_active)
                        max_route[route_id] = max(
                            max_route[route_id],
                            snap.routes[route_id]["active_sessions"],
                        )
                    time.sleep(0.03)
                finally:
                    manager.release_session(route_id)

            threads = [
                threading.Thread(target=worker, args=(route_id,))
                for route_id in ("alpha", "alpha", "corrupt", "corrupt", "alpha", "corrupt")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertLessEqual(max_global, 2)
            self.assertLessEqual(max_route["alpha"], 1)
            self.assertLessEqual(max_route["corrupt"], 1)
            self.assertEqual(observed[0], 6)
            snapshot = manager.snapshot()
            self.assertEqual(snapshot.global_active, 0)
            self.assertEqual(snapshot.routes["alpha"]["active_sessions"], 0)

    def test_request_above_bucket_capacity_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg, _ = config(
                Path(directory),
                peer_behaviors=("normal",),
                route_bandwidth_kbps=1,
                global_bandwidth_kbps=2,
            )
            manager = BudgetManager(cfg.global_limits, cfg.routes)
            self.assertEqual(manager.max_bandwidth_chunk("alpha"), 1024)
            with self.assertRaises(ValueError):
                manager.acquire_bandwidth("alpha", 1025)


if __name__ == "__main__":
    unittest.main()
