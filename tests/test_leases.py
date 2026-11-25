from __future__ import annotations

import time
import unittest

from speaker_diag.errors import DiagnosticError
from speaker_diag.leases import LeaseScheduler


class LeaseTests(unittest.TestCase):
    def scheduler(self, *, ttl_ms: int = 200, global_limit: int = 1) -> LeaseScheduler:
        return LeaseScheduler(
            global_limit=global_limit,
            worker_lease_limits={"W01": 2, "W02": 2},
            worker_concurrency_limits={"W01": 1, "W02": 1},
            lease_ttl_ms={"W01": ttl_ms, "W02": ttl_ms},
            metric_scale=10000,
        )

    def test_real_monotonic_expiry_releases_quota(self) -> None:
        scheduler = self.scheduler(ttl_ms=30)
        lease = scheduler.acquire("B001", "W01")
        self.assertEqual(scheduler.snapshot()["global_active_leases"], 1)
        time.sleep(0.055)
        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot["global_active_leases"], 0)
        self.assertEqual(snapshot["workers"][0]["expired_leases"], 1)
        replacement = scheduler.acquire("B002", "W01")
        self.assertNotEqual(lease.lease_id, replacement.lease_id)
        scheduler.release(replacement.lease_id, expected_version=replacement.version)

    def test_renewal_requires_matching_optimistic_version(self) -> None:
        scheduler = self.scheduler()
        lease = scheduler.acquire("B001", "W01")
        with self.assertRaises(DiagnosticError) as caught:
            scheduler.renew(lease.lease_id, lease.version + 1)
        self.assertEqual(caught.exception.code, "E_LEASE")
        self.assertEqual(caught.exception.field, "version")
        renewed = scheduler.renew(lease.lease_id, lease.version)
        self.assertEqual(renewed.version, lease.version + 1)
        scheduler.release(lease.lease_id, expected_version=renewed.version)

    def test_parallel_queue_records_and_replay_are_consistent(self) -> None:
        scheduler = self.scheduler(ttl_ms=2000, global_limit=2)

        def action(batch_id: str, worker_id: str, lease: object, extra: int) -> int:
            time.sleep(0.02)
            return extra * 2

        tasks = [
            ("B001", "W01", 1),
            ("B002", "W02", 2),
            ("B003", "W01", 3),
            ("B004", "W02", 4),
        ]
        results = scheduler.apply_parallel(tasks, action, max_workers=2)
        self.assertEqual({key: value["value"] for key, value in results.items()}, {
            "B001": 2,
            "B002": 4,
            "B003": 6,
            "B004": 8,
        })
        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot["global_active_leases"], 0)
        self.assertTrue(any(record["queue_duration_ms"] >= 0 for record in snapshot["queue_records"]))
        replay = scheduler.replay()
        self.assertTrue(replay["replayable"])
        max_global = max(step["active_total"] for step in replay["steps"])
        self.assertLessEqual(max_global, 2)
        for step in replay["steps"]:
            self.assertLessEqual(step["worker_active_leases"].get("W01", 0), 2)
            self.assertLessEqual(step["worker_active_leases"].get("W02", 0), 2)


if __name__ == "__main__":
    unittest.main()
