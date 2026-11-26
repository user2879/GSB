import os
import tempfile
import time
import unittest

from leasebatch.engine import RunEngine
from leasebatch.lease import LeaseManager
from leasebatch.plan import build_plan
from leasebatch.spec import load_spec_dir
from leasebatch.worker import WorkerPool


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec_dir(
            os.path.join(os.path.dirname(__file__), "..", "samples")
        )
        self.plan = build_plan(self.spec, 2, 42)

    def test_real_worker_subprocesses_return_mixed_formats(self):
        batch_ids = [b.batch_id for b in self.plan.batches]
        formats = {
            batch_ids[0]: "r1",
            batch_ids[1]: "r2",
            batch_ids[2]: "r3",
        }
        versions = {batch_ids[1]: 1, batch_ids[2]: 1}
        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "ledger.db")
            engine = RunEngine(self.spec, db_path)
            out = engine.execute(
                2, 42, version_map=versions, format_map=formats
            )
        source_formats = {
            item["normalized"]["source_format"]
            for item in out["run"]["results"]
        }
        self.assertEqual(source_formats, {"r1", "r2", "r3"})
        # r1 receipt must have the full filled-field provenance.
        r1_result = out["run"]["results"][0]["normalized"]
        self.assertIn("param_version", r1_result["filled_fields"])
        # Every batch is bound to a real lease id.
        self.assertTrue(
            all(item["lease_id"].startswith("L")
                for item in out["run"]["results"])
        )

    def test_never_exceeds_concurrency_ceilings(self):
        from leasebatch.scheduler import Scheduler

        observed = {"max_active": 0}
        manager = LeaseManager(self.spec)
        original_grant = manager.grant

        def tracking_grant(worker_id, batch_id):
            lease = original_grant(worker_id, batch_id)
            observed["max_active"] = max(
                observed["max_active"], manager.active_count()
            )
            self.assertLessEqual(
                manager.active_count(),
                self.spec.lease_policy.global_max_active,
            )
            return lease

        manager.grant = tracking_grant
        with tempfile.TemporaryDirectory() as directory:
            with WorkerPool(self.spec, directory) as pool:
                scheduler = Scheduler(self.spec, self.plan, pool)
                result = scheduler.run(manager)
        self.assertEqual(len(result["results"]), len(self.plan.batches))
        self.assertLessEqual(observed["max_active"],
                             self.spec.lease_policy.global_max_active)

    def test_wait_time_uses_monotonic_clock(self):
        from leasebatch.scheduler import Scheduler

        manager = LeaseManager(self.spec)
        with tempfile.TemporaryDirectory() as directory:
            with WorkerPool(self.spec, directory) as pool:
                scheduler = Scheduler(self.spec, self.plan, pool)
                result = scheduler.run(manager)
        for wait in result["waits"]:
            self.assertGreaterEqual(wait["wait_end_ms"],
                                    wait["wait_start_ms"])
            self.assertEqual(
                wait["wait_ms"],
                wait["wait_end_ms"] - wait["wait_start_ms"],
            )

    def test_real_monotonic_expiry_releases_slot(self):
        manager = LeaseManager(self.spec)
        lease = manager.grant("w1", "b1")
        time.sleep(
            self.spec.lease_policy.lease_duration_ms / 1000.0 + 0.05
        )
        reclaimed = manager.sweep_expired()
        self.assertEqual(reclaimed, [lease.lease_id])
        self.assertEqual(manager.active_count(), 0)


if __name__ == "__main__":
    unittest.main()
