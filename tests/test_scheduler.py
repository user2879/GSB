import unittest

from batch_receipt_ledger.clock import MonotonicClock
from batch_receipt_ledger.leases import LeaseManager
from batch_receipt_ledger.plan import build_plan
from batch_receipt_ledger.processes import WorkerPool
from batch_receipt_ledger.scheduler import Scheduler
from batch_receipt_ledger.spec import parse_spec

from tests._support import make_spec


class SchedulerTest(unittest.TestCase):
    def test_dual_limits_never_exceeded(self):
        raw = make_spec(workers=2, deliveries=12, batch_size=2,
                        lease_ms=2000, global_max=3)
        raw["workers"][0]["concurrency"] = 3
        raw["workers"][1]["concurrency"] = 1
        spec = parse_spec(raw)
        plans = build_plan(spec)
        delivery_index = {d.request_id: d for d in spec.deliveries}
        clock = MonotonicClock()
        lease_manager = LeaseManager(spec, clock)
        lease_manager.start_reaper(interval_ms=5)
        try:
            with WorkerPool(spec, raw) as pool:
                scheduler = Scheduler(spec, pool, lease_manager, clock)
                results = scheduler.run_plan(plans, delivery_index)
                audit = scheduler.audit_report()
        finally:
            lease_manager.stop_reaper()

        self.assertEqual(len(results), len(plans))
        self.assertLessEqual(audit["max_observed_active_leases"],
                             spec.global_max_active)
        self.assertLessEqual(audit["max_observed_inflight_batches"],
                             spec.global_max_active)
        self.assertTrue(any(item["wait_ms"] >= 0
                            for item in audit["wait_metrics"]))
        for item in audit["wait_metrics"]:
            self.assertIn("wait_start_ms", item)
            self.assertIn("wait_end_ms", item)
            self.assertEqual(item["wait_end_ms"] - item["wait_start_ms"],
                             item["wait_ms"])

    def test_results_are_deterministic_across_runs(self):
        raw = make_spec(workers=2, deliveries=6, batch_size=2)
        spec = parse_spec(raw)
        plans = build_plan(spec)
        delivery_index = {d.request_id: d for d in spec.deliveries}

        def once():
            clock = MonotonicClock()
            manager = LeaseManager(spec, clock)
            manager.start_reaper(interval_ms=5)
            try:
                with WorkerPool(spec, raw) as pool:
                    scheduler = Scheduler(spec, pool, manager, clock)
                    return scheduler.run_plan(plans, delivery_index)
            finally:
                manager.stop_reaper()

        first = once()
        second = once()
        self.assertEqual(
            [(r["batch_id"], r["results"]) for r in first],
            [(r["batch_id"], r["results"]) for r in second])


if __name__ == "__main__":
    unittest.main()
