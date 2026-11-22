import unittest

from workbench.allocator import (
    TRIM_BUDGET,
    TRIM_CAPACITY_OVERFLOW,
    TRIM_QUOTA,
    allocate_minute,
)
from workbench.estimator import FullEstimator
from workbench.testing_utils import spec_from_payload
from workbench.sample_data import build_cluster


class AllocatorTest(unittest.TestCase):
    def setUp(self):
        self.spec = spec_from_payload(build_cluster())
        self.rows = FullEstimator().run(self.spec)

    def test_budget_never_exceeded(self):
        for minute, estimates in enumerate(self.rows):
            allocations = allocate_minute(self.spec, minute, estimates)
            self.assertLessEqual(
                sum(item.allocated for item in allocations),
                self.spec.global_replica_budget,
            )

    def test_quota_and_max_caps_enforced(self):
        for minute, estimates in enumerate(self.rows):
            allocations = allocate_minute(self.spec, minute, estimates)
            for item in allocations:
                service = next(s for s in self.spec.services
                               if s.service_id == item.service_id)
                self.assertLessEqual(item.allocated, service.concurrency_quota)
                self.assertLessEqual(item.allocated, service.max_replicas)

    def test_rejection_non_negative_and_based_on_capacity(self):
        for minute, estimates in enumerate(self.rows):
            allocations = allocate_minute(self.spec, minute, estimates)
            estimate_map = {e.service_id: e for e in estimates}
            for item in allocations:
                self.assertGreaterEqual(item.rejected, 0)
                estimate = estimate_map[item.service_id]
                expected = max(
                    0, estimate.requests - item.capacity_requests
                )
                self.assertEqual(item.rejected, expected)

    def test_stable_ordering_budget_pressure(self):
        payload = build_cluster()
        payload["global_replica_budget"] = 3
        spec = spec_from_payload(payload)
        rows = FullEstimator().run(spec)
        first = allocate_minute(spec, 5, rows[5])
        second = allocate_minute(spec, 5, rows[5])
        self.assertEqual([a.as_tuple() for a in first],
                         [a.as_tuple() for a in second])
        reasons = {a.service_id: a.trim_reason for a in first}
        self.assertIn(TRIM_BUDGET, reasons.values())

    def test_quota_trim_reason(self):
        payload = build_cluster()
        payload["services"][0]["concurrency_quota"] = 1
        spec = spec_from_payload(payload)
        rows = FullEstimator().run(spec)
        allocations = allocate_minute(spec, 10, rows[10])
        embedding = next(a for a in allocations if a.service_id == "svc-embedding")
        self.assertIn(embedding.trim_reason, (TRIM_QUOTA, TRIM_BUDGET))
        self.assertEqual(embedding.effective_cap, 1)

    def test_overflow_trim_reason(self):
        payload = build_cluster()
        payload["services"][1]["max_replicas"] = 1
        spec = spec_from_payload(payload)
        rows = FullEstimator().run(spec)
        minute = spec.minutes - 1
        allocations = allocate_minute(spec, minute, rows[minute])
        ranker = next(a for a in allocations if a.service_id == "svc-ranker")
        self.assertEqual(ranker.trim_reason, TRIM_CAPACITY_OVERFLOW)


if __name__ == "__main__":
    unittest.main()
