import unittest

from elastic_scaler.common import ceil_div
from elastic_scaler.estimator import (
    REASON_BUDGET,
    REASON_QUOTA,
    decide_minute,
    estimate_service,
)
from elastic_scaler.spec import load_spec

SAMPLE = "samples/cluster_v1.json"


class EstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_spec(SAMPLE)

    def test_ceil_division_is_explicit_integer(self):
        self.assertEqual(ceil_div(0, 3), 0)
        self.assertEqual(ceil_div(1, 3), 1)
        self.assertEqual(ceil_div(3, 3), 1)
        self.assertEqual(ceil_div(4, 3), 2)
        self.assertEqual(ceil_div(10, 2), 5)

    def test_estimate_repeats_identically_and_exposes_intermediates(self):
        svc = self.spec.service("asr")
        first = estimate_service(svc, 4, svc.requests[4])
        second = estimate_service(svc, 4, svc.requests[4])
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertGreaterEqual(first.per_replica_capacity, 1)
        self.assertIn("integer ceiling", " ".join(first.rules))
        self.assertGreaterEqual(first.demand, first.raw_demand - 10_000)

    def test_trim_respects_global_budget_and_quota(self):
        decision = decide_minute(self.spec, 4, 1)
        self.assertLessEqual(
            sum(a.allocated_replicas for a in decision.allocations),
            self.spec.global_replica_budget,
        )
        reasons = {a.service: list(a.trim_reasons) for a in decision.allocations}
        self.assertIn(REASON_QUOTA, reasons["asr"])
        self.assertTrue(any(REASON_BUDGET in r for r in reasons.values()))
        for alloc in decision.allocations:
            svc = self.spec.service(alloc.service)
            self.assertLessEqual(alloc.allocated_replicas, svc.max_replicas)
            self.assertLessEqual(
                alloc.allocated_replicas, svc.concurrency_quota // svc.concurrency_limit
            )
            self.assertGreaterEqual(alloc.rejected_requests, 0)
            self.assertEqual(
                alloc.rejected_requests,
                max(0, alloc.estimate.requests - alloc.servable_requests),
            )

    def test_minimum_replicas_always_reserved(self):
        decision = decide_minute(self.spec, 7, 1)
        for alloc in decision.allocations:
            svc = self.spec.service(alloc.service)
            self.assertGreaterEqual(alloc.allocated_replicas, svc.min_replicas)


if __name__ == "__main__":
    unittest.main()
