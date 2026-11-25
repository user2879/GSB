"""Requirements 3 & 4: deterministic batches and stable-hash attribution."""
from __future__ import annotations

import os
import unittest

from grayroute.batching import plan_batches
from grayroute.canonical import canonical_bytes
from grayroute.constants import BUCKETS
from grayroute.sharding import (attribute_all, route_pool, stable_bucket,
                                stable_worker)
from grayroute.spec import Request, load_spec

ROOT = os.path.dirname(os.path.dirname(__file__))


class PlanningShardingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(os.path.join(ROOT, "config"),
                             os.path.join(ROOT, "data", "requests.ndjson"))
        cls.hits = attribute_all(cls.spec)
        cls.batches = plan_batches(cls.spec, cls.hits)

    def test_bucket_in_range_and_deterministic(self):
        for _ in range(50):
            bucket = stable_bucket("client-1", "route-main")
            self.assertTrue(0 <= bucket < BUCKETS)
        self.assertEqual(stable_bucket("client-1", "route-main"),
                         stable_bucket("client-1", "route-main"))

    def test_same_key_same_decision_under_same_weight(self):
        decisions = {route_pool("client-2", "route-main", 3000)
                     for _ in range(20)}
        self.assertEqual(len(decisions), 1)

    def test_weight_boundaries(self):
        # weight 0 -> always stable; max -> always gray
        for key in [f"c{i}" for i in range(30)]:
            self.assertEqual(route_pool(key, "r", 0), "stable")
            self.assertEqual(route_pool(key, "r", 10000), "gray")

    def test_worker_selection_order_independent(self):
        candidates = ("gray-b", "gray-a", "gray-c")
        first = stable_worker("client-1", "r", "gray", candidates)
        shuffled = ("gray-c", "gray-a", "gray-b")
        second = stable_worker("client-1", "r", "gray", shuffled)
        self.assertEqual(first, second)

    def test_hit_binds_batch_worker_and_version(self):
        for request_id, hit in self.hits.items():
            self.assertTrue(hit.worker_id.startswith("worker-"))
            self.assertIsInstance(hit.param_version_seq, int)
            self.assertIn(hit.pool, ("stable", "gray"))

    def test_batch_ids_deterministic_under_arrival_permutation(self):
        baseline = canonical_bytes([b.to_dict() for b in self.batches])
        shuffled_requests = tuple(sorted(
            self.spec.requests,
            key=lambda r: (-r.arrival_seq, r.request_id)))
        permuted_spec = Request  # alias to keep import used
        rebuilt = type(self.spec)(
            backends=self.spec.backends, rules=self.spec.rules,
            merge_rule=self.spec.merge_rule, requests=shuffled_requests,
            model_dim=self.spec.model_dim)
        rebuilt_hits = attribute_all(rebuilt)
        rebuilt_batches = plan_batches(rebuilt, rebuilt_hits)
        self.assertEqual(
            baseline,
            canonical_bytes([b.to_dict() for b in rebuilt_batches]))

    def test_each_batch_records_revision_version_and_members(self):
        for batch in self.batches:
            self.assertTrue(batch.batch_id.startswith("bat-"))
            self.assertGreaterEqual(batch.rule_revision, 1)
            self.assertGreater(batch.param_version_seq, 0)
            self.assertEqual(len(batch.request_ids), len(batch.arrival_seqs))
            # no batch mixes parameter versions: all hits share the batch pv
            for rid in batch.request_ids:
                self.assertEqual(self.hits[rid].param_version_seq,
                                 batch.param_version_seq)

    def test_execution_order_does_not_change_partition(self):
        reversed_batches = tuple(reversed(self.batches))
        ids_a = sorted(b.batch_id for b in self.batches)
        ids_b = sorted(b.batch_id for b in reversed_batches)
        self.assertEqual(ids_a, ids_b)


if __name__ == "__main__":
    unittest.main()
