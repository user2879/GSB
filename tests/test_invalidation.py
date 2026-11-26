import os
import unittest

from leasebatch.invalidation import baseline_versions, propagate
from leasebatch.fixedpoint import forward_batch
from leasebatch.plan import build_plan
from leasebatch.spec import load_spec_dir


class InvalidationTest(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec_dir(
            os.path.join(os.path.dirname(__file__), "..", "samples")
        )
        self.old_plan = build_plan(self.spec, 2, 42)

    def test_param_version_bump_invalidates_only_that_worker(self):
        old_versions = baseline_versions(self.spec)
        new_versions = dict(old_versions, w1=1)
        report = propagate(
            self.spec, self.old_plan, self.old_plan,
            old_versions, new_versions,
        )
        invalid = set(report["invalid_batch_ids"])
        self.assertEqual(
            invalid,
            {b.batch_id for b in self.old_plan.batches
             if b.worker_id == "w1"},
        )
        self.assertIn(
            "w2-b0000-1ae037d0b5bc", report["untouched_batch_ids"]
        )
        self.assertEqual(
            set(report["invalid_request_ids"]),
            {"r01", "r02", "r04", "r06"},
        )
        bases = {step["basis"] for step in report["steps"]
                 if step["level"] == "batch"}
        self.assertTrue(any("param_version_change:w1" in b for b in bases))

    def test_seed_change_invalidates_everything(self):
        new_plan = build_plan(self.spec, 2, 43)
        versions = baseline_versions(self.spec)
        report = propagate(
            self.spec, self.old_plan, new_plan, versions, versions,
            seed_changed=True,
        )
        self.assertEqual(
            len(report["invalid_batch_ids"]), len(self.old_plan.batches)
        )
        self.assertEqual(report["untouched_batch_ids"], [])

    def test_batch_size_change_propagates_partition_rebuild(self):
        new_plan = build_plan(self.spec, 3, 42)
        versions = baseline_versions(self.spec)
        report = propagate(
            self.spec, self.old_plan, new_plan, versions, versions,
            batch_size_changed=True,
        )
        self.assertTrue(report["invalid_batch_ids"])
        self.assertTrue(
            all("batch_size_change" in s["basis"]
                for s in report["steps"] if s["level"] == "batch")
        )

    def test_output_order_of_report_is_stable(self):
        versions = baseline_versions(self.spec)
        first = propagate(self.spec, self.old_plan, self.old_plan,
                          versions, dict(versions, w1=1))
        second = propagate(self.spec, self.old_plan, self.old_plan,
                           versions, dict(versions, w1=1))
        self.assertEqual(first["steps"], second["steps"])

    def test_untouched_batch_digests_remain_byte_identical(self):
        versions = baseline_versions(self.spec)
        report = propagate(
            self.spec, self.old_plan, self.old_plan,
            versions, dict(versions, w1=1),
        )
        untouched_id = report["untouched_batch_ids"][0]
        batch = next(b for b in self.old_plan.batches
                     if b.batch_id == untouched_id)
        worker = self.spec.worker_by_id[batch.worker_id]
        before = forward_batch(
            list(batch.request_ids), worker.worker_id, worker.model_dim,
            worker.hidden_dim, versions[worker.worker_id],
            self.old_plan.seed,
        )
        after = forward_batch(
            list(batch.request_ids), worker.worker_id, worker.model_dim,
            worker.hidden_dim, versions[worker.worker_id],
            self.old_plan.seed,
        )
        self.assertEqual(before["output_digest"], after["output_digest"])
        self.assertEqual(batch.content_digest,
                         build_plan(self.spec, 2, 42)
                         .by_worker()[worker.worker_id][batch.batch_index]
                         .content_digest)


if __name__ == "__main__":
    unittest.main()
