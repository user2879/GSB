import copy
import unittest
from pathlib import Path

from batchdiag.invalidation import (diff_workers, propagate, recompute_all,
                                    recompute_invalid, results_to_bytes)
from batchdiag.model import build_parameters, run_batch
from batchdiag.planner import build_plan, id_rule_bytes
from batchdiag.samples import generate, generate_changed_workers
from batchdiag.spec import Spec, load_spec

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "run" / "test_samples"


class PlanAndModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generate(SAMPLES)
        cls.spec = load_spec(SAMPLES / "requests.json",
                             SAMPLES / "workers.json",
                             SAMPLES / "merge_rules.json")

    def test_plan_byte_identical_under_dispatch_order(self):
        plan_a = build_plan(self.spec)
        shuffled = Spec(
            requests=list(reversed(self.spec.requests)),
            workers=list(reversed(self.spec.workers)),
            merge_rules=self.spec.merge_rules, model_dim=self.spec.model_dim)
        plan_b = build_plan(shuffled)
        self.assertEqual([b.batch_id for b in plan_a.batches],
                         [b.batch_id for b in plan_b.batches])
        self.assertEqual([b.requests for b in plan_a.batches],
                         [b.requests for b in plan_b.batches])

    def test_batch_id_uses_explicit_canonical_bytes(self):
        plan = build_plan(self.spec)
        batch = plan.batches[0]
        expected_prefix = f"B:{batch.worker_id}:{batch.batch_seq}:"
        self.assertTrue(batch.batch_id.startswith(expected_prefix))
        self.assertIn('"arrivals"', id_rule_bytes(
            batch.worker_id, batch.batch_seq, batch.arrivals).decode())

    def test_output_digest_order_independent(self):
        plan = build_plan(self.spec)
        batch = plan.batches[0]
        worker = self.spec.worker(batch.worker_id)
        params = build_parameters(worker)
        first = run_batch(worker, batch, params)
        reordered = copy.deepcopy(batch)
        object.__setattr__(reordered, "requests",
                           tuple(reversed(batch.requests)))
        vectors = reordered.vectors
        object.__setattr__(reordered, "vectors",
                           {k: vectors[k] for k in reversed(list(vectors))})
        second = run_batch(worker, reordered, params)
        self.assertEqual(first.output_digest, second.output_digest)
        self.assertEqual(first.param_digest, second.param_digest)

    def test_digest_binds_param_version(self):
        plan = build_plan(self.spec)
        batch = plan.batches[0]
        worker = self.spec.worker(batch.worker_id)
        first = run_batch(worker, batch)
        bumped = copy.deepcopy(worker)
        object.__setattr__(bumped, "param_version",
                           worker.param_version + 1)
        second = run_batch(bumped, batch)
        self.assertNotEqual(first.output_digest, second.output_digest)
        self.assertNotEqual(first.param_digest, second.param_digest)


class InvalidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generate(SAMPLES)
        cls.old = load_spec(SAMPLES / "requests.json",
                            SAMPLES / "workers.json",
                            SAMPLES / "merge_rules.json")
        generate_changed_workers(SAMPLES, SAMPLES / "workers_v2.json",
                                 bump_worker="w-bravo", new_version=2,
                                 new_seed=20261125 + 200)
        cls.new = load_spec(SAMPLES / "requests.json",
                            SAMPLES / "workers_v2.json",
                            SAMPLES / "merge_rules.json")

    def test_diff_locates_worker_and_fields(self):
        diffs = diff_workers(self.old, self.new)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["worker_id"], "w-bravo")
        self.assertEqual(set(diffs[0]["changed_fields"]),
                         {"seed", "param_version"})

    def test_propagation_boundary_is_only_changed_worker(self):
        report = propagate(self.old, self.new)
        affected_workers = {b["worker_id"] for b in report.invalid_batches}
        self.assertEqual(affected_workers, {"w-bravo"})
        self.assertEqual(len(report.invalid_batches), 2)
        invalid_request_workers = {r["worker_id"]
                                   for r in report.invalid_requests}
        self.assertEqual(invalid_request_workers, {"w-bravo"})
        self.assertTrue(all(b["direct"] for b in report.invalid_batches))
        self.assertIn("B:w-alpha:0:200231664249", report.valid_batch_ids)
        self.assertNotIn(
            "B:w-bravo:0:39cb4bb16fad", report.valid_batch_ids)
        # stable ordering
        seqs = [(b["worker_id"], b["batch_seq"])
                for b in report.invalid_batches]
        self.assertEqual(seqs, sorted(seqs))

    def test_incremental_matches_full_rebuild_byte_for_byte(self):
        old_plan = build_plan(self.old)
        new_plan = build_plan(self.new)
        report = propagate(self.old, self.new, old_plan, new_plan)
        invalid_ids = {b["batch_id"] for b in report.invalid_batches}
        fresh = recompute_invalid(self.new, new_plan, invalid_ids)
        merged = recompute_all(self.old, old_plan)
        merged.update(fresh)
        full = recompute_all(self.new, new_plan)
        self.assertEqual(results_to_bytes(merged), results_to_bytes(full))

    def test_untouched_batches_keep_identical_digests(self):
        old_plan = build_plan(self.old)
        new_plan = build_plan(self.new)
        old_results = recompute_all(self.old, old_plan)
        full = recompute_all(self.new, new_plan)
        report = propagate(self.old, self.new)
        for batch_id in report.valid_batch_ids:
            self.assertEqual(old_results[batch_id].output_digest,
                             full[batch_id].output_digest)

    def test_batch_size_change_changes_plan_for_that_worker(self):
        changed_workers = copy.deepcopy(self.old.workers)
        for worker in changed_workers:
            if worker.worker_id == "w-gamma":
                object.__setattr__(worker, "batch_size", 3)
        new_spec = Spec(requests=self.old.requests, workers=changed_workers,
                        merge_rules=self.old.merge_rules,
                        model_dim=self.old.model_dim)
        report = propagate(self.old, new_spec)
        self.assertIn("batch_size",
                      report.field_diffs[0]["changed_fields"])
        self.assertEqual({b["worker_id"] for b in report.invalid_batches},
                         {"w-gamma"})


if __name__ == "__main__":
    unittest.main()
