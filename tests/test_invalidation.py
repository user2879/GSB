import copy
import unittest

from batch_receipt_ledger.invalidation import (
    compute_invalidation, compare_with_full_recompute, plan_identity_report)
from batch_receipt_ledger.plan import build_plan
from batch_receipt_ledger.spec import parse_spec

from tests._support import make_spec


class InvalidationTest(unittest.TestCase):
    def test_param_version_only_invalidates_owner_batches(self):
        old_raw = make_spec(workers=2, deliveries=9, batch_size=3)
        old_spec = parse_spec(old_raw)
        new_raw = copy.deepcopy(old_raw)
        new_raw["workers"][0]["param_version"] = 2
        new_spec = parse_spec(new_raw)
        old_plans = build_plan(old_spec)
        new_plans = build_plan(new_spec)
        steps, untouched = compute_invalidation(
            old_spec, new_spec, old_plans, new_plans)
        self.assertTrue(all(step.worker_id == "worker-a" for step in steps))
        self.assertEqual(
            sorted(untouched),
            [plan.batch_id for plan in new_plans
             if plan.worker_id == "worker-b"])

    def test_seed_change_invalidates_everything(self):
        old_spec = parse_spec(make_spec())
        new_raw = make_spec(seed=999)
        new_spec = parse_spec(new_raw)
        steps, untouched = compute_invalidation(
            old_spec, new_spec, build_plan(old_spec), build_plan(new_spec))
        self.assertEqual(len(untouched), 0)
        self.assertTrue(all(step.reason == "seed_changed" for step in steps))

    def test_batch_size_change_repartitions_all(self):
        old_spec = parse_spec(make_spec(batch_size=3))
        new_spec = parse_spec(make_spec(batch_size=2))
        steps, untouched = compute_invalidation(
            old_spec, new_spec, build_plan(old_spec), build_plan(new_spec))
        self.assertEqual(len(untouched), 0)
        self.assertTrue(
            all("batch_size" in step.reason for step in steps
                if "repartition" in step.reason))

    def test_full_recompute_comparison(self):
        partial = [{"batch_id": "b1", "results": [{"request_id": "r",
                    "output_digest": "x"}]}]
        full = [{"batch_id": "b1", "results": [{"request_id": "r",
                 "output_digest": "x"}]},
                {"batch_id": "b2", "results": []}]
        report = compare_with_full_recompute(partial, full)
        self.assertTrue(report["all_equal"])

        full[0]["results"][0]["output_digest"] = "y"
        report = compare_with_full_recompute(partial, full)
        self.assertFalse(report["all_equal"])
        self.assertEqual(report["mismatches"][0]["batch_id"], "b1")

    def test_identity_report_lists_changes(self):
        old_spec = parse_spec(make_spec())
        new_raw = make_spec()
        new_raw["workers"][1]["param_version"] = 5
        new_spec = parse_spec(new_raw)
        report = plan_identity_report(old_spec, new_spec)
        self.assertTrue(any(change[0] == "param_version"
                            for change in report["changes"]))


if __name__ == "__main__":
    unittest.main()
