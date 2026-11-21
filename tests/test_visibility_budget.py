from __future__ import annotations

import unittest

from mvcc_workbench.budget import build_budget_plan
from mvcc_workbench.loader import dataset_from_dict
from mvcc_workbench.visibility import compute_visibility

from tests.helpers import base_dataset, deepcopy, sample


class VisibilityTests(unittest.TestCase):
    def test_expected_visible_versions_and_reasons(self) -> None:
        dataset = base_dataset()
        decisions = {
            query_id: compute_visibility(dataset, dataset.queries_by_id[query_id])
            for query_id in sorted(dataset.queries_by_id)
        }
        self.assertEqual(decisions["Q01"].visible_version_no, 3)
        self.assertEqual(decisions["Q01"].visible_txn_id, "T3")
        self.assertTrue(decisions["Q01"].predicate_matched)
        self.assertEqual(decisions["Q02"].visible_version_no, 1)
        self.assertFalse(decisions["Q02"].predicate_matched)
        self.assertEqual(decisions["Q03"].visible_version_no, 3)
        self.assertEqual(decisions["Q04"].visible_version_no, 3)
        self.assertEqual(decisions["Q05"].visible_version_no, 5)
        self.assertEqual(decisions["Q06"].visible_version_no, 5)
        self.assertEqual(decisions["Q07"].visible_version_no, 1)
        self.assertEqual(decisions["Q07"].visible_txn_id, "T3")
        self.assertEqual(decisions["Q08"].visible_version_no, 1)

        q01_codes = [step["code"] for step in decisions["Q01"].reason_chain]
        self.assertIn("SKIP_COMMIT_AFTER_SNAPSHOT", q01_codes)
        self.assertIn("SKIP_ROLLEDBACK", q01_codes)
        self.assertIn("VISIBLE_OWN_COMMITTED", q01_codes)
        q07_codes = [step["code"] for step in decisions["Q07"].reason_chain]
        self.assertIn("VISIBLE_OWN_WRITE_POLICY", q07_codes)
        self.assertTrue(decisions["Q07"].visibility_policy["own_uncommitted_visible"])
        self.assertEqual(decisions["Q01"].cost_units, 4)

    def test_budget_plan_is_stable_and_never_overspends(self) -> None:
        dataset = base_dataset()
        first = build_budget_plan(dataset)
        second = build_budget_plan(dataset)
        self.assertEqual(first.canonical_json(), second.canonical_json())
        reversed_payload = deepcopy(sample("base.json"))
        reversed_payload["records"].reverse()
        reversed_plan = build_budget_plan(dataset_from_dict(reversed_payload))
        self.assertEqual(first.canonical_json(), reversed_plan.canonical_json())
        self.assertEqual(first.total_allocated_cost, 35)
        self.assertEqual(len(first.rounds), 3)
        self.assertFalse(first.deferred)
        self.assertEqual(first.rounds[0].allocated_cost, 11)
        self.assertEqual(
            [allocation.query_id for allocation in first.rounds[0].allocations],
            ["Q01", "Q03", "Q05"],
        )

        constrained = deepcopy(sample("base.json"))
        constrained["budget"]["global_cost_budget"] = 9
        constrained["budget"]["round_cost_budget"] = 5
        plan = build_budget_plan(dataset_from_dict(constrained))
        self.assertLessEqual(plan.total_allocated_cost, 9)
        self.assertGreater(len(plan.deferred), 0)
        self.assertTrue(all(item.reason for item in plan.deferred))
        self.assertIn("Q07", {item.query_id for item in plan.deferred})


if __name__ == "__main__":
    unittest.main()
