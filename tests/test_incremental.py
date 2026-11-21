from __future__ import annotations

import copy
import unittest

from memdiag.engine import apply_incremental, compare_runs, recompute_full_numeric, run_initial
from memdiag.errors import IncrementalError
from memdiag.plan import apply_plan_changes, plan_from_dict
from tests.helpers import fixture_plan, fixture_samples, read_fixture_json


class IncrementalEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = fixture_plan()
        cls.samples = fixture_samples()
        cls.base = run_initial(cls.plan, cls.samples)

    def _apply(self, name: str):
        changes = read_fixture_json(name)
        plan, changed, details = apply_plan_changes(self.plan, changes)
        run, invalidated = apply_incremental(plan, self.samples, self.base, changed)
        return plan, changed, run, invalidated

    def test_scale_change_invalidates_only_downstream(self):
        plan, changed, run, invalidated = self._apply("changes_scale.json")
        self.assertEqual(invalidated, ("branch_a", "merge", "head"))
        self.assertEqual(changed, {"branch_a": "scale_change"})
        for layer_id in ("stem", "branch_b"):
            before = self.base.layers[layer_id]
            after = run.layers[layer_id]
            self.assertEqual(after.version, before.version)
            self.assertEqual(after.output_digest, before.output_digest)
            self.assertEqual(after.sample_output_digests, before.sample_output_digests)
            self.assertEqual(after.block_ids, before.block_ids)
        for layer_id in ("branch_a", "merge", "head"):
            self.assertEqual(run.layers[layer_id].version, self.base.layers[layer_id].version + 1)

    def test_invalidation_trace_has_trigger_and_path(self):
        _, _, run, _ = self._apply("changes_scale.json")
        trace = {step.layer_id: step for step in run.invalidation}
        self.assertEqual(trace["branch_a"].depth, 0)
        self.assertEqual(trace["branch_a"].trigger_source, "branch_a")
        self.assertEqual(trace["merge"].trigger_source, "branch_a")
        self.assertEqual(trace["merge"].path[0], "branch_a")
        self.assertEqual(trace["head"].depth, 2)
        self.assertEqual(trace["head"].path, ("branch_a", "merge", "head"))
        for step in run.invalidation:
            self.assertTrue(step.reason)
            self.assertEqual(step.path[-1], step.layer_id)

    def test_dimension_change_only_invalidates_downstream_branch(self):
        _, _, run, invalidated = self._apply("changes_dim.json")
        self.assertEqual(invalidated, ("branch_b", "merge", "head"))
        self.assertNotIn("branch_a", invalidated)
        self.assertEqual(
            run.layers["branch_a"].output_digest,
            self.base.layers["branch_a"].output_digest,
        )

    def test_reference_change_invalidates_old_and_new_dependents(self):
        _, _, run, invalidated = self._apply("changes_reference.json")
        self.assertEqual(invalidated, ("branch_b", "merge", "head"))
        self.assertEqual(run.layers["stem"].version, 1)
        self.assertEqual(run.layers["branch_a"].version, 1)

    def test_full_numeric_oracle_matches_incremental(self):
        plan, _, run, invalidated = self._apply("changes_scale.json")
        full = recompute_full_numeric(
            plan,
            self.samples,
            self.base,
            invalidated,
            invalidation_trace=run.invalidation,
        )
        result = compare_runs(run, full)
        self.assertTrue(result["ok"])
        self.assertEqual(run.content_digest, full.content_digest)

    def test_E_INCR_locates_first_layer_and_field(self):
        _, _, run, _ = self._apply("changes_scale.json")
        corrupted = copy.deepcopy(run)
        corrupted.layers["merge"].output_digest = "0" * 64
        full = recompute_full_numeric(
            run.plan,
            self.samples,
            self.base,
            run_invalidated := tuple(step.layer_id for step in run.invalidation),
            invalidation_trace=run.invalidation,
        )
        with self.assertRaises(IncrementalError) as caught:
            compare_runs(corrupted, full)
        self.assertEqual(caught.exception.code, "E_INCR")
        self.assertEqual(caught.exception.details["first_difference_layer"], "merge")
        self.assertEqual(caught.exception.details["field"], "output_digest")

    def test_propagation_independent_of_json_traversal_order(self):
        document = read_fixture_json("plan.json")
        reordered = copy.deepcopy(document)
        reordered["layers"] = list(reversed(reordered["layers"]))
        reordered["consumers"] = dict(reversed(list(reordered["consumers"].items())))
        reordered["memory_derived_from"] = dict(
            reversed(list(reordered["memory_derived_from"].items()))
        )
        alternate = plan_from_dict(reordered)
        self.assertEqual(alternate.digest, self.plan.digest)
        changes = read_fixture_json("changes_scale.json")
        plan_a, changed_a, _ = apply_plan_changes(self.plan, changes)
        plan_b, changed_b, _ = apply_plan_changes(alternate, changes)
        run_a, invalid_a = apply_incremental(plan_a, self.samples, self.base, changed_a)
        base_b = run_initial(alternate, self.samples)
        run_b, invalid_b = apply_incremental(plan_b, self.samples, base_b, changed_b)
        self.assertEqual(invalid_a, invalid_b)
        self.assertEqual(
            [step.to_dict() for step in run_a.invalidation],
            [step.to_dict() for step in run_b.invalidation],
        )
        self.assertEqual(run_a.content_digest, run_b.content_digest)


if __name__ == "__main__":
    unittest.main()
