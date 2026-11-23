from __future__ import annotations

import copy
import unittest
from dataclasses import replace

from config_pipeline.errors import E_DIVERGE, PipelineError
from config_pipeline.manifest import manifest_from_dict
from config_pipeline.planner import Plan, compare_plans, compile_plan
from tests.helpers import manifest


class PlannerTests(unittest.TestCase):
    def test_plan_bytes_are_independent_of_edit_order(self) -> None:
        original = manifest(ports=(41101, 41102), config_keys=("base", "scaled"))
        original["tasks"][0]["target_clusters"] = ["c0", "c1"]
        shuffled = copy.deepcopy(original)
        shuffled["clusters"].reverse()
        shuffled["tasks"][0]["config_keys"].reverse()
        shuffled["tasks"][0]["target_clusters"].reverse()
        shuffled["revisions"][0]["values"] = dict(
            reversed(list(shuffled["revisions"][0]["values"].items()))
        )
        left = compile_plan(manifest_from_dict(original))
        right = compile_plan(manifest_from_dict(shuffled))
        self.assertEqual(left.serialized(), right.serialized())
        self.assertEqual(left.plan_digest, right.plan_digest)

    def test_divergence_reports_first_stable_shard_and_field(self) -> None:
        parsed = manifest_from_dict(manifest(ports=(41111,)))
        plan = compile_plan(parsed)
        bad_shard = replace(
            plan.shards[0],
            config_values={"scaled": plan.shards[0].config_values["scaled"] + 1},
        )
        bad_plan = Plan(
            manifest_digest=plan.manifest_digest,
            shards=(bad_shard,) + plan.shards[1:],
            plan_digest=plan.plan_digest,
        )
        with self.assertRaises(PipelineError) as caught:
            compare_plans(plan, bad_plan)
        self.assertEqual(caught.exception.code, E_DIVERGE)
        self.assertEqual(caught.exception.shard_id, plan.shards[0].stable_id)
        self.assertIn("config_values.scaled", caught.exception.field or "")

    def test_selective_compile_preserves_old_source_revision(self) -> None:
        data = manifest(ports=(41121,))
        data["revisions"].append(
            {"version": 2, "parent": 1, "values": {"unused": 9}}
        )
        parsed = manifest_from_dict(data)
        latest = compile_plan(parsed)
        preserved_id = latest.shards[0].stable_id
        selective = compile_plan(parsed, {preserved_id: 1})
        self.assertEqual(selective.by_id[preserved_id].revision_version, 1)
        self.assertEqual(selective.by_id[latest.shards[1].stable_id].revision_version, 2)

    def test_planner_uses_task_concurrency_default(self) -> None:
        data = manifest(ports=(41131,), task_limit=1)
        del data["tasks"][0]["concurrency_limit"]
        parsed = manifest_from_dict(data)
        self.assertEqual(parsed.tasks[0].concurrency_limit, 3)


if __name__ == "__main__":
    unittest.main()
