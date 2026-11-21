import unittest

from configmesh.compiler import compile_version, diff_invalidation, build_dispatch_plan
from configmesh.fixedpoint import Grid
from configmesh.manifest import load_bundle


class IncrementalCompilationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, cls.versions = load_bundle(
            "examples/manifest.json", "examples/registry.json")
        cls.grid = Grid.create(3)

    def test_unchanged_artifacts_byte_identical(self):
        report = diff_invalidation(self.versions[0], self.versions[1], self.grid)
        prev = report["artifact_digests"]["previous"]
        cur = report["artifact_digests"]["current"]
        for key in report["byte_identical_unchanged"]:
            self.assertEqual(prev[key], cur[key])
        for key in report["invalidation_set"]:
            self.assertNotIn(key, report["byte_identical_unchanged"])

    def test_propagation_order_is_topological(self):
        report = diff_invalidation(self.versions[0], self.versions[1], self.grid)
        self.assertEqual(report["invalidation_set"][0], "base_temp")
        propagated = {s["key"]: s["propagated_from"]
                      for s in report["propagation_steps"] if s["propagated_from"]}
        self.assertEqual(propagated["heat_index"], "base_temp")
        self.assertEqual(propagated["flow_score"], "base_temp")

    def test_target_and_strategy_change_flags(self):
        report = diff_invalidation(self.versions[1], self.versions[2], self.grid)
        self.assertTrue(report["node_targets_changed"])
        self.assertTrue(report["strategy_changed"])
        self.assertIn("gamma_mix", report["added"])
        plan = build_dispatch_plan(compile_version(self.versions[2], self.grid),
                                   self.versions[2])
        self.assertEqual(plan["nodes"]["node_gamma"]["shard_count"], 2)
        self.assertIn("plan_digest", plan)


if __name__ == "__main__":
    unittest.main()
