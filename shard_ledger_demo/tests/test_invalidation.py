import unittest

from shard_ledger.compute import aggregate, effective_params
from shard_ledger.invalidation import propagate
from shard_ledger.spec import load_manifest

SAMPLE = __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "samples", "manifest.json")


class InvalidationTest(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(SAMPLE)
        self.completed = set(self.manifest.shards)

    def test_direct_and_indirect_propagation_is_stable(self):
        revision = self.manifest.revisions[2]
        ordered, basis = propagate(self.manifest, self.completed, revision)
        self.assertEqual(ordered, ["S1", "S2", "S3", "S5", "S7"])
        direct = [step["shard_id"] for step in basis if step["step"] == "direct"]
        indirect = [step["shard_id"] for step in basis if step["step"] == "dependency"]
        self.assertEqual(sorted(direct), ["S1", "S2", "S5", "S7"])
        self.assertEqual(indirect, ["S3"])
        self.assertNotIn("S4", ordered)
        self.assertNotIn("S6", ordered)

    def test_only_completed_shards_are_considered(self):
        revision = self.manifest.revisions[2]
        ordered, _ = propagate(self.manifest, {"S1"}, revision)
        self.assertEqual(ordered, ["S1", "S3", "S5", "S7"])

    def test_effective_params_unchanged_for_unaffected_node(self):
        before = effective_params(self.manifest.revisions, 1, "N1")
        after = effective_params(self.manifest.revisions, 2, "N1")
        self.assertEqual(before, after)
        changed = effective_params(self.manifest.revisions, 2, "N2")
        self.assertEqual(changed["scale"], 2500)
        self.assertEqual(changed["sample_count"], 96)


class ComputeDeterminismTest(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(SAMPLE)

    def test_aggregate_is_byte_stable_across_calls(self):
        first = aggregate("N1", "S4", 101, 1, self.manifest.revisions)
        second = aggregate("N1", "S4", 101, 1, self.manifest.revisions)
        self.assertEqual(first, second)

    def test_unaffected_node_summary_identical_after_revision(self):
        before = aggregate("N1", "S4", 101, 1, self.manifest.revisions)
        after = aggregate("N1", "S4", 101, 2, self.manifest.revisions)
        for field in ("sample_count", "group_count", "groups", "total_sum", "params"):
            self.assertEqual(before[field], after[field])
        self.assertEqual(before["params"], after["params"])

    def test_affected_node_summary_changes_after_revision(self):
        before = aggregate("N2", "S1", 202, 1, self.manifest.revisions)
        after = aggregate("N2", "S1", 202, 2, self.manifest.revisions)
        self.assertNotEqual(before["total_sum"], after["total_sum"])
        self.assertEqual(after["config_version"], 2)

    def test_groups_are_fixed_point_integers(self):
        summary = aggregate("N3", "S2", 303, 1, self.manifest.revisions)
        for group in summary["groups"]:
            for key in ("count", "sum", "min", "max", "xor"):
                self.assertIsInstance(group[key], int)


if __name__ == "__main__":
    unittest.main()
