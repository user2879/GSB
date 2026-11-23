from __future__ import annotations

import unittest

from config_pipeline.evolution import build_evolution
from config_pipeline.manifest import manifest_from_dict
from tests.helpers import evolved, manifest


class EvolutionTests(unittest.TestCase):
    def test_unrelated_config_key_does_not_invalidate_shards(self) -> None:
        original = manifest(ports=(41301, 41302))
        new = evolved(original, values={"unused": 5})
        delta = build_evolution(
            manifest_from_dict(original), manifest_from_dict(new)
        )
        self.assertEqual(delta.recompute_ids, ())
        self.assertEqual(delta.removed_ids, ())
        self.assertEqual(delta.diff["added"], [])
        self.assertEqual(delta.diff["removed"], [])
        self.assertEqual(delta.diff["changed"], [])
        self.assertEqual(len(delta.preserved_revisions), 6)
        for shard in delta.new_plan.shards:
            self.assertEqual(shard.revision_version, 1)

    def test_derived_dependency_propagates_with_stable_basis(self) -> None:
        original = manifest(ports=(41311, 41312))
        new = evolved(original, values={"base": 99})
        delta = build_evolution(
            manifest_from_dict(original), manifest_from_dict(new)
        )
        self.assertEqual(len(delta.recompute_ids), 6)
        self.assertEqual(tuple(sorted(delta.recompute_ids)), delta.recompute_ids)
        self.assertTrue(all(shard.revision_version == 2 for shard in delta.new_plan.shards))
        first = delta.invalidations[0]
        rendered = " ".join(first.basis)
        self.assertIn("base:10->99", rendered)
        self.assertIn("scaled -> base", rendered)
        self.assertIn("task:task", rendered)
        self.assertEqual(
            [item.first_revision for item in delta.invalidations],
            sorted(item.first_revision for item in delta.invalidations),
        )

    def test_target_and_shard_parameter_changes_are_added_and_removed(self) -> None:
        original = manifest(ports=(41321, 41322), task_shards=3, targets=("c0", "c1"))
        new = evolved(original, values={"unused": 2})
        new["tasks"][0]["target_clusters"] = ["c0"]
        new["tasks"][0]["shards"] = 2
        delta = build_evolution(
            manifest_from_dict(original), manifest_from_dict(new)
        )
        self.assertTrue(delta.removed_ids)
        self.assertEqual(len(delta.removed_ids), 4)
        self.assertIn("c1::task::s000000", delta.removed_ids)
        self.assertEqual(len(delta.recompute_ids), 2)
        all_reasons = " ".join(
            reason for item in delta.invalidations for reason in item.basis
        )
        self.assertIn("definition_changed", all_reasons)
        self.assertIn("target", all_reasons)


if __name__ == "__main__":
    unittest.main()
