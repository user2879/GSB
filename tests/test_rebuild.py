import unittest

from shardcast.errors import DivergeError
from shardcast.rebuild import (
    full_rebuild_state,
    incremental_state,
    verify_incremental_parity,
)
from shardcast.sample import canonical_json
from shardcast.spec import parse_spec

from tests.test_propagation import base_raw


class RebuildParityTests(unittest.TestCase):
    def test_full_and_invalidated_only_recompute_match_field_by_field(self) -> None:
        old_spec = parse_spec(base_raw())
        new_raw = base_raw()
        new_raw["shards"][0]["sample_rows"] = 77
        new_spec = parse_spec(new_raw)
        assignments = {shard_id: "n1" for shard_id in new_spec.shard_ids}
        preserved_full = full_rebuild_state(old_spec, assignments, 1)
        report = verify_incremental_parity(
            old_spec, new_spec, assignments, 1, preserved_full
        )
        invalidated = {step["shard_id"] for step in report["invalidated"]}
        self.assertEqual(invalidated, {"a", "b", "c", "d"})
        self.assertEqual(
            canonical_json(report["incremental"]["summaries"]),
            canonical_json(report["full_rebuild"]["summaries"]),
        )
        self.assertEqual(
            canonical_json(report["incremental"]["water_marks"]),
            canonical_json(report["full_rebuild"]["water_marks"]),
        )

    def test_unaffected_summaries_remain_byte_identical(self) -> None:
        old_spec = parse_spec(base_raw())
        new_raw = base_raw()
        new_raw["shards"][2]["sample_rows"] = 99
        new_spec = parse_spec(new_raw)
        assignments = {shard_id: "n1" for shard_id in new_spec.shard_ids}
        preserved = full_rebuild_state(old_spec, assignments, 1)
        report = verify_incremental_parity(
            old_spec, new_spec, assignments, 1, preserved
        )
        self.assertEqual(report["unaffected"], ["a", "b"])
        for shard_id in report["unaffected"]:
            self.assertEqual(
                canonical_json(preserved["summaries"][shard_id]),
                canonical_json(report["incremental"]["summaries"][shard_id]),
            )

    def test_divergence_reports_first_field_path(self) -> None:
        spec = parse_spec(base_raw())
        assignments = {shard_id: "n1" for shard_id in spec.shard_ids}
        full = full_rebuild_state(spec, assignments, 1)
        broken = {
            "summaries": {key: dict(value) for key, value in full["summaries"].items()},
            "ownership": full["ownership"],
            "water_marks": full["water_marks"],
        }
        broken["summaries"]["a"]["participating_rows"] += 1
        from shardcast.propagation import compare_field_by_field

        with self.assertRaises(DivergeError) as caught:
            compare_field_by_field(
                {"summaries": broken["summaries"]},
                {"summaries": full["summaries"]},
                context="rebuild",
            )
        self.assertIn("summaries.a.participating_rows", caught.exception.path)


if __name__ == "__main__":
    unittest.main()
