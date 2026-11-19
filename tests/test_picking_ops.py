from __future__ import annotations

import copy
import unittest

from pickworkbench.canonical import canonical_bytes, content_hash
from pickworkbench.engine import run_batch, run_click_query, run_ray_query
from pickworkbench.errors import E_SNAP_SPACING, WorkbenchError
from pickworkbench.generator import generate_sample_with_config
from pickworkbench.geometry import perform_drag, replay_drag_without_snap


class PickingAndOperationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene, cls.queries = generate_sample_with_config()
        cls.batch = run_batch(cls.scene, cls.queries)

    def test_normal_batch_passes_every_query(self) -> None:
        self.assertEqual(self.batch["summary"]["status"], "pass")
        self.assertEqual(self.batch["summary"]["passed"], 22)
        self.assertEqual(self.batch["summary"]["failed"], 0)
        self.assertIsNone(self.batch["first_affected"])

    def test_ray_records_repeat_byte_identically(self) -> None:
        query = next(item for item in self.queries["queries"] if item["kind"] == "ray")
        first = run_ray_query(self.scene, query)
        second = run_ray_query(self.scene, query)
        self.assertEqual(canonical_bytes(first), canonical_bytes(second))
        for field in ("object_id", "hit_distance", "hit_point", "intersection_method"):
            self.assertIn(field, first["winner"])

    def test_click_matches_equivalent_ray_and_tolerance_is_invariant(self) -> None:
        click_records = [record for record in self.batch["records"] if record["query_type"] == "click"]
        self.assertEqual(len(click_records), 8)
        changed_candidate_sets = 0
        for record in click_records:
            self.assertTrue(record["equivalence"]["matches"])
            self.assertTrue(record["tolerance_probe"]["winner_unchanged"])
            self.assertEqual(
                record["winner"]["object_id"],
                record["equivalent_ray_winner"]["object_id"],
            )
            self.assertAlmostEqual(
                record["winner"]["hit_distance"],
                record["equivalent_ray_winner"]["hit_distance"],
                places=10,
            )
            changed_candidate_sets += int(record["tolerance_probe"]["candidate_set_changed"])
        self.assertGreater(changed_candidate_sets, 0)

    def test_changing_tolerance_only_changes_candidate_membership(self) -> None:
        click = next(item for item in self.queries["queries"] if item["kind"] == "click")
        baseline = run_click_query(self.scene, copy.deepcopy(click))
        widened_query = copy.deepcopy(click)
        widened_query["tolerance_px"] = float(click["tolerance_px"]) + 16.0
        widened = run_click_query(self.scene, widened_query)
        self.assertGreaterEqual(
            len(widened["candidate_object_ids"]),
            len(baseline["candidate_object_ids"]),
        )
        self.assertEqual(baseline["winner"]["object_id"], widened["winner"]["object_id"])
        self.assertEqual(baseline["winner"]["hit_point"], widened["winner"]["hit_point"])

    def test_drag_repeatability_restore_and_orientation(self) -> None:
        query = next(item for item in self.queries["queries"] if item["kind"] == "drag")
        first = perform_drag(self.scene["camera"], query)
        second = perform_drag(self.scene["camera"], query)
        self.assertEqual(canonical_bytes(first), canonical_bytes(second))
        self.assertTrue(first["orientation_unchanged"])
        self.assertEqual(first["orientation_before"], first["orientation_after"])
        replay = replay_drag_without_snap(self.scene["camera"], query)
        for left, right in zip(replay["final_position"], first["unsnapped_position"]):
            self.assertAlmostEqual(left, right, places=10)
        spacing = float(query["snap_spacing"])
        self.assertAlmostEqual(first["final_position"][0] / spacing, round(first["final_position"][0] / spacing), places=9)
        self.assertAlmostEqual(first["final_position"][2] / spacing, round(first["final_position"][2] / spacing), places=9)
        self.assertAlmostEqual(first["final_position"][1], first["unsnapped_position"][1], places=12)

    def test_nonuniform_snap_spacing_is_rejected(self) -> None:
        query = copy.deepcopy(next(item for item in self.queries["queries"] if item["kind"] == "drag"))
        query["snap_spacing"] = [0.25, 0.20, 0.25]
        with self.assertRaises(WorkbenchError) as context:
            perform_drag(self.scene["camera"], query)
        self.assertEqual(context.exception.code, E_SNAP_SPACING)

    def test_batch_and_record_hashes_are_stable(self) -> None:
        second = run_batch(self.scene, self.queries)
        self.assertEqual(self.batch["records_hash"], second["records_hash"])
        self.assertEqual(content_hash(self.batch["records"]), second["records_hash"])
        for record in self.batch["records"]:
            payload = {key: value for key, value in record.items() if key != "content_hash"}
            self.assertEqual(record["content_hash"], content_hash(payload))


if __name__ == "__main__":
    unittest.main()
