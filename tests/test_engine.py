from __future__ import annotations

import unittest

from colmv.canonical import canonical_json
from colmv.changes import load_changes
from colmv.engine import MaterializationEngine, select_path
from colmv.replay import split_changes
from colmv.schema import load_config

from tests.helpers import SAMPLES, sample_paths


def _run_path(config, changes, *, path: str, budget: int = 1):
    batches = split_changes(changes, [3, 6, 3, 6, 2])
    engine = MaterializationEngine(config, budget=budget, path_mode=path)
    results = [engine.process_batch(batch, batch_no=index) for index, batch in enumerate(batches, start=1)]
    return engine, results


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = sample_paths(SAMPLES)
        cls.config = load_config(cls.paths["catalog"], cls.paths["views"])
        cls.changes = load_changes(cls.paths["changes"], cls.config.table)

    def test_paths_are_byte_identical(self) -> None:
        resident, _ = _run_path(self.config, self.changes, path="resident")
        rescan, _ = _run_path(self.config, self.changes, path="rescan")
        self.assertEqual(resident.materialized_bytes(), rescan.materialized_bytes())
        self.assertEqual(resident.materialized_digest(), rescan.materialized_digest())

    def test_slot_partitions_are_strictly_complementary(self) -> None:
        resident, resident_batches = _run_path(self.config, self.changes, path="resident")
        rescan, rescan_batches = _run_path(self.config, self.changes, path="rescan")
        self.assertGreater(resident.total_replacements, 0)
        self.assertGreater(rescan.total_rescans, 0)
        self.assertEqual(resident.total_hits, rescan.total_hits)
        for resident_batch, rescan_batch in zip(resident_batches, rescan_batches):
            partition = resident_batch["slot_partition"]
            self.assertTrue(partition["complete_for_affected"])
            self.assertTrue(partition["disjoint"])
            resident_ids = {
                (group["view_id"], canonical_json(group["group_key"]))
                for group in partition["resident_groups"]
            }
            rescan_ids = {
                (group["view_id"], canonical_json(group["group_key"]))
                for group in partition["rescan_affected_groups"]
            }
            affected_ids = {
                (group["view_id"], canonical_json(group["group_key"]))
                for group in resident_batch["affected_groups"]
            }
            self.assertEqual(resident_ids | rescan_ids, affected_ids)
            self.assertFalse(resident_ids & rescan_ids)
            cross_path_rescan_ids = {
                (group["view_id"], canonical_json(group["group_key"]))
                for group in rescan_batch["slot_partition"]["rescan_affected_groups"]
            }
            self.assertEqual(rescan_ids, cross_path_rescan_ids)

    def test_auto_path_switch_history(self) -> None:
        batches = split_changes(self.changes, [3, 6, 3, 6, 2])
        engine = MaterializationEngine(self.config, budget=1, path_mode="auto")
        for index, batch in enumerate(batches, start=1):
            engine.process_batch(batch, batch_no=index)
        self.assertEqual([decision["path"] for decision in engine.batch_decisions], ["resident", "rescan", "resident", "rescan", "resident"])
        self.assertEqual(len(engine.switch_history), 4)
        self.assertTrue(all(item["metrics"]["boundary_band"] for item in engine.switch_history))

    def test_boundary_is_explicit_and_integer_tie_break_is_stable(self) -> None:
        boundary = select_path(
            change_count=4,
            affected_page_count=2,
            view_count=3,
            budget=1,
        )
        self.assertTrue(boundary.boundary_affected)
        self.assertEqual(boundary.path, "resident")
        self.assertIn("tie_break", boundary.metrics)
        above = select_path(change_count=5, affected_page_count=2, view_count=3, budget=1)
        below = select_path(change_count=3, affected_page_count=2, view_count=3, budget=1)
        self.assertEqual(above.path, "rescan")
        self.assertEqual(below.path, "resident")
        self.assertFalse(above.boundary_affected)
        self.assertFalse(below.boundary_affected)

    def test_null_group_keys_are_explicitly_excluded(self) -> None:
        engine, _ = _run_path(self.config, self.changes, path="resident")
        materialized = engine.materialized_result()
        self.assertGreater(materialized["excluded_null_group_key_rows"]["sales_by_region"], 0)
        self.assertGreater(materialized["excluded_null_group_key_rows"]["sales_region_product_live"], 0)
        for view in materialized["views"]:
            for group in view["groups"]:
                self.assertFalse(any(value is None for value in group["group_key"]))

    def test_fixed_point_and_error_summary_are_present(self) -> None:
        engine, _ = _run_path(self.config, self.changes, path="resident")
        materialized = engine.materialized_result()
        self.assertEqual(materialized["fixed_point"]["scale"], 1000)
        self.assertEqual(materialized["fixed_point"]["quantization"], "truncate-toward-zero-at-ingest")
        self.assertEqual(materialized["fixed_point"]["division"], "truncate-toward-zero")
        summary = materialized["error_summary"]
        self.assertGreater(summary["total_upper_bound_fp_units"], 0)
        self.assertIsNotNone(summary["dominant_segment"])
        self.assertEqual(summary["single_value_quantization_error"]["upper_bound_fp_units"], 1)


if __name__ == "__main__":
    unittest.main()
