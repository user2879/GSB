from __future__ import annotations

import unittest

from mvcc_workbench.loader import load_spec
from mvcc_workbench.model import build_index
from mvcc_workbench.planner import MATERIALIZED, PER_QUERY, select_plan


class PlannerTests(unittest.TestCase):
    def test_sample_scale_choices(self) -> None:
        small = load_spec("data/demo_small.json")
        large = load_spec("data/demo_large.json")
        small_plan = select_plan(small, build_index(small))
        large_plan = select_plan(large, build_index(large))
        self.assertEqual(small_plan.selected_path, PER_QUERY)
        self.assertEqual(large_plan.selected_path, MATERIALIZED)
        self.assertEqual(large_plan.threshold_crossed_count, 2)
        self.assertIn("max_untriggered", large_plan.boundary_values["query_count"]["boundary"])

    def test_tie_band_uses_threshold_majority_and_marks_influence(self) -> None:
        transactions = []
        records = []
        previous = None
        for version in range(1, 9):
            tx_id = f"T{version:02d}"
            transactions.append(
                {
                    "id": tx_id,
                    "start_seq": version,
                    "outcome": "committed",
                    "commit_seq": version + 20,
                }
            )
            records.append(
                {
                    "row_key": "r",
                    "tx_id": tx_id,
                    "version": version,
                    "value_digest": f"sha256:{version:064x}",
                    "prev_version": previous,
                }
            )
            previous = version
        data = {
            "format": "mvcc-diagnostic-spec",
            "format_version": 1,
            "name": "tie-band",
            "transactions": transactions,
            "records": records,
            "read_points": [{"id": "RP", "snapshot_seq": 30, "tx_id": "T01"}],
            "queries": [
                {"id": "Q", "read_point_id": "RP", "row_key": "r", "predicate": {"op": "exists"}}
            ],
            "budget": {"global_concurrency": 2, "per_read_point_concurrency": 1},
        }
        from mvcc_workbench.loader import validate_spec_object

        spec = validate_spec_object(data)
        plan = select_plan(spec, build_index(spec))
        self.assertTrue(plan.tie_band["in_band"])
        self.assertTrue(plan.threshold_influenced)
        self.assertEqual(plan.selected_path, plan.default_path)
        self.assertIn("ESTIMATE_TIE_BAND", plan.reason_codes)


if __name__ == "__main__":
    unittest.main()
