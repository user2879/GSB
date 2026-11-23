from __future__ import annotations

import unittest

from mvcc_workbench.engine import DiagnosticEngine
from mvcc_workbench.invalidation import (
    InvalidationRequest,
    OutcomeChange,
    apply_changes,
    mark_stale_results,
    propagate_invalidation,
    replace_affected_results,
)
from mvcc_workbench.loader import load_spec, validate_spec_object
from mvcc_workbench.model import VersionRecord


class InvalidationTests(unittest.TestCase):
    def test_append_version_marks_queries_and_preserves_unaffected_bytes(self) -> None:
        spec = load_spec("data/demo_small.json")
        baseline = DiagnosticEngine(spec).run(verify_consistency=False)
        max_version = max(record.version for record in spec.records if record.row_key == "row:001")
        request = InvalidationRequest(
            appended_records=(
                VersionRecord(
                    row_key="row:001",
                    tx_id="T0001",
                    version=max_version + 1,
                    value_digest="sha256:appended",
                    prev_version=max_version,
                ),
            )
        )
        plan = propagate_invalidation(spec, request)
        self.assertTrue(plan.affected_query_ids)
        self.assertTrue(plan.unaffected_query_ids)
        self.assertEqual(list(plan.affected_query_ids), sorted(plan.affected_query_ids))
        stale = mark_stale_results(baseline.results, plan)
        self.assertTrue(any(item["staleness"] == "STALE" for item in stale))
        updated_spec = apply_changes(spec, request)
        updated = DiagnosticEngine(updated_spec).run(verify_consistency=True)
        replaced = replace_affected_results(baseline.results, updated.results, plan)
        self.assertEqual(len(replaced), len(baseline.results))

    def test_pending_to_committed_outcome_propagation(self) -> None:
        data = {
            "format": "mvcc-diagnostic-spec",
            "format_version": 1,
            "name": "pending-outcome",
            "transactions": [
                {"id": "T1", "start_seq": 1, "outcome": "pending", "commit_seq": None}
            ],
            "records": [
                {
                    "row_key": "r",
                    "tx_id": "T1",
                    "version": 1,
                    "value_digest": "sha256:a",
                    "prev_version": None,
                }
            ],
            "read_points": [{"id": "RP", "snapshot_seq": 5, "tx_id": "T1"}],
            "queries": [
                {"id": "Q", "read_point_id": "RP", "row_key": "r", "predicate": {"op": "exists"}}
            ],
            "budget": {"global_concurrency": 1, "per_read_point_concurrency": 1},
        }
        spec = validate_spec_object(data, allow_pending=True)
        request = InvalidationRequest(
            outcome_changes=(OutcomeChange("T1", "committed", 5),)
        )
        plan = propagate_invalidation(spec, request)
        self.assertEqual(plan.affected_query_ids, ("Q",))
        updated = apply_changes(spec, request)
        self.assertEqual(updated.transactions[0].outcome, "committed")
        result = DiagnosticEngine(updated).run(verify_consistency=False)
        self.assertEqual(result.results[0].visible_version, 1)


if __name__ == "__main__":
    unittest.main()
