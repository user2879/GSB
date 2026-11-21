from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from arbitration_workbench.workbench import Workbench

from .helpers import copy_sample_workspace, read_evidence_rows


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = copy_sample_workspace(Path(self.temp.name))
        self.workbench = Workbench(root=self.root, persist_edits=True)
        self.initial = self.workbench.run(use_cache=False, mode="full")
        self.assertEqual(48, len(self.initial.tickets))

    def tearDown(self):
        self.temp.cleanup()

    def test_type_weight_recomputes_only_referencing_tickets(self):
        references = self.workbench.reference_index()["evidence_types"]["known_device"]
        self.assertTrue(references)
        from arbitration_workbench.engine import score_ticket as original_score

        with mock.patch("arbitration_workbench.engine.score_ticket", wraps=original_score) as patched:
            affected = self.workbench.apply_edit(
                {"op": "set_type_weight", "evidence_type": "known_device", "value": "0.26"}
            )
            result = self.workbench.run()
        self.assertEqual(sorted(references), affected)
        called_ids = {call.args[0] for call in patched.call_args_list}
        self.assertEqual(set(references), called_ids)
        stats = result.diagnostics["invalidation_and_reuse"]["stages"]
        self.assertEqual(len(references), stats["score"]["computed"])
        self.assertEqual(48 - len(references), stats["score"]["reused"])
        self.assertTrue(self.workbench.compare_incremental_with_full(result)["equal"])

    def test_threshold_reuses_score_and_relations(self):
        from arbitration_workbench.engine import apply_relations as original_relations
        from arbitration_workbench.engine import decide_ticket as original_decision
        from arbitration_workbench.engine import score_ticket as original_score

        with (
            mock.patch("arbitration_workbench.engine.score_ticket", wraps=original_score) as score_mock,
            mock.patch("arbitration_workbench.engine.apply_relations", wraps=original_relations) as relation_mock,
            mock.patch("arbitration_workbench.engine.decide_ticket", wraps=original_decision) as decision_mock,
        ):
            affected = self.workbench.apply_edit({"op": "set_threshold", "value": "1.05"})
            result = self.workbench.run()
        self.assertEqual(48, len(affected))
        self.assertEqual(0, score_mock.call_count)
        self.assertEqual(0, relation_mock.call_count)
        self.assertGreater(decision_mock.call_count, 0)
        stats = result.diagnostics["invalidation_and_reuse"]["stages"]
        self.assertEqual(48, stats["score"]["reused"])
        self.assertEqual(48, stats["relations"]["reused"])
        self.assertEqual(48, stats["decision"]["computed"])
        self.assertEqual(48, stats["convert"]["computed"])
        self.assertTrue(self.workbench.compare_incremental_with_full(result)["equal"])

    def test_target_version_only_recomputes_conversion(self):
        from arbitration_workbench.engine import decide_ticket as original_decision
        from arbitration_workbench.engine import score_ticket as original_score

        with (
            mock.patch("arbitration_workbench.engine.score_ticket", wraps=original_score) as score_mock,
            mock.patch("arbitration_workbench.engine.decide_ticket", wraps=original_decision) as decision_mock,
        ):
            self.workbench.apply_edit({"op": "set_target_version", "value": "v3"})
            result = self.workbench.run()
        self.assertEqual(0, score_mock.call_count)
        self.assertEqual(0, decision_mock.call_count)
        stats = result.diagnostics["invalidation_and_reuse"]["stages"]
        self.assertEqual(48, stats["score"]["reused"])
        self.assertEqual(48, stats["relations"]["reused"])
        self.assertEqual(48, stats["decision"]["reused"])
        self.assertEqual(48, stats["convert"]["computed"])
        self.assertTrue(self.workbench.compare_incremental_with_full(result)["equal"])

    def test_single_evidence_replacement_is_ticket_local(self):
        from arbitration_workbench.engine import score_ticket as original_score

        rows = read_evidence_rows(self.root)
        target = next(row for row in rows if row["ticket_id"] == "T0010")
        with mock.patch("arbitration_workbench.engine.score_ticket", wraps=original_score) as patched:
            affected = self.workbench.apply_edit(
                {
                    "op": "replace_evidence",
                    "ticket_id": target["ticket_id"],
                    "evidence_seq": target["evidence_seq"],
                    "record": {
                        "timestamp": target["timestamp"] + 7,
                        "terms": sorted(set(target["terms"] + ["changed"])),
                    },
                }
            )
            result = self.workbench.run()
        self.assertEqual(["T0010"], affected)
        self.assertEqual({"T0010"}, {call.args[0] for call in patched.call_args_list})
        self.assertTrue(self.workbench.compare_incremental_with_full(result)["equal"])

    def test_arbitrary_edit_sequence_matches_blank_full_rebuild_every_time(self):
        rows = read_evidence_rows(self.root)
        replacement_target = next(row for row in rows if row["ticket_id"] == "T0020")
        sequence = [
            {"op": "set_type_weight", "evidence_type": "low_signal", "value": "0.002"},
            {"op": "set_role_trust", "source_role": "backend", "value": "0.87"},
            {
                "op": "upsert_relation",
                "relation": {
                    "id": "rel-phish-telemetry",
                    "kind": "support",
                    "left": {"evidence_type": "phishing_report", "source_roles": ["external"]},
                    "right": {"evidence_type": "telemetry_anomaly", "source_roles": ["backend"]},
                    "bonus": "0.03",
                },
            },
            {"op": "delete_relation", "relation_id": "rel-admin-vs-billing"},
            {"op": "set_threshold", "value": "1.10"},
            {"op": "set_half_life", "value": 5400},
            {
                "op": "replace_evidence",
                "ticket_id": "T0020",
                "evidence_seq": replacement_target["evidence_seq"],
                "record": {"timestamp": replacement_target["timestamp"] + 19, "terms": ["audit", "replaced"]},
            },
            {"op": "set_target_version", "value": "v3"},
            {"op": "set_target_version", "value": "v1"},
            {"op": "set_fusion", "value": "mean"},
            {
                "op": "set_priority_bands",
                "bands": [
                    {"level": 0, "name": "P0", "min_score": "2.00"},
                    {"level": 1, "name": "P1", "min_score": "1.40"},
                    {"level": 2, "name": "P2", "min_score": "0.85"},
                    {"level": 3, "name": "P3", "min_score": "0.00"},
                ],
            },
        ]
        for index, edit in enumerate(sequence, start=1):
            with self.subTest(index=index, op=edit["op"]):
                self.workbench.apply_edit(edit)
                result = self.workbench.run()
                comparison = self.workbench.compare_incremental_with_full(result)
                self.assertTrue(comparison["equal"], comparison)


if __name__ == "__main__":
    unittest.main()
