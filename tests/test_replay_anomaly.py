from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from gray_merge_lab.diagnostics import anomaly_suite
from gray_merge_lab.errors import LabError
from gray_merge_lab.replay import authoritative_recompute, replay_batch, replay_remaining
from tests.helpers import build_engine


class AuthoritativeRecomputeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.engine, self.ledger, self.spec = build_engine(
            self.root, sample_count=4, batch_size=2, weight_bp=10000, duration=9
        )
        self.executions = self.engine.process_samples(self.spec.samples)
        self.payload = self.engine.payload()

    def tearDown(self):
        self.ledger.close()
        self.temp.cleanup()

    def test_authoritative_recompute_matches_quick_payload(self):
        result = authoritative_recompute(self.ledger, self.payload)
        self.assertTrue(result["ok"])
        self.assertEqual(result["completed_batches"], 2)
        self.assertEqual(result["fixed_scale_exp"], 3)
        self.assertEqual(len(result["batches"]), 2)

    def test_preview_destination_mismatch_is_e_preview(self):
        tampered = copy.deepcopy(self.payload)
        tampered["preview_batches"][0]["destination"] = "tampered"
        with self.assertRaises(LabError) as caught:
            authoritative_recompute(self.ledger, tampered)
        self.assertEqual(caught.exception.code, "E_PREVIEW")
        self.assertEqual(caught.exception.batch_id, self.payload["preview_batches"][0]["batch_id"])
        self.assertIn("destination", caught.exception.field)

    def test_stat_mismatch_reports_first_rule_batch(self):
        tampered = copy.deepcopy(self.payload)
        tampered["rule_stats"]["rule-1"]["raw_duration_sum_ms"] += 1
        with self.assertRaises(LabError) as caught:
            authoritative_recompute(self.ledger, tampered)
        self.assertEqual(caught.exception.code, "E_PREVIEW")
        self.assertEqual(caught.exception.batch_id, self.payload["preview_batches"][0]["batch_id"])
        self.assertIn("rule_stats", caught.exception.field)

    def test_conclusion_mismatch_reports_first_batch(self):
        tampered = copy.deepcopy(self.payload)
        tampered["conclusions"][1]["destination"] = "tampered"
        with self.assertRaises(LabError) as caught:
            authoritative_recompute(self.ledger, tampered)
        self.assertEqual(caught.exception.code, "E_PREVIEW")
        self.assertEqual(caught.exception.batch_id, self.payload["conclusions"][1]["batch_id"])
        self.assertIn("conclusions[1]", caught.exception.field)

    def test_fixed_point_record_tamper_is_e_replay(self):
        record = copy.deepcopy(self.ledger.batches()[0])
        record["duration_q"] += 1
        with self.assertRaises(LabError) as caught:
            replay_batch(record)
        self.assertEqual(caught.exception.code, "E_REPLAY")
        self.assertEqual(caught.exception.field, "duration_q")
        self.assertEqual(caught.exception.batch_id, record["batch_id"])

    def test_merge_conclusion_tamper_is_e_replay(self):
        record = copy.deepcopy(self.ledger.batches()[0])
        record["conclusion"]["destination"] = "tampered"
        with self.assertRaises(LabError) as caught:
            replay_batch(record)
        self.assertEqual(caught.exception.code, "E_REPLAY")
        self.assertIn("conclusion", caught.exception.field)

    def test_route_tamper_is_e_replay(self):
        record = copy.deepcopy(self.ledger.batches()[0])
        record["routing_input"]["samples"][0]["client_key"] = "different-client"
        with self.assertRaises(LabError) as caught:
            replay_batch(record)
        self.assertEqual(caught.exception.code, "E_REPLAY")
        self.assertIn(caught.exception.field, {"destination", "buckets", "batch_id"})

    def test_replay_remaining_advances_from_snapshot_progress(self):
        partial = copy.deepcopy(self.payload)
        partial["progress"]["completed_batches"] = 1
        partial["progress"]["batch_ids"] = partial["progress"]["batch_ids"][:1]
        result = replay_remaining(self.ledger, partial)
        self.assertTrue(result["ok"])
        self.assertEqual(result["resumed_after_batch_index"], 1)
        self.assertEqual(result["replayed_batches"], 1)
        self.assertEqual(result["batch_ids"], [self.payload["preview_batches"][1]["batch_id"]])

    def test_replay_progress_outside_ledger_is_e_replay(self):
        payload = copy.deepcopy(self.payload)
        payload["progress"]["completed_batches"] = 99
        with self.assertRaises(LabError) as caught:
            replay_remaining(self.ledger, payload)
        self.assertEqual(caught.exception.code, "E_REPLAY")
        self.assertEqual(caught.exception.field, "progress.completed_batches")

    def test_hot_update_reinference_keeps_h1_h2_old_h3_and_new_h3(self):
        batch_id = self.executions[0].plan.batch_id
        original = self.ledger.receipt_rows(batch_id)
        self.assertEqual([item["h"] for item in original], ["h1", "h2", "h3"])
        engine = self.engine
        engine.hot_update_backend("canary", {"seed": 999, "hidden_dim": 7})
        reinfer = engine.reinfer_existing_batch(batch_id)
        receipts = self.ledger.receipt_rows(batch_id)
        self.assertEqual(len(receipts), 4)
        self.assertEqual([item["h"] for item in receipts], ["h1", "h2", "h3", "h3"])
        self.assertEqual(reinfer["adopted_merge_seq"], 3)
        self.assertEqual(reinfer["backend_param_version"], 2)
        self.assertEqual(reinfer["rejected_count"], 3)
        result = authoritative_recompute(self.ledger, engine.payload())
        self.assertTrue(result["ok"])

    def test_anomaly_suite_covers_requested_injections(self):
        results = anomaly_suite(self.engine)
        labels = {item["label"] for item in results}
        self.assertIn("receipt_digest_corruption", labels)
        self.assertIn("missing_backend_parameter_field", labels)
        self.assertIn("snapshot_digest_mismatch", labels)
        self.assertIn("preview_payload_mismatch", labels)
        self.assertIn("replay_fixed_point_mismatch", labels)
        self.assertTrue(all(item["ok"] for item in results), results)


if __name__ == "__main__":
    unittest.main()
