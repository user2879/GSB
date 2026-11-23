from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gray_merge_lab.errors import LabError
from gray_merge_lab.ledger import LEDGER_FORMAT_VERSION, Ledger
from gray_merge_lab.snapshot import SnapshotStore, verify_snapshot_document
from tests.helpers import build_engine, test_spec


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.engine, self.ledger, self.spec = build_engine(self.root, sample_count=4, duration=7)
        self.executions = self.engine.process_samples(self.spec.samples)

    def tearDown(self):
        self.ledger.close()
        self.temp.cleanup()

    def test_batch_and_receipt_transaction_is_persisted(self):
        integrity = self.ledger.verify_integrity()
        self.assertTrue(integrity["ok"])
        self.assertEqual(integrity["format_version"], LEDGER_FORMAT_VERSION)
        self.assertEqual(integrity["completed_batches"], 2)
        self.assertEqual(len(self.ledger.batches()), 2)
        self.assertEqual(len(self.ledger.receipt_rows()), 6)
        self.assertEqual(self.ledger.cursor()["next_batch_index"], 2)
        first = self.ledger.batches()[0]
        self.assertEqual(first["raw_duration_ms"], 7)
        self.assertEqual(first["duration_q"], 7000)
        self.assertEqual(first["samples_q"], 2000)
        self.assertEqual(first["conclusion"]["merge_seq"], 2)

    def test_content_digest_tamper_is_detected(self):
        self.ledger.conn.execute(
            "UPDATE batches SET record_json=? WHERE batch_id=?",
            ("{}", self.ledger.batches()[0]["batch_id"]),
        )
        with self.assertRaises(LabError) as caught:
            self.ledger.verify_integrity()
        self.assertEqual(caught.exception.code, "E_LEDGER")

    def test_failed_refresh_rolls_back_batch_and_receipts(self):
        target = Ledger(self.root / "rollback.sqlite3")
        record = self.ledger.batches()[0]
        receipts = self.ledger.receipt_rows(record["batch_id"])
        try:
            with mock.patch.object(
                target, "_refresh_metadata", side_effect=sqlite3.OperationalError("injected")
            ):
                with self.assertRaises(LabError) as caught:
                    target.record_batch(copy.deepcopy(record), receipts)
            self.assertEqual(caught.exception.code, "E_LEDGER")
            self.assertEqual(len(target.batches()), 0)
            self.assertEqual(len(target.receipt_rows()), 0)
            self.assertEqual(target.cursor()["completed_batches"], 0)
        finally:
            target.close()

    def test_snapshot_contains_fixed_statistics_and_conclusions(self):
        payload = self.engine.payload()
        self.assertEqual(payload["scale_exp"], 3)
        self.assertEqual(len(payload["preview_batches"]), 2)
        self.assertEqual(len(payload["conclusions"]), 2)
        stats = payload["rule_stats"]["rule-1"]
        self.assertEqual(stats["batch_count"], 2)
        self.assertEqual(stats["raw_duration_sum_ms"], 14)
        self.assertEqual(stats["raw_sample_sum"], 4)
        self.assertEqual(stats["avg_duration_q"], 7000)
        self.assertEqual(stats["avg_samples_q"], 2000)
        self.assertEqual(stats["fixedpoint"]["single_value_error_bound_original"], 0.001)
        self.assertEqual([item["batch_index"] for item in payload["preview_batches"]], [0, 1])


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.engine, self.ledger, self.spec = build_engine(self.root, sample_count=4, duration=5)
        self.engine.process_samples(self.spec.samples)
        self.store = SnapshotStore(self.root / "snapshot.json")
        self.store.write(self.engine.manifest(), self.engine.payload())

    def tearDown(self):
        self.ledger.close()
        self.temp.cleanup()

    def test_valid_snapshot_loads_without_replacement(self):
        result = self.store.load_or_replace(self.engine.manifest(), self.engine.payload())
        self.assertFalse(result.replaced)
        self.assertEqual(result.differences, [])
        document = self.store.read_valid()
        verify_snapshot_document(document)
        self.assertEqual(document["snapshot_format_version"], 1)

    def test_manifest_mismatch_replaces_whole_snapshot(self):
        old_manifest = copy.deepcopy(self.engine.manifest())
        changed = copy.deepcopy(old_manifest)
        changed["scale_exp"] += 1
        with self.assertRaises(LabError) as caught:
            self.store.load_or_replace(changed, {"replacement": True})
        self.assertEqual(caught.exception.code, "E_SNAPSHOT")
        self.assertTrue(any(item["field"] == "manifest.scale_exp" for item in caught.exception.details["differences"]))
        document = json.loads((self.root / "snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(document["manifest"]["scale_exp"], changed["scale_exp"])
        self.assertEqual(document["payload"], {"replacement": True})
        self.assertTrue((self.root / "snapshot.json.rejected.json").exists())

    def test_content_digest_mismatch_is_replaced_and_reported(self):
        document = json.loads((self.root / "snapshot.json").read_text(encoding="utf-8"))
        document["content_digest"] = "sha256:" + "0" * 64
        (self.root / "snapshot.json").write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(LabError) as caught:
            self.store.read_valid()
        self.assertEqual(caught.exception.code, "E_SNAPSHOT")
        self.assertEqual(caught.exception.details["field"], "content_digest")
        with self.assertRaises(LabError) as replaced:
            self.store.load_or_replace(self.engine.manifest(), {"fresh": True})
        self.assertEqual(replaced.exception.code, "E_SNAPSHOT")
        document = json.loads((self.root / "snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(document["payload"], {"fresh": True})
        verify_snapshot_document(document)

    def test_snapshot_content_digest_is_stable_for_same_payload(self):
        first = json.loads((self.root / "snapshot.json").read_text(encoding="utf-8"))
        self.store.write(self.engine.manifest(), self.engine.payload())
        second = json.loads((self.root / "snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(first["content_digest"], second["content_digest"])


if __name__ == "__main__":
    unittest.main()

