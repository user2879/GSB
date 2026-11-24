from __future__ import annotations

import json
import sqlite3
import unittest

from tlschain.errors import DomainError
from tlschain.ledger import Ledger
from tlschain.spec import load_spec

from tests.helpers import CERT_PATH, HANDSHAKE_PATH, MODIFIED_CERT_PATH, temporary_directory


class LedgerTests(unittest.TestCase):
    def test_snapshot_whole_replacement_and_diff(self) -> None:
        directory = temporary_directory("tlschain-ledger-snapshot-")
        path = directory / "ledger.sqlite"
        base = load_spec(CERT_PATH, HANDSHAKE_PATH)
        modified = load_spec(MODIFIED_CERT_PATH, HANDSHAKE_PATH)
        with Ledger(path) as ledger:
            initial = ledger.sync_snapshot(base)
            self.assertTrue(initial["replaced"])
            self.assertEqual(initial["reason"], "missing")
            unchanged = ledger.sync_snapshot(base)
            self.assertFalse(unchanged["replaced"])
            replaced = ledger.sync_snapshot(modified)
            self.assertTrue(replaced["replaced"])
            self.assertEqual(replaced["reason"], "content_digest_mismatch")
            self.assertEqual(replaced["changed_cert_ids"], ["leaf-expired"])
            self.assertEqual(replaced["replacement_mode"], "whole_snapshot_replace")
            ledger.conn.execute("UPDATE session_snapshot SET format_version = 99 WHERE id = 1")
            version_replaced = ledger.sync_snapshot(modified)
            self.assertTrue(version_replaced["replaced"])
            self.assertEqual(version_replaced["reason"], "format_version_mismatch")

    def test_snapshot_lists_added_and_removed_certificates(self) -> None:
        directory = temporary_directory("tlschain-ledger-add-remove-")
        path = directory / "ledger.sqlite"
        document = json.loads(CERT_PATH.read_text(encoding="utf-8"))
        extra = dict(document["certificates"][4])
        extra["id"] = "leaf-extra"
        extra["serial"] = 3999
        extra["issuer"] = "root-ca"
        document["certificates"].append(extra)
        extra_path = directory / "certificates_extra.json"
        extra_path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        base = load_spec(CERT_PATH, HANDSHAKE_PATH)
        with_extra = load_spec(extra_path, HANDSHAKE_PATH)
        with Ledger(path) as ledger:
            ledger.sync_snapshot(base)
            added = ledger.sync_snapshot(with_extra)
            self.assertEqual(added["added_cert_ids"], ["leaf-extra"])
            self.assertEqual(added["removed_cert_ids"], [])
            removed = ledger.sync_snapshot(base)
            self.assertEqual(removed["added_cert_ids"], [])
            self.assertEqual(removed["removed_cert_ids"], ["leaf-extra"])

    def test_integer_format_version_and_content_digest(self) -> None:
        directory = temporary_directory("tlschain-ledger-version-")
        path = directory / "ledger.sqlite"
        record = {"seq": 1, "decision": "PASS", "expected_match": True, "affected_by_error": False}
        with Ledger(path) as ledger:
            summary = ledger.commit_batch(1, [record], "spec", 1, [])
            self.assertEqual(ledger.progress_cursor(), 1)
            self.assertEqual(summary["content_digest"], ledger.content_digest())
            row = ledger.conn.execute("SELECT format_version FROM ledger_info WHERE id = 1").fetchone()
            self.assertIsInstance(row["format_version"], int)
            self.assertEqual(row["format_version"], 1)
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE batches SET progress_cursor = 99 WHERE batch_index = 1")
            connection.commit()
        with self.assertRaises(DomainError) as raised:
            Ledger(path)
        self.assertEqual(raised.exception.code, "E_REPLAY")
        self.assertEqual(raised.exception.field_path, "ledger.content_digest")

    def test_record_digest_tamper_is_detected(self) -> None:
        directory = temporary_directory("tlschain-ledger-record-")
        path = directory / "ledger.sqlite"
        record = {"seq": 1, "decision": "PASS", "expected_match": True, "affected_by_error": False}
        with Ledger(path) as ledger:
            ledger.commit_batch(1, [record], "spec", 1, [])
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE messages SET record_json = ? WHERE seq = 1", (json.dumps({**record, "decision": "FAIL"}),))
            connection.commit()
        with Ledger(path) as ledger:
            with self.assertRaises(DomainError) as raised:
                ledger.committed_records()
        self.assertEqual(raised.exception.code, "E_REPLAY")
        self.assertEqual(raised.exception.message_seq, 1)
        self.assertEqual(raised.exception.field_path, "ledger.record_digest")

    def test_unknown_sequences_follow_progress_cursor(self) -> None:
        directory = temporary_directory("tlschain-ledger-unknown-")
        path = directory / "ledger.sqlite"
        spec = load_spec(CERT_PATH, HANDSHAKE_PATH)
        records = [
            {"seq": 1, "decision": "PASS", "expected_match": True, "affected_by_error": False},
            {"seq": 2, "decision": "PASS", "expected_match": True, "affected_by_error": False},
        ]
        with Ledger(path) as ledger:
            ledger.commit_batch(1, records, "spec", 2, [])
            self.assertEqual(ledger.unknown_sequences(spec), [3, 4, 5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
