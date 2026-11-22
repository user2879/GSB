from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from txnmsg.canonical import canonical_json
from txnmsg.errors import RecordError, SnapshotError, SpecError
from txnmsg.replay import replay_events
from txnmsg.store import Ledger

from tests.helpers import load_domain, make_ledger


class LedgerTests(unittest.TestCase):
    def test_state_machine_and_contiguous_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            ledger = make_ledger(path)
            try:
                ledger.advance_state("t1", "prepared")
                self.assertEqual(ledger.query_prefixes()[0].prefix, -1)
                ledger.advance_state("t1", "committing")
                ledger.advance_state("t1", "committed")
                orders = ledger.query_prefixes()[0]
                self.assertEqual((orders.prefix, orders.truncated_position), (1, 2))
                self.assertEqual(orders.truncated_reason, "pending")

                ledger.advance_state("t2", "prepared")
                ledger.advance_state("t2", "committing")
                ledger.advance_state("t2", "committed")
                orders = ledger.query_prefixes()[0]
                self.assertEqual(orders.prefix, 3)
                self.assertEqual(orders.truncated_position, 4)

                ledger.advance_state("t3", "prepared")
                ledger.advance_state("t3", "aborted")
                payments = ledger.query_prefixes()[1]
                self.assertEqual(payments.prefix, -1)
                self.assertEqual(payments.truncated_position, 0)
                self.assertEqual(payments.truncated_reason, "aborted")
            finally:
                ledger.close()

    def test_aborted_transaction_truncates_later_committed_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            ledger = make_ledger(path)
            try:
                ledger.advance_state("t1", "prepared")
                ledger.advance_state("t1", "aborted")
                ledger.advance_state("t2", "prepared")
                ledger.advance_state("t2", "committing")
                ledger.advance_state("t2", "committed")
                orders = ledger.query_prefixes()[0]
                self.assertEqual(orders.prefix, -1)
                self.assertEqual(orders.truncated_position, 0)
                self.assertEqual(orders.truncated_reason, "aborted")
                self.assertEqual(orders.truncated_transaction_id, "t1")
            finally:
                ledger.close()

    def test_invalid_transition_does_not_append_event(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            ledger = make_ledger(path)
            try:
                ledger.advance_state("t1", "prepared")
                with self.assertRaises(SpecError) as raised:
                    ledger.advance_state("t1", "committed")
                self.assertEqual(raised.exception.code, "E_SPEC")
                self.assertEqual(ledger._event_cursor(), 1)
                ledger.advance_state("t1", "committing")
                ledger.advance_state("t1", "committed")
                self.assertEqual(ledger._event_cursor(), 3)
            finally:
                ledger.close()

    def test_commit_writes_visibility_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            ledger = make_ledger(path)
            try:
                ledger.advance_state("t1", "prepared")
                ledger.advance_state("t1", "committing")
                self.assertEqual(ledger.visibility(), ())
                result = ledger.advance_state("t1", "committed")
                rows = ledger.visibility()
                self.assertEqual([row["position"] for row in rows], [0, 1])
                self.assertTrue(
                    all(row["effective_event_seq"] == result.event_seq for row in rows)
                )
                self.assertEqual(ledger._event_cursor(), 3)
            finally:
                ledger.close()

    def test_record_length_corruption_is_e_rec(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            ledger = make_ledger(path)
            try:
                ledger.advance_state("t1", "prepared")
            finally:
                ledger.close()
            connection = sqlite3.connect(str(path))
            connection.execute("UPDATE events SET body_length=body_length+1 WHERE event_seq=1")
            connection.commit()
            connection.close()
            with self.assertRaises(RecordError) as raised:
                Ledger(path)
            self.assertEqual(raised.exception.code, "E_REC")
            self.assertEqual(raised.exception.record_seq, 1)
            self.assertEqual(raised.exception.event_seq, 1)
            self.assertEqual(raised.exception.field, "body_length")

    def test_visibility_digest_corruption_is_e_rec(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            ledger = make_ledger(path)
            try:
                ledger.advance_state("t1", "prepared")
                ledger.advance_state("t1", "committing")
                ledger.advance_state("t1", "committed")
            finally:
                ledger.close()
            connection = sqlite3.connect(str(path))
            connection.execute("UPDATE visibility SET body_digest='sha256:bad' WHERE position=0")
            connection.commit()
            connection.close()
            with self.assertRaises(RecordError) as raised:
                Ledger(path)
            self.assertEqual(raised.exception.field, "body_digest")
            self.assertIsNotNone(raised.exception.record_seq)

    def test_ledger_digest_mismatch_is_snapshot_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ledger.sqlite"
            copy_path = root / "copy.sqlite"
            ledger = make_ledger(path)
            try:
                ledger.advance_state("t1", "prepared")
            finally:
                ledger.close()
            shutil.copy2(path, copy_path)
            connection = sqlite3.connect(str(copy_path))
            connection.execute(
                "UPDATE meta SET value=? WHERE key='ledger_digest'",
                ("sha256:" + "0" * 64,),
            )
            connection.commit()
            connection.close()
            with self.assertRaises(SnapshotError) as raised:
                Ledger(copy_path)
            self.assertEqual(raised.exception.code, "E_SNAPSHOT")
            self.assertEqual(raised.exception.field, "ledger_digest")

    def test_batch_boundaries_produce_byte_identical_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = make_ledger(root / "first.sqlite")
            second = make_ledger(root / "second.sqlite")
            try:
                first.advance_many(
                    [
                        ("t1", "prepared"),
                        ("t1", "committing"),
                        ("t1", "committed"),
                        ("t2", "prepared"),
                        ("t2", "committing"),
                        ("t2", "committed"),
                    ]
                )
                for transaction_id, state in (
                    ("t1", "prepared"),
                    ("t1", "committing"),
                    ("t1", "committed"),
                    ("t2", "prepared"),
                    ("t2", "committing"),
                    ("t2", "committed"),
                ):
                    second.advance_state(transaction_id, state)
                first_json = canonical_json(
                    [report.as_dict() for report in first.query_prefixes()]
                )
                second_json = canonical_json(
                    [report.as_dict() for report in second.query_prefixes()]
                )
                self.assertEqual(first_json, second_json)

                partitions, transactions = load_domain()
                events = list(first.events())
                shuffled = list(reversed(events))
                outcome = replay_events(partitions, transactions, shuffled)
                replay_json = canonical_json(
                    [report.as_dict() for report in outcome.partitions]
                )
                self.assertEqual(first_json, replay_json)
            finally:
                first.close()
                second.close()


if __name__ == "__main__":
    unittest.main()
