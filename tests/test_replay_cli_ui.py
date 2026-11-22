from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from txnmsg.canonical import canonical_json
from txnmsg.cli import main as cli_main
from txnmsg.demo import run_demo
from txnmsg.diagnostics import build_diagnostics
from txnmsg.errors import ReplayError
from txnmsg.replay_verify import replay_verify
from txnmsg.snapshots import create_snapshot, next_snapshot_path
from txnmsg.store import Ledger

from tests.helpers import extra_transaction, make_ledger


class ReplayCliUiTests(unittest.TestCase):
    def test_replay_reports_first_event_cursor_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = make_ledger(root / "ledger.sqlite")
            try:
                ledger.advance_state("t1", "prepared")
                ledger.advance_state("t1", "committing")
                ledger.advance_state("t1", "committed")
                snapshot = create_snapshot(ledger, next_snapshot_path(root))
                ledger.ingest(extra_transaction())
                ledger.advance_state("t4", "prepared")
                with self.assertRaises(ReplayError) as raised:
                    replay_verify(ledger, snapshot.path)
                self.assertEqual(raised.exception.code, "E_REPLAY")
                self.assertEqual(raised.exception.field, "event_cursor")
                self.assertEqual(raised.exception.event_seq, 4)
            finally:
                ledger.close()

    def test_replay_reports_state_field_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = make_ledger(root / "ledger.sqlite")
            try:
                ledger.advance_state("t1", "prepared")
                ledger.advance_state("t1", "committing")
                ledger.advance_state("t1", "committed")
                ledger.connection.execute(
                    "UPDATE transactions SET current_state='prepared' WHERE transaction_id='t1'"
                )
                ledger.connection.commit()
                with self.assertRaises(ReplayError) as raised:
                    replay_verify(ledger)
                self.assertEqual(raised.exception.transaction_id, "t1")
                self.assertEqual(
                    raised.exception.field, "transactions[t1].current_state"
                )
            finally:
                ledger.close()

    def test_diagnostics_is_stable_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "ledger.sqlite"
            ledger = make_ledger(database)
            try:
                ledger.advance_state("t1", "prepared")
                ledger.advance_state("t1", "committing")
                ledger.advance_state("t1", "committed")
                digest = ledger.computed_ledger_digest()
            finally:
                ledger.close()
            with Ledger(database, auto_recover=False, readonly=True) as readonly:
                first = build_diagnostics(readonly, root)
                second = build_diagnostics(readonly, root)
                self.assertEqual(canonical_json(first), canonical_json(second))
                self.assertEqual(readonly.computed_ledger_digest(), digest)
                self.assertEqual(first["ledger_format_version"], 1)
                self.assertEqual(first["partitions"][0]["prefix"], 1)
                self.assertEqual(first["partitions"][0]["state_distribution"]["committed"], 1)
                self.assertIn("snapshots", first)

    def test_cli_validation_anomaly_codes_and_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "samples"
            samples.mkdir()
            partitions = {
                "partitions": [
                    {"partition_id": "orders", "partition_version": 1, "max_position": 10}
                ]
            }
            (samples / "partitions.json").write_text(json.dumps(partitions), encoding="utf-8")
            corrupt = samples / "corrupt.json"
            corrupt.write_text('{"transactions": [', encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli_main(
                    [
                        "validate",
                        "--workdir",
                        str(root),
                        "--partitions",
                        str(samples / "partitions.json"),
                        "--transactions",
                        str(corrupt),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stdout.getvalue())["code"], "E_SPEC")
            records = [
                json.loads(line)
                for line in (root / "logs" / "business.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[-1]["code"], "E_SPEC")

            duplicate = {
                "transactions": [
                    {
                        "transaction_id": "a",
                        "txn_seq": 1,
                        "partition_id": "orders",
                        "local_txn_version": 1,
                        "commit_time_ms": 1,
                        "messages": [{"position": 0, "payload": 1}],
                    },
                    {
                        "transaction_id": "b",
                        "txn_seq": 2,
                        "partition_id": "orders",
                        "local_txn_version": 1,
                        "commit_time_ms": 2,
                        "messages": [{"position": 0, "payload": 2}],
                    },
                ]
            }
            duplicate_path = samples / "duplicate.json"
            duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli_main(
                    [
                        "validate",
                        "--workdir",
                        str(root),
                        "--partitions",
                        str(samples / "partitions.json"),
                        "--transactions",
                        str(duplicate_path),
                    ]
                )
            self.assertEqual(code, 4)
            self.assertEqual(json.loads(stdout.getvalue())["code"], "E_ORDER")

    def test_cli_ledger_digest_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = make_ledger(root / "ledger.sqlite")
            try:
                ledger.advance_state("t1", "prepared")
            finally:
                ledger.close()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli_main(
                    ["inject", "--workdir", str(root), "--kind", "ledger-digest"]
                )
            self.assertEqual(code, 5)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["code"], "E_SNAPSHOT")
            self.assertTrue(Path(payload["input"]).exists())

    def test_end_to_end_demo(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_demo(directory)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["committed_event_crash"]["exit_code"], 74)
            self.assertTrue(
                result["committed_event_crash"]["recovery"][
                    "persisted_prefix_repaired"
                ]
            )
            self.assertEqual(result["committing_crash"]["exit_code"], 74)
            unknown = result["committing_crash"]["recovery"]["unknown_transactions"]
            self.assertEqual(unknown[0]["transaction_id"], "tx-order-003")
            self.assertEqual(unknown[0]["intervals"], [[4, 5]])
            self.assertFalse(
                result["committing_crash"]["recovery"]["exit_report_available"]
            )
            self.assertEqual(result["replay_after_restore"]["status"], "PASS")
            self.assertEqual(result["snapshot_restore"]["event_cursor"], 17)
            self.assertIn("tx-pay-003", result["snapshot_restore"]["diff"]["transaction_removed"])
            self.assertTrue(result["snapshot_restore"]["backup_readonly"])
            self.assertEqual(
                {key: value["code"] for key, value in result["anomalies"].items()},
                {
                    "corrupt_batch": "E_SPEC",
                    "duplicate_position": "E_ORDER",
                    "ledger_digest": "E_SNAPSHOT",
                },
            )
            self.assertEqual(
                result["structured_log"]["codes"], ["E_ORDER", "E_SNAPSHOT", "E_SPEC"]
            )
            self.assertTrue(result["diagnostics"]["stable_render"])
            summary = Path(result["summary_path"])
            self.assertTrue(summary.exists())
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8"))["status"], "PASS")
            for backup in (Path(directory) / "backups").glob("*.sqlite"):
                os.chmod(backup, 0o644)


if __name__ == "__main__":
    unittest.main()
