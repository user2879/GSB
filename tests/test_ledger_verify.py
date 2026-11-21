from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from txn_mw.canonical import canonical_json
from txn_mw.engine import Engine, EngineConfig
from txn_mw.ledger import load_ledger, replace_ledger, validate_replaced_ledger
from txn_mw.logging import StructuredLogger
from txn_mw.store import Store, ledger_content_digest
from txn_mw.verify import verify_database, verify_ledger_file


class LedgerAndVerifyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.log_path = self.root / "structured.jsonl"
        self.log = StructuredLogger(self.log_path)

    def tearDown(self):
        self.temp.cleanup()

    def build_source(self, path: Path) -> None:
        engine = Engine(path, config=EngineConfig(), logger=self.log, start_worker=False)
        for txn_id, commit in (("ledger-a", True), ("ledger-b", False)):
            engine.persist_half(
                {
                    "txn_id": txn_id,
                    "version": 3,
                    "body": {"txn": txn_id},
                    "txn_timeout_ms": 500,
                    "query": {"host": "127.0.0.1", "port": 9},
                    "query_max": 3,
                }
            )
            engine.mark_half_ack(txn_id)
            engine.producer_command({"txn_id": txn_id, "version": 3}, "commit" if commit else "rollback")
        engine.pull({"txn_id": "ledger-a", "consumer_id": "consumer", "request_id": "pull-a"})
        engine.store.export_ledger(self.root / "source-ledger.json")
        engine.store.close()

    def test_whole_replacement_with_version_mismatch_and_readonly_backup(self):
        self.build_source(self.root / "source.db")
        value = load_ledger(self.root / "source-ledger.json")
        value["format_version"] = 2
        value["content_digest"] = ledger_content_digest(value)
        replacement = self.root / "replacement.json"
        replacement.write_text(canonical_json(value), encoding="utf-8")

        target = Store(self.root / "target.db")
        result = replace_ledger(target, replacement, logger=self.log)
        self.assertTrue(result["ok"])
        self.assertTrue(result["full_replacement"])
        self.assertFalse(result["incremental_merge"])
        self.assertTrue(Path(result["backup_path"]).exists())
        self.assertEqual(os.stat(result["backup_path"]).st_mode & 0o777, 0o444)
        self.assertEqual(target.get_meta("format_version"), "2")
        self.assertTrue(verify_database(target)["ok"])
        self.assertEqual(validate_replaced_ledger(target._conn), [])
        target.close()
        self.assertTrue(verify_ledger_file(replacement)["ok"])
        lines = [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(line.get("code") == "E_REPLACE" for line in lines))

    def test_replacement_validation_reports_first_inconsistent_field(self):
        self.build_source(self.root / "source.db")
        value = load_ledger(self.root / "source-ledger.json")
        for transaction in value["transactions"]:
            if transaction["txn_id"] == "ledger-a":
                transaction["query_count"] = 99
        value["content_digest"] = ledger_content_digest(value)
        replacement = self.root / "bad-replacement.json"
        replacement.write_text(canonical_json(value), encoding="utf-8")
        target = Store(self.root / "target.db")
        result = replace_ledger(target, replacement, logger=self.log)
        target.close()
        self.assertFalse(result["ok"])
        self.assertTrue(
            any(
                error["txn_id"] == "ledger-a" and error["field"] == "query_count"
                for error in result["validation_errors"]
            )
        )

    def test_verifier_reports_first_transaction_whose_replay_disagrees(self):
        engine = Engine(self.root / "verify.db", config=EngineConfig(), logger=self.log, start_worker=False)
        engine.persist_half(
            {
                "txn_id": "verify-a",
                "version": 1,
                "body": {"ok": True},
                "txn_timeout_ms": 500,
                "query": {"host": "127.0.0.1", "port": 9},
                "query_max": 3,
            }
        )
        engine.mark_half_ack("verify-a")
        engine.producer_command({"txn_id": "verify-a"}, "commit")
        engine.pull({"txn_id": "verify-a", "consumer_id": "c", "request_id": "p"})
        self.assertTrue(verify_database(engine.store)["ok"])
        with engine.store.transaction() as conn:
            conn.execute("UPDATE txns SET disposition='pending' WHERE txn_id='verify-a'")
            engine.store.refresh_digest(conn)
        result = verify_database(engine.store, logger=self.log)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "E_VERIFY")
        self.assertEqual(result["first_violation"]["txn_id"], "verify-a")
        lines = [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(line.get("code") == "E_VERIFY" for line in lines))
        engine.store.close()

    def test_engine_replaces_runtime_memory_without_merging_old_transactions(self):
        self.build_source(self.root / "source.db")
        target = Engine(self.root / "target-runtime.db", config=EngineConfig(), logger=self.log, start_worker=False)
        target.persist_half(
            {
                "txn_id": "only-in-target",
                "version": 1,
                "body": {"target": True},
                "txn_timeout_ms": 500,
                "query": {"host": "127.0.0.1", "port": 9},
                "query_max": 3,
            }
        )
        self.assertIn("only-in-target", target._runtime_txns)
        result = target.replace_from_ledger(self.root / "source-ledger.json")
        self.assertTrue(result["full_replacement"])
        self.assertTrue(result["runtime_memory_replacement"])
        self.assertNotIn("only-in-target", target._runtime_txns)
        self.assertEqual(set(target._runtime_txns), {"ledger-a", "ledger-b"})
        self.assertEqual(len(target._runtime_idempotency), 1)
        self.assertTrue(verify_database(target.store)["ok"])
        target.store.close()


if __name__ == "__main__":
    unittest.main()
