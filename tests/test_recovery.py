from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from txn_mw.engine import Engine, EngineConfig
from txn_mw.logging import StructuredLogger
from txn_mw.store import Store
from txn_mw.verify import verify_database


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "mw.db"
        self.log = StructuredLogger(self.root / "structured.jsonl")
        self.config = EngineConfig(query_budget_ms=50, query_retry_ms=10, query_tick_ms=5)

    def tearDown(self):
        self.temp.cleanup()

    def test_unfinished_query_is_marked_unknown_and_requeued(self):
        engine = Engine(self.db, config=self.config, logger=self.log, start_worker=False)
        engine.persist_half(
            {
                "txn_id": "recover-query",
                "version": 1,
                "body": {"case": "unfinished-query"},
                "txn_timeout_ms": 100,
                "query": {"host": "127.0.0.1", "port": 9},
                "query_max": 3,
            }
        )
        engine.mark_half_ack("recover-query")
        now = time.monotonic_ns()
        with engine.store.transaction() as conn:
            tx = engine._get_tx(conn, "recover-query")
            tx.query_count = 1
            tx.query_inflight = True
            tx.first_query_mono_ns = now
            tx.last_query_mono_ns = now
            engine._save_tx(conn, tx)
            engine._append(
                conn,
                tx,
                "query_started",
                {
                    "attempt": 1,
                    "started_mono_ns": now,
                    "first_query_mono_ns": now,
                    "deadline_mono_ns": tx.deadline_mono_ns,
                },
            )
            engine.store.refresh_digest(conn)
        engine.store.close()

        recovered = Engine(self.db, config=self.config, logger=self.log, start_worker=False)
        report = recovered.recovery_report
        self.assertIn("recover-query", report["incomplete_queries_marked_unknown_and_requeued"])
        self.assertIn("recover-query", report["pending_query_transactions"])
        self.assertFalse(report["shutdown_report_available"])
        trace = recovered.transaction_trace("recover-query")
        self.assertFalse(trace["query_inflight"])
        self.assertEqual(trace["last_query_result"], "unknown")
        self.assertEqual(trace["events"][-1]["type"], "recovery_query_unknown")
        self.assertTrue(verify_database(recovered.store)["ok"])
        recovered.store.close()

    def test_tail_truncation_is_quarantined_and_reported(self):
        engine = Engine(self.db, config=self.config, logger=self.log, start_worker=False)
        engine.persist_half(
            {
                "txn_id": "tail",
                "version": 1,
                "body": {"case": "tail"},
                "txn_timeout_ms": 100,
                "query": {"host": "127.0.0.1", "port": 9},
                "query_max": 3,
            }
        )
        engine.mark_half_ack("tail")
        with engine.store.transaction() as conn:
            row = conn.execute("SELECT row_id, record FROM events ORDER BY event_seq DESC LIMIT 1").fetchone()
            conn.execute(
                "UPDATE events SET record=? WHERE row_id=?",
                (bytes(row["record"])[:-1], row["row_id"]),
            )
            engine.store.refresh_digest(conn)
        engine.store.close()

        recovered = Engine(self.db, config=self.config, logger=self.log, start_worker=False)
        report = recovered.recovery_report
        self.assertEqual(len(report["tail_incomplete_records"]), 1)
        self.assertEqual(report["tail_incomplete_records"][0]["path"], "$.record.length")
        with recovered.store.read() as conn:
            invalid = conn.execute("SELECT COUNT(*) AS n FROM events WHERE valid=0").fetchone()["n"]
            quarantined = conn.execute("SELECT COUNT(*) AS n FROM quarantine").fetchone()["n"]
        self.assertEqual(invalid, 1)
        self.assertEqual(quarantined, 1)
        lines = [json.loads(line) for line in (self.root / "structured.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(line.get("code") == "E_LOG" for line in lines))
        recovered.store.close()
        recovered_again = Engine(self.db, config=self.config, logger=self.log, start_worker=False)
        with recovered_again.store.read() as conn:
            quarantined_again = conn.execute("SELECT COUNT(*) AS n FROM quarantine").fetchone()["n"]
        self.assertEqual(quarantined_again, 1)
        recovered_again.store.close()

    def test_unfinished_pull_requires_re_pull(self):
        engine = Engine(self.db, config=self.config, logger=self.log, start_worker=False)
        with engine.store.transaction() as conn:
            conn.execute(
                "INSERT INTO pulls(request_id,txn_id,consumer_id,started_mono_ns,status) VALUES(?,?,?,?, 'pending')",
                ("pull-unfinished", "txn-unfinished", "consumer", time.monotonic_ns()),
            )
            engine.store.refresh_digest(conn)
        engine.store.close()
        recovered = Engine(self.db, config=self.config, logger=self.log, start_worker=False)
        pending = recovered.recovery_report["incomplete_pulls_requiring_re_pull"]
        self.assertEqual(pending[0]["request_id"], "pull-unfinished")
        self.assertEqual(pending[0]["status"], "re_pull_required")
        with recovered.store.read() as conn:
            status = conn.execute("SELECT status FROM pulls WHERE request_id='pull-unfinished'").fetchone()["status"]
        self.assertEqual(status, "re_pull_required")
        recovered.store.close()


if __name__ == "__main__":
    unittest.main()
