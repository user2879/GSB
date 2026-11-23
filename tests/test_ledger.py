from __future__ import annotations

import json
import sqlite3
import threading
import unittest
from pathlib import Path

from config_diag.canonical import canonical_json
from config_diag.clock import ManualClock
from config_diag.engine import ConfigEngine
from config_diag.model import ChangeEdit, ChangeOperation
from config_diag.errors import DiagnosticError
from config_diag.recovery import recover_and_replay
from config_diag.structured_log import JsonlLogger, NullLogger
from config_diag.validation import load_bundle

from .support import SAMPLES, local_temp


class LedgerTests(unittest.TestCase):
    def test_repeated_operation_is_idempotent_and_has_no_duplicate_row(self) -> None:
        spec, clients, changes = load_bundle(SAMPLES)
        with local_temp("ledger-idem-") as tmp:
            engine = ConfigEngine(tmp, logger=NullLogger(), clock=ManualClock(1_000_000_000))
            try:
                engine.initialize(spec, clients, now_ns=1_000_000_000)
                first = engine.apply_operation(changes.operations[0], now_ns=1_100_000_000)
                rows_before = engine.ledger.raw_key_rows()
                meta_before = engine.ledger.meta()
                journal_before = engine.ledger.journal_rows()
                second = engine.apply_operation(changes.operations[0], now_ns=1_200_000_000)
                rows_after = engine.ledger.raw_key_rows()
                meta_after = engine.ledger.meta()
                journal_after = engine.ledger.journal_rows()
                self.assertFalse(first["idempotent"])
                self.assertTrue(second["idempotent"])
                self.assertEqual(rows_before, rows_after)
                self.assertEqual(len(rows_after), 11)
                self.assertEqual(meta_before["ledger_version"], meta_after["ledger_version"])
                self.assertEqual(len(journal_before), len(journal_after))
                self.assertEqual(canonical_json(rows_before), canonical_json(rows_after))
            finally:
                engine.close()

    def test_new_operation_with_same_key_and_version_is_payload_idempotent(self) -> None:
        spec, clients, changes = load_bundle(SAMPLES)
        with local_temp("ledger-same-version-") as tmp:
            engine = ConfigEngine(tmp, logger=NullLogger(), clock=ManualClock(1_000_000_000))
            try:
                engine.initialize(spec, clients, now_ns=1_000_000_000)
                engine.apply_operation(changes.operations[0], now_ns=1_100_000_000)
                rows_before = engine.ledger.raw_key_rows()
                meta_before = engine.ledger.meta()
                journal_before = engine.ledger.journal_rows()
                retry = ChangeOperation(
                    operation_id="retry-same-key-version",
                    sequence=99,
                    description="same key/version/payload retry",
                    changes=(ChangeEdit(key="pricing.base_fee", version=2, value=1500),),
                )
                result = engine.apply_operation(retry, now_ns=1_200_000_000)
                self.assertTrue(result["idempotent"])
                self.assertEqual(result["idempotency_reason"], "same_key_version_same_payload")
                self.assertEqual(engine.ledger.raw_key_rows(), rows_before)
                self.assertEqual(engine.ledger.meta()["ledger_version"], meta_before["ledger_version"])
                self.assertEqual(len(engine.ledger.journal_rows()), len(journal_before))
                self.assertEqual(len(engine.ledger.raw_key_rows()), 11)
            finally:
                engine.close()

    def test_uncommitted_transaction_is_invisible_and_commit_is_whole(self) -> None:
        spec, clients, changes = load_bundle(SAMPLES)
        with local_temp("ledger-atomic-") as tmp:
            initializer = ConfigEngine(tmp, logger=NullLogger(), clock=ManualClock(1_000_000_000))
            try:
                initializer.initialize(spec, clients, now_ns=1_000_000_000)
            finally:
                initializer.close()

            entered = threading.Event()
            release = threading.Event()
            failures: list[BaseException] = []

            def worker() -> None:
                worker_engine = ConfigEngine(tmp, logger=NullLogger(), clock=ManualClock(1_100_000_000))
                try:
                    worker_engine.apply_operation(
                        changes.operations[0],
                        now_ns=1_100_000_000,
                        before_commit=lambda: (entered.set(), release.wait(timeout=10)),
                    )
                except BaseException as exc:  # surfaced in the main test thread
                    failures.append(exc)
                finally:
                    worker_engine.close()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            self.assertTrue(entered.wait(timeout=10))

            reader = sqlite3.connect(str(tmp / "runtime" / "ledger.sqlite3"), timeout=5)
            reader.row_factory = sqlite3.Row
            try:
                old_key = reader.execute(
                    "SELECT value, version, content_digest FROM ledger_keys WHERE key = 'pricing.base_fee'"
                ).fetchone()
                old_meta = reader.execute(
                    "SELECT ledger_version, content_digest FROM ledger_meta WHERE id = 1"
                ).fetchone()
                pending = reader.execute(
                    "SELECT status FROM change_journal ORDER BY journal_id DESC LIMIT 1"
                ).fetchone()
                self.assertEqual(tuple(old_key), (1200, 1, old_key["content_digest"]))
                self.assertEqual(old_meta["ledger_version"], 1)
                self.assertEqual(pending["status"], "pending")
            finally:
                reader.close()

            release.set()
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
            if failures:
                raise failures[0]

            reader = sqlite3.connect(str(tmp / "runtime" / "ledger.sqlite3"), timeout=5)
            reader.row_factory = sqlite3.Row
            try:
                new_key = reader.execute(
                    "SELECT value, version FROM ledger_keys WHERE key = 'pricing.base_fee'"
                ).fetchone()
                new_meta = reader.execute("SELECT ledger_version FROM ledger_meta WHERE id = 1").fetchone()
                statuses = [
                    row["status"]
                    for row in reader.execute("SELECT status FROM change_journal ORDER BY journal_id")
                ]
                self.assertEqual(tuple(new_key), (1500, 2))
                self.assertEqual(new_meta["ledger_version"], 2)
                self.assertEqual(statuses, ["committed"])
            finally:
                reader.close()

    def test_replay_detects_tampered_invalidation_trace(self) -> None:
        spec, clients, changes = load_bundle(SAMPLES)
        with local_temp("ledger-replay-trace-") as tmp:
            logger = JsonlLogger(tmp / "logs" / "diagnostics.jsonl")
            engine = ConfigEngine(tmp, logger=logger, clock=ManualClock(1_000_000_000))
            try:
                engine.initialize(spec, clients, now_ns=1_000_000_000)
                engine.apply_operation(changes.operations[0], now_ns=1_100_000_000)
            finally:
                engine.close()
            db_path = tmp / "runtime" / "ledger.sqlite3"
            conn = sqlite3.connect(str(db_path))
            try:
                row = conn.execute("SELECT journal_id, trace_json FROM change_journal LIMIT 1").fetchone()
                trace = json.loads(row[1])
                trace["affected"] = trace["affected"][:-1]
                trace["propagation"] = trace["propagation"][:-1]
                conn.execute(
                    "UPDATE change_journal SET trace_json = ? WHERE journal_id = ?",
                    (json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":")), row[0]),
                )
                conn.commit()
            finally:
                conn.close()
            reopened = ConfigEngine(tmp, logger=logger, clock=ManualClock(2_000_000_000))
            try:
                with self.assertRaises(DiagnosticError) as raised:
                    recover_and_replay(reopened, tmp, logger=logger)
            finally:
                reopened.close()
            self.assertEqual(raised.exception.code, "E_REPLAY")
            self.assertEqual(raised.exception.field, "invalidation_trace")
            self.assertEqual(raised.exception.key, "pricing.base_fee")

    def test_snapshot_mismatch_creates_readonly_backup_and_replaces_whole_state(self) -> None:
        spec, clients, _ = load_bundle(SAMPLES)
        with local_temp("ledger-snapshot-") as tmp:
            logger = JsonlLogger(tmp / "logs" / "diagnostics.jsonl")
            engine = ConfigEngine(tmp, logger=logger, clock=ManualClock(1_000_000_000))
            try:
                engine.initialize(spec, clients, now_ns=1_000_000_000)
                good_digest = engine.ledger.meta()["content_digest"]
            finally:
                engine.close()
            db_path = tmp / "runtime" / "ledger.sqlite3"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute("UPDATE ledger_meta SET content_digest = ? WHERE id = 1", ("f" * 64,))
                conn.commit()
            finally:
                conn.close()

            reopened = ConfigEngine(tmp, logger=logger, clock=ManualClock(1_000_000_000))
            try:
                events = [item for item in reopened.open_events if item["code"] == "E_SNAPSHOT"]
                self.assertEqual(len(events), 1)
                event = events[0]
                backup = Path(event["backup"])
                self.assertTrue(backup.exists())
                self.assertEqual(backup.stat().st_mode & 0o222, 0)
                self.assertEqual(event["mutation_mode"], "whole_snapshot_replace")
                self.assertEqual(reopened.ledger.meta()["content_digest"], good_digest)
                self.assertTrue(all(record["status"] == "active" for record in reopened.ledger.records().values()))
            finally:
                reopened.close()
            log_lines = [
                json.loads(line)
                for line in logger.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(item["code"] == "E_SNAPSHOT" for item in log_lines))

    def test_format_version_mismatch_is_also_snapshot_repaired(self) -> None:
        spec, clients, _ = load_bundle(SAMPLES)
        with local_temp("ledger-format-") as tmp:
            logger = JsonlLogger(tmp / "logs" / "diagnostics.jsonl")
            engine = ConfigEngine(tmp, logger=logger, clock=ManualClock(1_000_000_000))
            try:
                engine.initialize(spec, clients, now_ns=1_000_000_000)
            finally:
                engine.close()
            db_path = tmp / "runtime" / "ledger.sqlite3"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute("UPDATE ledger_meta SET format_version = 99 WHERE id = 1")
                conn.commit()
            finally:
                conn.close()
            reopened = ConfigEngine(tmp, logger=logger, clock=ManualClock(1_000_000_000))
            try:
                events = [item for item in reopened.open_events if item["code"] == "E_SNAPSHOT"]
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["mismatch"]["reason"], "format_version_mismatch")
                self.assertEqual(reopened.ledger.meta()["format_version"], 1)
                self.assertEqual(events[0]["mutation_mode"], "whole_snapshot_replace")
            finally:
                reopened.close()

    def test_replay_detects_first_tampered_key_as_e_replay(self) -> None:
        spec, clients, changes = load_bundle(SAMPLES)
        with local_temp("ledger-replay-tamper-") as tmp:
            logger = JsonlLogger(tmp / "logs" / "diagnostics.jsonl")
            engine = ConfigEngine(tmp, logger=logger, clock=ManualClock(1_000_000_000))
            try:
                engine.initialize(spec, clients, now_ns=1_000_000_000)
                engine.apply_operation(changes.operations[0], now_ns=1_100_000_000)
                engine.apply_operation(changes.operations[1], now_ns=1_200_000_000)
                # Keep the ledger self-consistent so this is a replay mismatch,
                # not a snapshot digest alarm.
                engine.debug_tamper_record("pricing.net_fee", delta=1)
                with self.assertRaises(DiagnosticError) as raised:
                    recover_and_replay(engine, tmp, logger=logger)
            finally:
                engine.close()
            exc = raised.exception
            self.assertEqual(exc.code, "E_REPLAY")
            self.assertEqual(exc.key, "pricing.net_fee")
            self.assertEqual(exc.field, "value")
            self.assertEqual(exc.context["ledger"], 1251)
            self.assertEqual(exc.context["replay"], 1250)
            log_lines = [
                json.loads(line)
                for line in logger.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(item["code"] == "E_REPLAY" for item in log_lines))


if __name__ == "__main__":
    unittest.main()
