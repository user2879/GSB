import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slowquery_diag.engine import DiagnosticEngine  # noqa: E402
from slowquery_diag.errors import ReplayMismatchError  # noqa: E402
from slowquery_diag.ledger import DiagnosticLedger  # noqa: E402
from slowquery_diag.scheduler import ReplayJob  # noqa: E402

EXAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)
DECLARATION = os.path.join(EXAMPLE_DIR, "declaration.json")


class LedgerRecoveryTests(unittest.TestCase):
    def test_reopen_preserves_committed_rows_and_digest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "ledger.db")
            engine = DiagnosticEngine(db_path, scale_exponent=1)
            result = engine.run(DECLARATION)
            digest = engine.ledger.current_digest()
            engine.close()

            reopened = DiagnosticLedger(db_path).open()
            self.assertIsNone(getattr(reopened, "replacement_reason", None))
            self.assertEqual(reopened.current_digest(), digest)
            rows = reopened.load_replay_jobs(result.scheme_version)
            self.assertEqual(rows, [])
            reopened.close()

    def test_unknown_intent_survives_reopen_and_recovery_reruns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "ledger.db")
            catalog_path = os.path.join(tmpdir, "catalog.json")
            engine = DiagnosticEngine(db_path, scale_exponent=1)
            result = engine.run(DECLARATION)
            engine.build_replay_catalog(result.bundle, catalog_path)
            quota = {s.source_id: 1 for s in result.bundle.sources}
            config = {
                "catalog_path": catalog_path,
                "global_concurrency": 2,
                "per_source_quota": quota,
                "socket_timeout_ms": 2000,
            }
            result = engine.run(DECLARATION, replay=True, replay_config=config)
            scheme = engine.scheme_version

            victim = result.bundle.candidates[0]
            source_id = victim.target_sources[0]
            engine.ledger.write_replay_intent(ReplayJob(
                "inflight_candidate", source_id, victim.rewritten_sql
            ))
            with open(catalog_path, encoding="utf-8") as handle:
                catalog = json.load(handle)
            catalog["replay_catalog"].append({
                "candidate_id": "inflight_candidate", "source_id": source_id,
                "est_latency_us": 999, "est_rows_scanned": 7, "stall_ms": 0,
            })
            with open(catalog_path, "w", encoding="utf-8") as handle:
                json.dump(catalog, handle)
            engine.close()

            recovered = DiagnosticLedger(db_path).open()
            unknown_scheme = recovered.latest_scheme_with_unknown_jobs()
            self.assertEqual(unknown_scheme, scheme)
            unknown = recovered.unknown_replay_jobs(unknown_scheme)
            self.assertEqual(unknown[0]["candidate_id"], "inflight_candidate")
            recovered.begin_scheme(unknown_scheme)
            engine2 = DiagnosticEngine(db_path, scale_exponent=1)
            engine2.ledger = recovered
            engine2.scheme_version = unknown_scheme
            rerun = engine2.recover_pending_replays(config)
            self.assertEqual(len(rerun), 1)
            self.assertTrue(rerun[0].ok)
            self.assertEqual(rerun[0].ack["est_latency_us"], 999)
            self.assertEqual(
                recovered.unknown_replay_jobs(unknown_scheme), []
            )
            recovered.close()

    def test_digest_mismatch_replaces_snapshot_and_keeps_backup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "ledger.db")
            engine = DiagnosticEngine(db_path, scale_exponent=1)
            engine.run(DECLARATION)
            engine.close()

            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.execute(
                "UPDATE ledger_meta SET value='tampered' WHERE key='content_digest'"
            )
            conn.commit()
            conn.close()

            reopened = DiagnosticLedger(db_path).open()
            self.assertEqual(reopened.replacement_reason, "content digest mismatch")
            self.assertTrue(os.path.exists(db_path + ".readonly.bak"))
            mode = os.stat(db_path + ".readonly.bak").st_mode & 0o777
            self.assertEqual(mode, 0o444)
            reopened.close()

    def test_format_version_mismatch_replaces_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "ledger.db")
            engine = DiagnosticEngine(db_path, scale_exponent=1)
            engine.run(DECLARATION)
            engine.close()

            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.execute(
                "UPDATE ledger_meta SET value='999' WHERE key='format_version'"
            )
            conn.commit()
            conn.close()

            reopened = DiagnosticLedger(db_path).open()
            self.assertIn("format_version", reopened.replacement_reason)
            reopened.close()

    def test_replay_mismatch_locates_first_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "ledger.db")
            engine = DiagnosticEngine(db_path, scale_exponent=1)
            result = engine.run(DECLARATION)
            with self.assertRaises(ReplayMismatchError) as ctx:
                engine.ledger.verify_replay_against_ledger(
                    result.scheme_version,
                    [{"candidate_id": "ghost", "source_id": "s",
                      "ack": {"candidate_id": "ghost", "source_id": "s",
                              "est_latency_us": 1, "est_rows_scanned": 1}}],
                )
            self.assertEqual(ctx.exception.code, "E_REPLAY")
            self.assertEqual(ctx.exception.field, "existence")
            engine.close()


if __name__ == "__main__":
    unittest.main()
