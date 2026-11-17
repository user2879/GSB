import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from chaoskit.errors import LedgerError, ReplayError
from chaoskit.ledger import LEDGER_FILENAME, Ledger, open_or_replace_ledger
from chaoskit.replay import verify_identity, verify_timeline_entries
from chaoskit.spec import load_spec
from chaoskit.timeline import build_timeline, timeline_digest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample_experiment.json"
DRIFT = ROOT / "examples" / "sample_experiment_drift.json"


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tmp.name)
        self.spec = load_spec(str(SAMPLE))
        self.entries = build_timeline(self.spec)

    def tearDown(self):
        self.tmp.cleanup()

    def test_initialize_and_digest_roundtrip(self):
        ledger, event = open_or_replace_ledger(self.workdir / LEDGER_FILENAME, self.spec, self.entries)
        self.assertFalse(event["replaced"])
        row = ledger.experiment_row()
        self.assertEqual(row["timeline_digest"], timeline_digest(self.entries))
        self.assertIsNone(ledger.stored_content_digest())
        ledger.mark_running(0)
        self.assertIsNotNone(ledger.stored_content_digest())
        ledger.verify_content_digest(ledger.stored_content_digest())
        ledger.close()

    def test_unknown_reclassification_on_reopen(self):
        ledger, _ = open_or_replace_ledger(self.workdir / LEDGER_FILENAME, self.spec, self.entries)
        ledger.mark_running(1)
        ledger.close()

        reopened, event = open_or_replace_ledger(self.workdir / LEDGER_FILENAME, self.spec, self.entries)
        self.assertEqual(event["unknown_indexes"], [1])
        report = reopened.recovery_report(self.spec)
        self.assertIn(1, report["unknown_indexes"])
        self.assertEqual(report["recovered_committed"], 0)
        self.assertFalse(report["exit_report_promised"])
        reopened.close()

    def test_tampered_content_raises_ledger_error(self):
        ledger, _ = open_or_replace_ledger(self.workdir / LEDGER_FILENAME, self.spec, self.entries)
        ledger.mark_running(0)
        ledger.close()
        conn = sqlite3.connect(str(self.workdir / LEDGER_FILENAME))
        conn.execute("UPDATE injections SET target_id='tampered' WHERE idx=0")
        conn.commit()
        conn.close()
        with self.assertRaises(LedgerError) as ctx:
            open_or_replace_ledger(self.workdir / LEDGER_FILENAME, self.spec, self.entries)
        self.assertEqual(ctx.exception.code, "E_LEDGER")

    def test_whole_snapshot_replacement_with_readonly_backup(self):
        ledger, _ = open_or_replace_ledger(self.workdir / LEDGER_FILENAME, self.spec, self.entries)
        ledger.mark_running(0)
        ledger.close()

        drift = load_spec(str(DRIFT))
        drift_entries = build_timeline(drift)
        db_path = self.workdir / LEDGER_FILENAME
        replaced, event = open_or_replace_ledger(db_path, drift, drift_entries)
        self.assertTrue(event["replaced"])
        self.assertEqual(event["skipped_injections"], 0)
        backup = Path(event["backup_path"])
        self.assertTrue(backup.exists())
        mode = backup.stat().st_mode
        self.assertEqual(mode & 0o222, 0, "backup must be read-only")

        report = replaced.recovery_report(drift)
        self.assertEqual(len(report["replacement_events"]), 1)
        stored = replaced.experiment_row()
        self.assertEqual(stored["seed"], drift.seed)
        replaced.close()

    def test_replay_seed_and_structural_mismatch(self):
        drift = load_spec(str(DRIFT))
        ledger, _ = open_or_replace_ledger(self.workdir / LEDGER_FILENAME, self.spec, self.entries)
        with self.assertRaises(ReplayError) as seed_ctx:
            verify_identity(drift, ledger)
        self.assertEqual(seed_ctx.exception.code, "E_REPLAY")
        self.assertEqual(seed_ctx.exception.field, "seed")

        same_seed = replace(self.spec, seed=self.spec.seed)
        # structural mismatch: drift changed a duration while identity already
        # differs; build a spec that keeps seed but alters a duration directly
        drifted_injections = [
            replace(item, duration_ms=item.duration_ms + 50) if item.injection_id == "inj-orders-delay"
            else item
            for item in same_seed.injections
        ]
        structural_spec = replace(same_seed, injections=drifted_injections)
        with self.assertRaises(ReplayError) as struct_ctx:
            verify_timeline_entries(structural_spec, ledger)
        self.assertEqual(struct_ctx.exception.field, "duration_ms")
        ledger.close()


if __name__ == "__main__":
    unittest.main()
