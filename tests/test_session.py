from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from drift_diag.canonical import canonical_json
from drift_diag.errors import DiagnosticError
from drift_diag.events import load_events
from drift_diag.ledger import LedgerStore
from drift_diag.replay import verify_replay
from drift_diag.session import run_session

from .helpers import ROOT, config, sample_path, write_ndjson


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.db = Path(self.tempdir.name) / "ledger.sqlite3"

    def run_v3(self, events: str = "events.ndjson", **config_overrides):
        return run_session(
            declaration_path=sample_path("declaration-v3.json"),
            events_path=sample_path(events),
            db_path=self.db,
            defaults_version="defaults-v1",
            config=config(**config_overrides),
        )

    def test_full_session_converges_persists_batches_and_replays(self):
        session = self.run_v3(task_delay_ms=1.0)
        self.assertTrue(session.overall_converged)
        self.assertTrue(session.replay_verified)
        self.assertEqual(session.latest_batch["event_cursor"], 18)
        self.assertEqual(len(session.batches), 6)
        self.assertEqual(session.recovered_unknown, [])
        self.assertTrue(session.snapshot_install["replaced"])
        self.assertTrue(all(not decision["drift"] for decision in session.latest_batch["decisions"]))
        self.assertEqual(session.replay_report["decisions_checked"], 108)
        with LedgerStore(self.db, read_only=True) as ledger:
            self.assertEqual(ledger.verify_integrity()["batches_checked"], 6)

    def test_reopening_is_idempotent_and_does_not_duplicate_batches(self):
        first = self.run_v3()
        first_decisions = canonical_json(first.latest_batch["decisions"])
        second = self.run_v3()
        self.assertFalse(second.snapshot_install["replaced"])
        self.assertEqual(len(second.batches), len(first.batches))
        self.assertEqual(canonical_json(second.latest_batch["decisions"]), first_decisions)
        self.assertTrue(second.replay_verified)

    def test_boundary_and_divergent_samples(self):
        boundary = self.run_v3("events-boundary.ndjson", tolerance_num=1, tolerance_den=6)
        cache = boundary.latest_batch["workload_metrics"]["workloads"]["cache"]
        self.assertTrue(boundary.overall_converged)
        self.assertEqual(cache["drift_fields"], ["probe_ms"])
        self.assertTrue(cache["boundary_sensitive"])
        self.assertTrue(cache["error_affected"])

        self.db.unlink()
        divergent = self.run_v3("events-diverge.ndjson", tolerance_num=1, tolerance_den=6)
        cache = divergent.latest_batch["workload_metrics"]["workloads"]["cache"]
        self.assertFalse(divergent.overall_converged)
        self.assertEqual(cache["drift_fields"], ["resource_limit", "probe_ms"])
        self.assertFalse(cache["converged"])

    def test_forced_termination_recovers_only_committed_batches(self):
        command = [
            sys.executable,
            "-m",
            "drift_diag",
            "run",
            "--declaration",
            str(sample_path("declaration-v3.json")),
            "--events",
            str(sample_path("events.ndjson")),
            "--db",
            str(self.db),
            "--defaults-version",
            "defaults-v1",
            "--global-limit",
            "3",
            "--per-workload-quota",
            "1",
            "--batch-events",
            "4",
            "--crash-after-attempt",
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 91)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")
        with LedgerStore(self.db) as ledger:
            self.assertEqual(ledger.read_batches(), [])
            inflight = ledger.conn.execute("SELECT COUNT(*) AS n FROM inflight").fetchone()["n"]
            self.assertEqual(inflight, 1)
        recovered = self.run_v3()
        self.assertEqual({item["workload"] for item in recovered.recovered_unknown}, {"api-gateway", "worker", "cache"})
        self.assertEqual(recovered.latest_batch["event_cursor"], 18)
        self.assertEqual(recovered.ledger_summary["unknown_workloads"], [])

    def test_replay_locates_first_changed_observed_field(self):
        session = self.run_v3()
        raw_events = [
            json.loads(line)
            for line in sample_path("events.ndjson").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        raw_events[-1]["value"] = "other-parent"
        changed_events_path = Path(self.tempdir.name) / "changed.ndjson"
        write_ndjson(changed_events_path, raw_events)
        changed_events = load_events(changed_events_path, session.target)
        with LedgerStore(self.db, read_only=True) as ledger:
            with self.assertRaises(DiagnosticError) as context:
                verify_replay(ledger, session.target, changed_events, config())
        error = context.exception
        self.assertEqual(error.code, "E_REPLAY")
        self.assertEqual(error.item, "cache")
        self.assertEqual(error.field, "parent_ref")
        self.assertEqual(error.details["category"], "observed")
        self.assertEqual(error.details["event_cursor"], 18)

    def test_declaration_evolution_replaces_snapshot(self):
        self.run_v3()
        with LedgerStore(self.db) as ledger:
            batch_count_before = len(ledger.read_batches())
        session = run_session(
            declaration_path=sample_path("declaration-v2.json"),
            events_path=sample_path("events.ndjson"),
            db_path=self.db,
            defaults_version="defaults-v1",
            config=config(),
        )
        self.assertTrue(session.snapshot_install["replaced"])
        self.assertIn("declaration_format_version_changed", session.snapshot_install["reasons"])
        self.assertIsNotNone(session.snapshot_install["backup_path"])
        self.assertEqual(len(session.batches), batch_count_before)
        self.assertEqual(session.target.declaration_format_versions, (2,))


if __name__ == "__main__":
    unittest.main()
