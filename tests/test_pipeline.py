import json
import shutil
import tempfile
import unittest
from pathlib import Path

from cfgdist.config.grid import Grid
from cfgdist.config.manifest import load_manifest, load_registry
from cfgdist.persistence.database import Database
from cfgdist.persistence.snapshot import SnapshotStore
from cfgdist.runtime.replay import verify_snapshot_replay
from cfgdist.runtime.session import Session
from cfgdist.runtime.view import ViewRow, verify_view, build_view
from cfgdist.util.logging_util import StructuredLogger

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cfgdist-it-"))
        self.data = self.tmp / "data"
        self.logs = self.tmp / "logs"
        self.logger = StructuredLogger(self.logs / "it.log")
        self.registry = load_registry(SAMPLES / "registry.json")
        self.manifest = load_manifest(SAMPLES / "manifest.json", self.registry)
        self.grid = Grid.from_q(2)
        self.db = Database(self.data / "it.sqlite", self.logger)
        self.store = SnapshotStore(self.db, self.data / "snapshots", self.logger)

    def tearDown(self):
        self.db.close()
        self.logger.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_session_converges_persists_and_replays(self):
        session = Session(
            "it-session", self.manifest, self.registry, self.grid,
            self.db, self.store, self.logger,
            concurrency=2, max_attempts=2, log_dir=str(self.logs),
            operation_timeout_ms=8000,
        )
        previous = None
        for manifest_version in self.manifest.versions:
            result = session.run_version(manifest_version)
            self.assertTrue(result.convergence.converged)
            self.assertEqual(result.state, "PERSISTED")
            self.assertLessEqual(
                result.scheduler.observed_max_inflight, 2,
                "scheduler must never exceed the global concurrency cap",
            )
            snapshot = self.store.load("it-session", manifest_version.version)
            verify_snapshot_replay(
                self.manifest, self.registry, self.grid, snapshot, previous
            )
            previous = manifest_version

        # Every snapshot file exists on disk.
        for version in (1, 2, 3):
            self.assertTrue((self.data / "snapshots" / f"it-session-v{version}.json").exists())

    def test_view_matches_authoritative_and_flags_staleness(self):
        session = Session(
            "it-view", self.manifest, self.registry, self.grid,
            self.db, self.store, self.logger,
            concurrency=2, max_attempts=2, log_dir=str(self.logs),
            operation_timeout_ms=8000,
        )
        result = session.run_version(self.manifest.versions[0])
        rows = build_view(
            self.manifest.versions[0], result.compiled, result.convergence
        )
        # Identical authoritative view passes byte comparison.
        verify_view(rows, rows, self.manifest.versions[0].version)

        # A row citing an older version is stale and must not trigger E_VIEW.
        stale = [
            ViewRow(r.key, r.version, r.derived_summary, r.verdict, True,
                    r.source_version - 1)
            for r in rows
        ]
        verify_view(stale, rows, self.manifest.versions[0].version)

    def test_receipt_attempts_recorded_with_request_metadata(self):
        session = Session(
            "it-meta", self.manifest, self.registry, self.grid,
            self.db, self.store, self.logger,
            concurrency=1, max_attempts=2, log_dir=str(self.logs),
            operation_timeout_ms=8000,
        )
        result = session.run_version(self.manifest.versions[0])
        for record in result.scheduler.records:
            self.assertTrue(record.request_id.startswith("r1-"))
            self.assertEqual(record.effective_version, 1)
            self.assertEqual(record.final_outcome, "ok")
            for attempt in record.attempts:
                self.assertGreaterEqual(attempt.ended_wait_ms, attempt.started_wait_ms)
                self.assertGreaterEqual(attempt.ended_send_ms, attempt.started_send_ms)


if __name__ == "__main__":
    unittest.main()
