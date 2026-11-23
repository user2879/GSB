from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gray_merge_lab.viewmodel import DiagnosticSession, SessionPaths


class ViewModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = Path(__file__).resolve().parent.parent / "gray_merge_lab" / "data"
        self.paths = SessionPaths(
            self.data / "backends.json",
            self.data / "rules.json",
            self.data / "samples.ndjson",
            self.root,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_quick_view_and_authority_are_separate(self):
        session = DiagnosticSession(self.paths).initialize()
        try:
            self.assertIsNone(session.snapshot_error)
            self.assertEqual(len(session.quick_rows()), 2)
            self.assertEqual(len(session.preview_batches()), 8)
            self.assertTrue(all(row["source"] == "snapshot" for row in session.quick_rows()))
            outcome = session.recompute_authoritative()
            self.assertTrue(outcome["ok"])
            self.assertEqual(outcome["completed_batches"], 8)
        finally:
            session.close()

    def test_checkpoint_recovery_replays_remaining(self):
        session = DiagnosticSession(self.paths).initialize()
        try:
            self.assertTrue(self.paths.checkpoint.exists())
            outcome = session.recover_from_checkpoint()
            self.assertTrue(outcome["ok"])
            self.assertEqual(outcome["source"], "checkpoint")
            self.assertEqual(outcome["resumed_after_batch_index"], 4)
            self.assertEqual(outcome["replayed_batches"], 4)
        finally:
            session.close()

    def test_hot_update_and_anomaly_actions(self):
        session = DiagnosticSession(self.paths).initialize()
        try:
            result = session.hot_update_and_reinfer()
            self.assertEqual(result["backend_param_version"], 2)
            self.assertEqual(result["receipt_count"], 4)
            self.assertTrue(session.recompute_authoritative()["ok"])
            anomalies = session.inject_anomalies()
            self.assertTrue(all(item["ok"] for item in anomalies))
        finally:
            session.close()

    def test_reopen_after_hot_update_requires_snapshot_replacement(self):
        session = DiagnosticSession(self.paths).initialize()
        try:
            session.hot_update_and_reinfer()
        finally:
            session.close()
        reopened = DiagnosticSession(self.paths).initialize()
        try:
            self.assertIsNotNone(reopened.snapshot_error)
            self.assertEqual(reopened.snapshot_error["error"], "E_SNAPSHOT")
            self.assertTrue(any(
                item["field"] == "manifest.backend_set_version"
                for item in reopened.snapshot_error["details"]["differences"]
            ))
            self.assertTrue(reopened.recompute_authoritative()["ok"])
        finally:
            reopened.close()

    def test_reset_rebuilds_session(self):
        session = DiagnosticSession(self.paths).initialize()
        try:
            same = session.reset()
            self.assertIs(same, session)
            self.assertEqual(len(session.preview_batches()), 8)
            self.assertTrue(session.recompute_authoritative()["ok"])
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()

