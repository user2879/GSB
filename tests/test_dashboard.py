from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from drift_diag.dashboard import build_dashboard_model, render_dashboard_text
from drift_diag.ledger import LedgerStore
from drift_diag.session import run_session
from drift_diag.ui import ui_environment

from .helpers import config, sample_path


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.db = Path(self.tempdir.name) / "ledger.sqlite3"
        self.session = run_session(
            declaration_path=sample_path("declaration-v3.json"),
            events_path=sample_path("events.ndjson"),
            db_path=self.db,
            defaults_version="defaults-v1",
            config=config(tolerance_num=1, tolerance_den=6, task_delay_ms=1.0),
        )

    def test_render_contains_every_required_section_in_stable_order(self):
        first = render_dashboard_text(self.db)
        second = render_dashboard_text(self.db)
        self.assertEqual(first, second)
        sections = [
            "[NORMALIZED TARGET STATE]",
            "[LATEST FIELD DECISIONS]",
            "[WORKLOAD CONVERGENCE AND ERROR BOUNDS]",
            "[SCHEDULE TRACE]",
            "[LEDGER BATCHES]",
            "[SNAPSHOT REPLACEMENTS]",
        ]
        positions = [first.index(section) for section in sections]
        self.assertEqual(positions, sorted(positions))
        self.assertLess(first.index("workload=api-gateway"), first.index("workload=cache"))
        self.assertLess(first.index("workload=cache"), first.index("workload=worker"))
        self.assertIn("api-gateway/replicas drift=False", first)
        self.assertIn("error_upper_units=0", first)
        self.assertIn("queue_wait_ms=", first)
        self.assertIn("quota_wait_ms=", first)

    def test_read_only_refresh_does_not_change_ledger(self):
        before = build_dashboard_model(self.db)
        text_one = render_dashboard_text(self.db)
        text_two = render_dashboard_text(self.db)
        after = build_dashboard_model(self.db)
        self.assertEqual(before["model_digest"], after["model_digest"])
        self.assertEqual(text_one, text_two)
        with LedgerStore(self.db, read_only=True) as ledger:
            integrity = ledger.verify_integrity()
            self.assertTrue(integrity["ok"])
            self.assertEqual(integrity["batches_checked"], len(self.session.batches))

    def test_ui_environment_is_reported_without_import_failure(self):
        status = ui_environment()
        self.assertIn("tkinter_available", status)
        if not status["tkinter_available"]:
            self.assertIn("error", status)


if __name__ == "__main__":
    unittest.main()
