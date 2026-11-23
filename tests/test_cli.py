from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from drift_diag.cli import main

from .helpers import sample_path


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.db = Path(self.tempdir.name) / "ledger.sqlite3"

    def run_cli(self, args):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def base_run_args(self):
        return [
            "run",
            "--declaration",
            str(sample_path("declaration-v3.json")),
            "--events",
            str(sample_path("events.ndjson")),
            "--db",
            str(self.db),
            "--global-limit",
            "2",
            "--per-workload-quota",
            "1",
            "--batch-events",
            "6",
        ]

    def test_check_command_works_without_tkinter(self):
        code, stdout, stderr = self.run_cli(["check"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertTrue(payload["sqlite3_available"])
        self.assertIn("tkinter_available", payload)
        self.assertEqual(stderr, "")

    def test_declaration_corruption_injection_is_e_format(self):
        code, _, stderr = self.run_cli(self.base_run_args() + ["--inject", "declaration-corrupt"])
        self.assertEqual(code, 2)
        error = json.loads(stderr)
        self.assertEqual(error["code"], "E_FORMAT")
        self.assertEqual(error["field"], "<json>")

    def test_unknown_event_field_injection_is_e_event(self):
        code, _, stderr = self.run_cli(self.base_run_args() + ["--inject", "unknown-event-field"])
        self.assertEqual(code, 3)
        error = json.loads(stderr)
        self.assertEqual(error["code"], "E_EVENT")
        self.assertEqual(error["field"], "field")

    def test_ledger_digest_injection_is_e_snapshot(self):
        code, stdout, stderr = self.run_cli(self.base_run_args())
        self.assertEqual(code, 0, stderr)
        self.assertTrue(json.loads(stdout)["overall_converged"])
        code, _, stderr = self.run_cli(self.base_run_args() + ["--inject", "ledger-digest"])
        self.assertEqual(code, 5)
        error = json.loads(stderr)
        self.assertEqual(error["code"], "E_SNAPSHOT")
        self.assertEqual(error["field"], "content_digest")

    def test_verify_command_and_output_file(self):
        code, _, stderr = self.run_cli(self.base_run_args())
        self.assertEqual(code, 0, stderr)
        verifier = [
            "verify",
            "--declaration",
            str(sample_path("declaration-v3.json")),
            "--events",
            str(sample_path("events.ndjson")),
            "--db",
            str(self.db),
        ]
        code, stdout, stderr = self.run_cli(verifier)
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["batches_checked"], 4)

    def test_tolerance_syntax_is_validated(self):
        code, _, stderr = self.run_cli(self.base_run_args() + ["--tolerance", "bad"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stderr)["code"], "E_FORMAT")

    def test_propagate_command_recomputes_upgrade_and_defaults(self):
        code, stdout, stderr = self.run_cli(
            [
                "propagate",
                "--declaration",
                str(sample_path("declaration-v2.json")),
                "--events",
                str(sample_path("events.ndjson")),
                "--change-set",
                str(sample_path("change-set-v2-upgrade.json")),
            ]
        )
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["propagation"]["affected_fields"]), 18)
        self.assertIn("worker/replicas", payload["propagation"]["affected_decisions"])
        self.assertTrue(payload["propagation"]["unaffected_byte_stable"])

    def test_propagate_referenced_delete_returns_e_ref(self):
        code, _, stderr = self.run_cli(
            [
                "propagate",
                "--declaration",
                str(sample_path("declaration-v3.json")),
                "--events",
                str(sample_path("events.ndjson")),
                "--change-set",
                str(sample_path("change-delete-referenced.json")),
            ]
        )
        self.assertEqual(code, 4)
        error = json.loads(stderr)
        self.assertEqual(error["code"], "E_REF")
        self.assertEqual(error["field"], "parent_ref")
        self.assertEqual(len(error["details"]["referrers"]), 2)


if __name__ == "__main__":
    unittest.main()
