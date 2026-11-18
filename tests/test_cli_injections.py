from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from quota_app.cli import main
from quota_app.injections import run_injection
from quota_app.ledger import Ledger

from .helpers import make_legacy_workspace, write_events


class CliInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.configs = make_legacy_workspace(self.root, checkpoint=True)
        self.events = write_events(self.root / "events.ndjson")
        self.ledger = self.root / "ledger.sqlite3"
        self.log = self.root / "logs.jsonl"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run(self, *arguments: str) -> tuple[int, dict]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(list(arguments))
        return code, json.loads(output.getvalue())

    def test_injection_functions_return_required_codes(self) -> None:
        expected = {
            "corrupt-config": "E_FORMAT",
            "missing-checkpoint": "E_UPGRADE",
            "timestamp-regression": "E_ORDER",
        }
        for kind, code in expected.items():
            with self.subTest(kind=kind):
                result = run_injection(kind, config_dir=self.configs, events_path=self.events)
                self.assertTrue(result.passed)
                self.assertEqual(result.observed_code, code)

    def test_cli_run_injection_flags(self) -> None:
        code, payload = self._run(
            "run",
            "--inject-timestamp-regression",
            "--config-dir",
            str(self.configs),
            "--events",
            str(self.events),
            "--ledger",
            str(self.ledger),
            "--log",
            str(self.log),
        )
        self.assertEqual(code, 0)
        self.assertEqual(payload["code"], "E_ORDER")
        self.assertTrue(payload["passed"])

    def test_cli_inject_subcommand(self) -> None:
        code, payload = self._run(
            "inject",
            "missing-checkpoint",
            "--config-dir",
            str(self.configs),
            "--events",
            str(self.events),
            "--ledger",
            str(self.ledger),
            "--log",
            str(self.log),
        )
        self.assertEqual(code, 0)
        self.assertEqual(payload["code"], "E_UPGRADE")
        entries = [
            json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(entries[-1]["code"], "E_UPGRADE")
        self.assertEqual(entries[-1]["operation"], "inject.missing-checkpoint")

    def test_full_cli_run_and_recover(self) -> None:
        code, payload = self._run(
            "run",
            "--config-dir",
            str(self.configs),
            "--events",
            str(self.events),
            "--ledger",
            str(self.ledger),
            "--log",
            str(self.log),
            "--batch-size",
            "3",
        )
        self.assertEqual(code, 0)
        self.assertEqual(payload["code"], "OK")
        self.assertEqual(payload["ingest"]["judgments_written"], 10)
        self.assertEqual(
            payload["ingest"]["decisions"],
            {"ALLOW": 6, "RATE_LIMITED": 2, "OVER_QUOTA": 2},
        )
        self.assertTrue(payload["replay"]["ok"])
        self.assertEqual(len(payload["view"]["judgments"]), 10)

        recover_code, recover_payload = self._run(
            "recover",
            "--config-dir",
            str(self.configs),
            "--events",
            str(self.events),
            "--ledger",
            str(self.ledger),
            "--log",
            str(self.log),
        )
        self.assertEqual(recover_code, 0)
        self.assertEqual(recover_payload["code"], "OK")
        self.assertTrue(recover_payload["replay"]["ok"])
        self.assertFalse(recover_payload["recovery"]["rerun_required"])

    def test_validate_after_upgrade_is_clean(self) -> None:
        code, payload = self._run(
            "validate",
            "--config-dir",
            str(self.configs),
            "--events",
            str(self.events),
            "--ledger",
            str(self.ledger),
            "--log",
            str(self.log),
        )
        self.assertEqual(code, 0)
        self.assertEqual(payload["active_revision"], 3)
        self.assertEqual(payload["format_version"], 3)
        self.assertEqual(len(payload["volumes"]), 4)

    def test_cli_edit_replaces_view_and_optional_ingest(self) -> None:
        code, payload = self._run(
            "edit",
            "--config-dir",
            str(self.configs),
            "--events",
            str(self.events),
            "--ledger",
            str(self.ledger),
            "--log",
            str(self.log),
            "--volume",
            "child",
            "--capacity-bytes",
            "800",
            "--ingest",
            "--batch-size",
            "3",
        )
        self.assertEqual(code, 0)
        self.assertEqual(payload["code"], "OK")
        self.assertEqual(payload["active_revision"], 4)
        self.assertTrue(payload["config_replaced_wholesale"])
        self.assertEqual(payload["replacement"]["affected_volumes"], ["child"])
        self.assertIn("capacity_bytes", payload["replacement"]["changed_fields"][0]["fields"])
        self.assertEqual(payload["ingest"]["judgments_written"], 10)
        self.assertTrue(payload["replay"]["ok"])
        with Ledger(self.ledger) as ledger:
            versions = ledger.list_config_versions()
            self.assertEqual(versions[-1]["revision"], 4)
            self.assertEqual(versions[-2]["stale"], 1)
            self.assertEqual(len(ledger.load_current_judgments(4)), 10)


if __name__ == "__main__":
    unittest.main()
