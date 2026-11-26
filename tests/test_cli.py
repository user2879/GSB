from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from affinity_scheduler.samples import write_sample

from tests.helpers import parse_stdout_json, run_cli


class CliIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sample = write_sample(self.root, port_base=40600)
        self.log = self.root / "logs" / "cli.jsonl"
        self.ledger = self.root / "data" / "cli.sqlite3"
        self.state = self.root / "data" / "cli-state.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _schedule(self) -> dict:
        completed = run_cli(
            [
                "--log-file",
                str(self.log),
                "schedule",
                "--spec",
                self.sample["spec"],
                "--records",
                self.sample["records"],
                "--state",
                str(self.state),
                "--ledger",
                str(self.ledger),
                "--mode",
                "incremental",
            ]
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return parse_stdout_json(completed)

    def test_record_injections_and_structured_logs(self) -> None:
        width = run_cli(
            [
                "--log-file",
                str(self.log),
                "normalize-records",
                "--records",
                self.sample["records"],
                "--spec",
                self.sample["spec"],
                "--inject",
                "v1-width",
            ]
        )
        self.assertEqual(width.returncode, 2)
        self.assertEqual(parse_stdout_json(width)["code"], "E_WIDTH")
        corrupt = run_cli(
            [
                "--log-file",
                str(self.log),
                "normalize-records",
                "--records",
                self.sample["records"],
                "--spec",
                self.sample["spec"],
                "--inject",
                "record-corruption",
            ]
        )
        self.assertEqual(corrupt.returncode, 2)
        self.assertEqual(parse_stdout_json(corrupt)["code"], "E_FORMAT")
        events = [
            json.loads(line)
            for line in self.log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        error_events = [event for event in events if event["code"] in {"E_WIDTH", "E_FORMAT"}]
        self.assertEqual({event["code"] for event in error_events}, {"E_WIDTH", "E_FORMAT"})
        for event in error_events:
            self.assertIsInstance(event["monotonic_ms"], int)

    def test_verify_passes_then_reports_first_e_verify_field(self) -> None:
        self._schedule()
        verified = run_cli(
            [
                "--log-file",
                str(self.log),
                "verify",
                "--ledger",
                str(self.ledger),
                "--spec",
                self.sample["spec"],
                "--records",
                self.sample["records"],
            ]
        )
        self.assertEqual(verified.returncode, 0, verified.stderr)
        self.assertTrue(parse_stdout_json(verified)["ok"])

        connection = sqlite3.connect(self.ledger)
        try:
            row = connection.execute(
                "SELECT conclusion_json FROM ledger_entries WHERE pod_id='pod-0001'"
            ).fetchone()
            conclusion = json.loads(row[0])
            conclusion["total_score"] = conclusion["total_score"] + 1
            connection.execute(
                "UPDATE ledger_entries SET conclusion_json=? WHERE pod_id='pod-0001'",
                (json.dumps(conclusion, sort_keys=True),),
            )
            connection.commit()
        finally:
            connection.close()
        broken = run_cli(
            [
                "--log-file",
                str(self.log),
                "verify",
                "--ledger",
                str(self.ledger),
                "--spec",
                self.sample["spec"],
                "--records",
                self.sample["records"],
            ]
        )
        self.assertEqual(broken.returncode, 2)
        payload = parse_stdout_json(broken)
        self.assertEqual(payload["code"], "E_VERIFY")
        self.assertEqual(payload["pod_id"], "pod-0001")
        self.assertEqual(payload["details"]["field"], "total_score")
        self.assertIn("node_name", payload)

    def test_ledger_digest_injection_reports_e_snapshot(self) -> None:
        self._schedule()
        injected = run_cli(
            [
                "--log-file",
                str(self.log),
                "ledger-sync",
                "--ledger",
                str(self.ledger),
                "--spec",
                self.sample["spec"],
                "--records",
                self.sample["records"],
                "--state",
                str(self.state),
                "--inject-digest",
            ]
        )
        self.assertEqual(injected.returncode, 2)
        self.assertEqual(parse_stdout_json(injected)["code"], "E_SNAPSHOT")


if __name__ == "__main__":
    unittest.main()
