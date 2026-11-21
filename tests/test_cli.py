from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from memdiag.ledger import Ledger
from memdiag.util import read_json, atomic_write_json
from tests.helpers import ROOT, TemporaryDirectoryTestCase


class CliTests(TemporaryDirectoryTestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "memdiag.cli", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=40,
        )

    def test_validate_command_is_machine_readable(self):
        process = self._run(
            "validate",
            "--plan",
            "data/plan.json",
            "--samples",
            "data/samples.json",
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        payload = json.loads(process.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["plan_version"], 1)
        self.assertEqual(payload["layers"], ["branch_a", "branch_b", "head", "merge", "stem"])

    def test_check_incremental_command_passes(self):
        report = self.temp_dir / "check.json"
        process = self._run(
            "check-incremental",
            "--plan",
            "data/plan.json",
            "--samples",
            "data/samples.json",
            "--changes",
            "data/changes_scale.json",
            "--report",
            str(report),
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        payload = json.loads(process.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["invalidated_layers"], ["branch_a", "merge", "head"])
        self.assertTrue(report.is_file())

    def test_run_report_and_snapshot_commands(self):
        db = self.temp_dir / "cli.sqlite3"
        snapshots = self.temp_dir / "snapshots"
        report = self.temp_dir / "report.json"
        process = self._run(
            "run",
            "--plan",
            "data/plan.json",
            "--samples",
            "data/samples.json",
            "--db",
            str(db),
            "--snapshots",
            str(snapshots),
            "--session-id",
            "cli-full",
            "--report",
            str(report),
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        payload = json.loads(process.stdout)
        self.assertEqual(payload["session_id"], "cli-full")
        self.assertEqual(len(payload["layers"]), 5)
        self.assertTrue(report.is_file())
        replay_process = self._run("replay", "--db", str(db), "--session-id", "cli-full")
        self.assertEqual(replay_process.returncode, 0, replay_process.stderr)
        self.assertTrue(json.loads(replay_process.stdout)["ok"])

    def test_oom_is_reported_and_recorded_in_ledger(self):
        plan = read_json(ROOT / "data" / "plan.json")
        plan["memory_budget_mib"] = 1
        plan["layers"][0]["param_count"] = 2_000_000
        plan_path = self.temp_dir / "oom_plan.json"
        atomic_write_json(plan_path, plan)
        db = self.temp_dir / "oom.sqlite3"
        process = self._run(
            "run",
            "--plan",
            str(plan_path),
            "--samples",
            "data/samples.json",
            "--db",
            str(db),
            "--snapshots",
            str(self.temp_dir / "snapshots"),
            "--session-id",
            "oom-session",
            "--no-snapshot",
            "--report",
            str(self.temp_dir / "oom_report.json"),
        )
        self.assertEqual(process.returncode, 4, process.stderr)
        error = json.loads(process.stderr)
        self.assertEqual(error["code"], "E_OOM")
        self.assertGreater(error["needed_blocks"], error["free_blocks"])
        self.assertFalse((self.temp_dir / "oom_report.json").exists())
        ledger = Ledger(db)
        try:
            rows = ledger.connection.execute(
                "SELECT code, details_json FROM ledger_events WHERE session_id=?",
                ("oom-session",),
            ).fetchall()
            self.assertEqual([row["code"] for row in rows], ["E_OOM"])
            details = json.loads(rows[0]["details_json"])
            self.assertEqual(details["needed_blocks"], error["needed_blocks"])
            self.assertEqual(details["free_blocks"], error["free_blocks"])
        finally:
            ledger.close()


if __name__ == "__main__":
    unittest.main()
