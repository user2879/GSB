from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from quota_workbench.session import WorkbenchSession
from quota_workbench.snapshot import save_snapshot
from tests.helpers import ROOT, SAMPLES


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "app.py"), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def parse_stdout(completed: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(completed.stdout)


class CliDemoTests(unittest.TestCase):
    def test_specialized_anomaly_codes_and_structured_log(self) -> None:
        with tempfile.TemporaryDirectory(prefix="quota-cli-") as temp:
            log_path = Path(temp) / "structured.ndjson"
            quota = run_cli(
                "--inject", "quota-overflow", "--log", str(log_path)
            )
            self.assertEqual(quota.returncode, 11, quota.stderr)
            self.assertEqual(parse_stdout(quota)["error"]["code"], "E_QUOTA")
            reserve = run_cli(
                "--inject", "reserve-overflow", "--log", str(log_path)
            )
            self.assertEqual(reserve.returncode, 12, reserve.stderr)
            self.assertEqual(parse_stdout(reserve)["error"]["code"], "E_RESERVE")

            snapshot_path = Path(temp) / "snapshot.json"
            session = WorkbenchSession.open(
                SAMPLES / "spec.json", SAMPLES / "normal.ndjson", Path(temp) / "source.sqlite3"
            )
            try:
                session.run_next()
                save_snapshot(session, snapshot_path)
            finally:
                session.close()
            snapshot = run_cli(
                "--inject",
                "snapshot-digest-mismatch",
                "--snapshot",
                str(snapshot_path),
                "--log",
                str(log_path),
            )
            self.assertEqual(snapshot.returncode, 14, snapshot.stderr)
            self.assertEqual(parse_stdout(snapshot)["error"]["code"], "E_SNAPSHOT")
            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [item["error"]["code"] for item in records],
                ["E_QUOTA", "E_RESERVE", "E_SNAPSHOT"],
            )

    def test_cli_run_all_and_unsatisfied_exit_code(self) -> None:
        with tempfile.TemporaryDirectory(prefix="quota-cli-run-") as temp:
            normal = run_cli(
                "--db",
                str(Path(temp) / "normal.sqlite3"),
                "--events",
                str(SAMPLES / "normal.ndjson"),
                "--run-all",
                "--log",
                str(Path(temp) / "log.ndjson"),
            )
            self.assertEqual(normal.returncode, 0, normal.stderr)
            normal_output = parse_stdout(normal)
            self.assertTrue(normal_output["completed"])
            self.assertEqual(normal_output["cursor"], 4)
            self.assertEqual(normal_output["status"], "OK")

            unsat = run_cli(
                "--db",
                str(Path(temp) / "unsat.sqlite3"),
                "--events",
                str(SAMPLES / "unsat.ndjson"),
                "--run-all",
                "--log",
                str(Path(temp) / "log.ndjson"),
            )
            self.assertEqual(unsat.returncode, 13, unsat.stderr)
            unsat_output = parse_stdout(unsat)
            self.assertEqual(unsat_output["status"], "UNSATISFIED")
            self.assertEqual(unsat_output["results"][0]["terminal"]["code"], "E_UNSATISFIED")

    def test_crash_recovery_via_cli(self) -> None:
        with tempfile.TemporaryDirectory(prefix="quota-cli-crash-") as temp:
            db_path = Path(temp) / "crash.sqlite3"
            crashed = run_cli(
                "--db",
                str(db_path),
                "--events",
                str(SAMPLES / "normal.ndjson"),
                "--crash-at-event",
                "1",
                "--log",
                str(Path(temp) / "log.ndjson"),
            )
            self.assertEqual(crashed.returncode, 137, crashed.stderr)
            reopened = run_cli(
                "--db",
                str(db_path),
                "--events",
                str(SAMPLES / "normal.ndjson"),
                "--log",
                str(Path(temp) / "log.ndjson"),
            )
            self.assertEqual(reopened.returncode, 0, reopened.stderr)
            output = parse_stdout(reopened)
            self.assertEqual(output["recovery"]["status"], "UNKNOWN_REQUIRES_RERUN")
            self.assertEqual(output["recovery"]["unknown_event_ids"], [1])
            self.assertEqual(output["cursor"], 0)

    def test_ui_module_is_importable_without_tk_runtime(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", "from quota_workbench.ui import launch_ui; print(callable(launch_ui))"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "True")

    def test_demo_is_repeatable(self) -> None:
        results_path = ROOT / "demo_output" / "results.json"
        first = subprocess.run(
            [sys.executable, str(ROOT / "demo.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        first_bytes = results_path.read_bytes()
        first_result = json.loads(first_bytes)
        second = subprocess.run(
            [sys.executable, str(ROOT / "demo.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        second_bytes = results_path.read_bytes()
        second_result = json.loads(second_bytes)
        self.assertEqual(first_result["status"], "PASS")
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first_result["summary_digest"], second_result["summary_digest"])
        self.assertEqual(
            [item["code"] for item in first_result["anomaly_checks"]["checks"]],
            ["E_QUOTA", "E_RESERVE", "E_SNAPSHOT"],
        )


if __name__ == "__main__":
    unittest.main()
