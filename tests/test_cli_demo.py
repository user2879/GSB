from __future__ import annotations

import json
import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from isolation_lab.errors import BusinessError


ROOT = Path(__file__).resolve().parent.parent


def cli(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "isolation_lab.cli", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


class CliAndDemoTests(unittest.TestCase):
    def test_cli_run_snapshot_replay_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "lab.sqlite"
            snapshots = Path(tmp) / "snapshots.sqlite"
            run = cli(
                "run",
                "--script",
                "samples/current-v3.json",
                "--db",
                str(db),
                "--snapshots",
                str(snapshots),
                "--snapshot-id",
                "cli-normal",
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("outcome=COMMITTED", run.stdout)
            self.assertIn("data_version=3", run.stdout)
            replay = cli(
                "snapshot-replay",
                "--snapshots",
                str(snapshots),
                "--snapshot-id",
                "cli-normal",
            )
            self.assertEqual(replay.returncode, 0, replay.stderr)
            self.assertIn("reproducible=true", replay.stdout)
            ledger = cli("ledger", "--db", str(db), "--json")
            self.assertEqual(ledger.returncode, 0, ledger.stderr)
            payload = json.loads(ledger.stdout)
            self.assertGreater(len(payload), 0)
            self.assertEqual(payload[0]["source_version"], 3)
            self.assertEqual(payload[0]["trace_event_seq"], 1)

    def test_cli_validation_error_has_business_code_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "script_id": "bad",
                        "initial_rows": {},
                        "sessions": [
                            {
                                "session_id": "S1",
                                "operations": [
                                    {"op_no": 1, "type": "write", "row_key": "r"}
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            completed = cli("validate", "--script", str(bad), "--json")
            self.assertEqual(completed.returncode, 2)
            error = json.loads(completed.stderr)["error"]
            self.assertEqual(error["code"], "E_SCRIPT")
            self.assertEqual(error["session_id"], "S1")
            self.assertIn("$.sessions[0].operations[0].value", error["path"])

    def test_cli_upgrade_output_reloads_and_is_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            upgraded = Path(tmp) / "legacy-v1-upgraded.json"
            upgrade = cli(
                "upgrade",
                "--script",
                "samples/legacy-v1.json",
                "--output",
                str(upgraded),
            )
            self.assertEqual(upgrade.returncode, 0, upgrade.stderr)
            equivalent = cli(
                "equivalence",
                "--left",
                "samples/legacy-v1.json",
                "--right",
                str(upgraded),
            )
            self.assertEqual(equivalent.returncode, 0, equivalent.stderr)
            self.assertIn("equivalent=true", equivalent.stdout)

    def test_demo_is_repeatable_in_same_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "demo"
            first = cli("demo", "--workdir", str(workdir))
            second = cli("demo", "--workdir", str(workdir))
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first.stdout, second.stdout)
            self.assertEqual(first.stdout, (workdir / "demo_report.txt").read_text(encoding="utf-8"))
            self.assertIn("[1] normal concurrent editing", first.stdout)
            self.assertIn("[2] lock-wait timeout", first.stdout)
            self.assertIn("[3] deadlock arbitration", first.stdout)
            self.assertIn("[4] legacy v1/v2 upgrade then scheduling", first.stdout)
            self.assertIn("replay=REPRODUCIBLE", first.stdout)
            self.assertIn("upgrade_equivalence=OK", first.stdout)
            self.assertIn("outcome=E_LOCKTIMEOUT", first.stdout)
            self.assertIn("outcome=E_DEADLOCK", first.stdout)

    def test_gui_module_compiles_and_reports_missing_tk(self) -> None:
        source = ROOT / "isolation_lab" / "gui.py"
        py_compile.compile(str(source), doraise=True)
        try:
            import tkinter  # noqa: F401
        except ModuleNotFoundError:
            from isolation_lab.gui import launch_gui

            with self.assertRaises(BusinessError) as captured:
                launch_gui(db_path=":memory:", snapshots_path=":memory:")
            self.assertEqual(captured.exception.code, "E_SCRIPT")
            self.assertEqual(captured.exception.path, "$.gui")


if __name__ == "__main__":
    unittest.main()
