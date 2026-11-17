from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from exp_tracker.cli import main
from exp_tracker.demo import run_demo

from .helpers import SAMPLE_PATH


class CliDemoTests(unittest.TestCase):
    def test_demo_is_byte_stable_when_repeated_in_same_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first = run_demo(workspace)
            second = run_demo(workspace)
            self.assertEqual(
                json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
            )
            self.assertEqual(
                [item["scenario"] for item in first["scenarios"]],
                [
                    "1_import_and_upgrade",
                    "2_whole_record_replacement",
                    "3_snapshot_restore_and_replay",
                ],
            )
            self.assertTrue(first["scenarios"][0]["upgrade"]["processed"])
            self.assertTrue(first["scenarios"][1]["replacement"]["whole_record_replaced"])
            self.assertTrue(first["scenarios"][2]["restore"]["restore"]["is_different"])
            self.assertEqual(
                first["scenarios"][2]["restore"]["replay"]["status"], "pass"
            )
            self.assertTrue((workspace / "reports" / "demo-report.json").exists())

    def test_cli_init_list_and_error_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "init",
                        "--workspace",
                        str(workspace),
                        "--from",
                        str(SAMPLE_PATH),
                    ]
                )
            self.assertEqual(code, 0)
            result = json.loads(stdout.getvalue())
            self.assertTrue(result["ok"])
            self.assertEqual(len(result["runs"]), 3)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["list", "--workspace", str(workspace)])
            self.assertEqual(code, 0)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["format_versions"], [3])
            self.assertEqual(result["source_format_versions"], [1, 2, 3])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["curve", "baseline", "--workspace", str(workspace)])
            self.assertEqual(code, 0)
            chart_result = json.loads(stdout.getvalue())
            self.assertIn("#", "\n".join(chart_result["charts"]["loss"]))
            self.assertIn("step 1", chart_result["charts"]["accuracy"][-1])

            invalid = workspace / "invalid.json"
            invalid.write_text('{"format_version": 99}\n', encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["commit", "--workspace", str(workspace), "--record", str(invalid)])
            self.assertEqual(code, 2)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["error"]["code"], "E_FORMAT")
            self.assertEqual(result["error"]["details"]["path"], "$.format_version")

    def test_cli_workspace_option_before_and_after_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "--workspace",
                        str(workspace),
                        "init",
                        "--from",
                        str(SAMPLE_PATH),
                    ]
                )
            self.assertEqual(code, 0)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["--workspace", str(workspace), "list"])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
