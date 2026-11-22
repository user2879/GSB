from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .helpers import load_json, run_cli, sample


class CliIntegrationTests(unittest.TestCase):
    def test_both_paths_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.json"
            report = root / "report.json"
            proc = run_cli(
                "run", "--spec", str(sample("large_range")), "--path", "both",
                "--ledger", str(root / "ledger.sqlite"),
                "--snapshot", str(root / "session.snapshot.json"),
                "--trace-out", str(trace), "--report", str(report), "--quiet")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = load_json(report)
            self.assertTrue(data["canonical_bytes_equal"])
            row_trace = root / "trace.row.json"
            range_trace = root / "trace.range.json"
            self.assertEqual(row_trace.read_bytes(), range_trace.read_bytes())
            self.assertEqual(len({len(row_trace.read_bytes()), len(range_trace.read_bytes())}), 1)

    def test_spec_error_exit_and_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "bad.json"
            bad.write_text(json.dumps({
                "schema_version": 1,
                "row_keys": ["A"],
                "transactions": [
                    {"id": "T1", "start_seq": 1, "isolation": "read_committed",
                     "operations": [{"seq": 1, "kind": "read", "row": "A"}]},
                    {"id": "T2", "start_seq": 1, "isolation": "read_committed",
                     "operations": [{"seq": 1, "kind": "read", "row": "A"}]},
                ],
            }), encoding="utf-8")
            report = root / "report.json"
            proc = run_cli("validate", "--spec", str(bad), "--report", str(report),
                           "--quiet")
            self.assertEqual(proc.returncode, 2)
            data = load_json(report)
            self.assertEqual(data["status"], "E_SPEC")
            self.assertEqual(data["error"]["txid"], "T2")
            self.assertEqual(data["error"]["field"], "start_seq")

    def test_timeout_and_divergence_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            timeout_report = root / "timeout.json"
            timeout = run_cli(
                "run", "--spec", str(sample("timeout")),
                "--ledger", str(root / "timeout.sqlite"),
                "--snapshot", str(root / "timeout.snapshot.json"),
                "--wait-timeout-ms", "30", "--wait-budget-ms", "500",
                "--inject", "hold-lock=SLOW:1", "--report", str(timeout_report),
                "--quiet")
            self.assertEqual(timeout.returncode, 3)
            self.assertEqual(load_json(timeout_report)["execution"]["timeouts"][0]["code"],
                             "E_TIMEOUT")

            diverge_report = root / "diverge.json"
            diverge = run_cli(
                "run", "--spec", str(sample("wait")), "--path", "both",
                "--ledger", str(root / "diverge.sqlite"),
                "--snapshot", str(root / "diverge.snapshot.json"),
                "--inject", "diverge", "--report", str(diverge_report), "--quiet")
            self.assertEqual(diverge.returncode, 5)
            error = load_json(diverge_report)["error"]
            self.assertEqual(error["code"], "E_DIVERGE")
            self.assertEqual((error["txid"], error["row"], error["field"]),
                             ("HOLDER", "lock:A", "event"))


if __name__ == "__main__":
    unittest.main()

