from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from cdn_edge.cli import main
from cdn_edge.demo import run_demo
from cdn_edge.errors import PipelineError

from tests.helpers import free_port_base


class DemoAndCliTests(unittest.TestCase):
    def test_demo_three_sessions_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cdn-edge-demo-test-") as temp:
            base = free_port_base(need=24, start=20000)
            report = run_demo(Path(temp) / "demo.sqlite3", port_base=base)
            self.assertTrue(report["ok"])
            self.assertEqual([item["phase"] for item in report["sessions"]], ["cold-fill", "version-update", "grid-perturb"])
            self.assertEqual(report["sessions"][0]["batch_count"], 1)
            self.assertEqual(report["sessions"][1]["batch_count"], 3)
            self.assertEqual(report["sessions"][2]["grid"]["power"], 2)
            self.assertEqual(report["sessions"][1]["batches"][1]["version_consistency_rate_units"], 499)
            self.assertLess(report["sessions"][1]["batches"][1]["version_consistency_rate_units"], 1000)
            self.assertTrue(report["diagnostics"]["ledger_self_consistency"]["ok"])
            self.assertTrue(report["diagnostics"]["whole_replacement"]["whole_replacement"])
            self.assertEqual(report["diagnostics"]["replay_from_genesis"]["checked_batches"], [1, 2, 3])
            self.assertEqual(report["diagnostics"]["replay_from_batch_1"]["checked_batches"], [2, 3])
            self.assertGreaterEqual(report["snapshot_count"], 10)

    def test_injection_codes_are_specific(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cdn-edge-anomaly-test-") as temp:
            with self.assertRaises(PipelineError) as cycle:
                run_demo(Path(temp) / "cycle.sqlite3", port_base=free_port_base(need=24, start=22000), inject="cycle")
            self.assertEqual(cycle.exception.code, "E_CYCLE")
            with self.assertRaises(PipelineError) as grid:
                run_demo(Path(temp) / "grid.sqlite3", port_base=free_port_base(need=24, start=24000), inject="grid")
            self.assertEqual(grid.exception.code, "E_GRID")
            with self.assertRaises(PipelineError) as snapshot:
                run_demo(Path(temp) / "snapshot.sqlite3", port_base=free_port_base(need=24, start=26000), inject="snapshot")
            self.assertEqual(snapshot.exception.code, "E_SNAPSHOT")

    def test_cli_spec_check_error_exit_code(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cdn-edge-cli-test-") as temp:
            path = Path(temp) / "bad.json"
            path.write_text('{"schema_version":1}', encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["spec-check", "--spec", str(path)]), 2)


if __name__ == "__main__":
    unittest.main()
