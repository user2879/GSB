from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from autoscale_diag.cli import main as cli_main
from autoscale_diag.demo import run_demo
from autoscale_diag.engine import simulate
from autoscale_diag.ledger import Ledger
from autoscale_diag.spec import load_spec
from autoscale_diag.viewmodel import load_dashboard_model

from tests.helpers import SPEC_V1, mutated_sample


class CliDemoUiIntegrationTests(unittest.TestCase):
    def test_invalid_spec_cli_emits_espec_and_field_path(self) -> None:
        payload = mutated_sample(lambda data: data["workloads"][0].__setitem__("cooldown_ms", 0))
        with tempfile.TemporaryDirectory() as temporary:
            spec_path = Path(temporary) / "invalid.json"
            spec_path.write_text(json.dumps(payload), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = cli_main(["validate", str(spec_path)])
            error = json.loads(stderr.getvalue())
        self.assertEqual(code, 2)
        self.assertEqual(error["business_code"], "E_SPEC")
        self.assertEqual(error["field_path"], "workloads[0].cooldown_ms")

    def test_compare_paths_cli_reports_identical_digest(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli_main(["compare-paths", str(SPEC_V1)])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(payload["byte_identical_decisions_and_metrics"])
        self.assertEqual(
            payload["full"]["decision_digest"], payload["incremental"]["decision_digest"]
        )

    def test_incomplete_prefix_cannot_produce_exit_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            db = Path(temporary) / "prefix.db"
            spec = load_spec(SPEC_V1)
            result = simulate(spec)
            with Ledger(db) as ledger:
                ledger.create_run(spec, run_id="prefix", path=result.path)
                for minute in result.minutes[:2]:
                    ledger.commit_minute("prefix", minute, spec.minute_count)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = cli_main(["report", "--db", str(db), "--run-id", "prefix"])
            error = json.loads(stderr.getvalue())
            self.assertEqual(code, 4)
            self.assertEqual(error["business_code"], "E_LEDGER")
            self.assertEqual(error["details"]["committed_minutes"], 2)

    def test_three_scenario_demo_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "demo"
            result = run_demo(output)
            self.assertTrue(result["ok"], result)
            self.assertTrue(all(result["checks"].values()))
            for name in (
                "01_cold_start.json",
                "02_cooldown_suppression.json",
                "03_budget_replacement.json",
                "04_path_parity.json",
                "05_replay_guard.json",
                "06_ui_viewmodel_check.json",
                "summary.json",
            ):
                self.assertTrue((output / name).exists(), name)
            self.assertTrue((output / "v1" / "report.html").exists())
            self.assertTrue((output / "v1" / "snapshot.json").exists())
            self.assertTrue((output / "v2" / "report.json").exists())

            model = load_dashboard_model(output / "workbench.db", "demo")
            self.assertEqual(len(model.rows), 36)
            self.assertGreater(len(model.cooling_suppressions), 0)
            self.assertTrue(
                [(row["minute_index"], row["workload_id"]) for row in model.rows]
                == sorted((row["minute_index"], row["workload_id"]) for row in model.rows)
            )

    def test_ui_check_runs_without_tkinter_dependency_at_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            db = Path(temporary) / "ui.db"
            spec = load_spec(SPEC_V1)
            result = simulate(spec)
            with Ledger(db) as ledger:
                ledger.create_run(spec, run_id="ui", path=result.path)
                for minute in result.minutes:
                    ledger.commit_minute("ui", minute, spec.minute_count)
                ledger.finalize_run("ui", spec.minute_count)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = cli_main(
                    ["ui", "--db", str(db), "--run-id", "ui", "--check"]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["stable_order"])
            self.assertEqual(payload["rows"], 36)


if __name__ == "__main__":
    unittest.main()
