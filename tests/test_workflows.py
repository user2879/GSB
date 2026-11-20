from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scorecard_app.binning import bin_column as real_bin_column
from scorecard_app.engine import ScorecardEngine
from scorecard_app.errors import (
    E_COLUMN_BINNING_FAILED,
    E_MONOTONIC_FAILED,
    E_MONOTONIC_UNRESOLVED,
)
from scorecard_app.monotonic import solve_monotonic as real_solve_monotonic
from scorecard_app.synthetic import generate_dataset

from tests.helpers import ROOT, base_inputs, clone, content_view


class IncrementalConsistencyTests(unittest.TestCase):
    def test_edit_sequence_matches_full_rebuild_and_invalidates_minimally(self) -> None:
        experiment, schema = base_inputs(rows=420)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            incremental_state = root / "incremental"
            first = ScorecardEngine(experiment, schema, incremental_state).run(
                report_path=root / "initial.json"
            )
            self.assertEqual(first["status"], "ok")
            self.assertEqual(first["snapshot"]["version"], 5)

            edits = [
                (lambda e, s: e["columns"]["age"].__setitem__("monotonic_direction", "descending"), {"age"}),
                (lambda e, s: e["columns"]["income"].__setitem__("min_bin_ratio", 0.10), {"income"}),
                (lambda e, s: e["monotonic"].__setitem__("conflict_priority", ["left", "max_gap"]), {"age", "bureau_score", "income", "tenure_months"}),
                (lambda e, s: e["binning"].__setitem__("max_bins", 5), {"age", "bureau_score", "channel", "merchant_cluster", "occupation", "tenure_months"}),
                (lambda e, s: self._set_seed_offset(s, "income", 7), {"income"}),
            ]

            for index, (mutate, expected_invalidated) in enumerate(edits, start=1):
                mutate(experiment, schema)
                incremental = ScorecardEngine(experiment, schema, incremental_state, allow_revision=True).run(
                    report_path=root / f"incremental-{index}.json",
                    allow_revision=True,
                )
                self.assertEqual(incremental["status"], "ok")
                self.assertEqual(set(incremental["cache"]["invalidated_columns"]), expected_invalidated)

                fresh_state = root / f"full-{index}"
                full = ScorecardEngine(clone(experiment), clone(schema), fresh_state).run(
                    report_path=root / f"full-{index}.json"
                )
                self.assertEqual(content_view(incremental), content_view(full))
                if index == 1:
                    for column in incremental["columns"]:
                        self.assertEqual(column["cache"]["binning"], "hit")
                        self.assertEqual(column["cache"]["woe"], "hit")
                        if column["name"] == "age":
                            self.assertEqual(column["cache"]["monotonic"], "miss")
                        else:
                            self.assertEqual(column["cache"]["monotonic"], "hit")

    @staticmethod
    def _set_seed_offset(schema: dict, column: str, offset: int) -> None:
        for spec in schema["columns"]:
            if spec["name"] == column:
                spec["seed_offset"] = offset
                return
        raise AssertionError(column)

    def test_single_column_replacement_keeps_target_and_other_columns_stable(self) -> None:
        _, schema = base_inputs(rows=360)
        before = generate_dataset(schema)
        changed_schema = clone(schema)
        self._set_seed_offset(changed_schema, "income", 11)
        after = generate_dataset(changed_schema)
        self.assertEqual(before.target, after.target)
        self.assertEqual(before.columns["age"], after.columns["age"])
        self.assertNotEqual(before.columns["income"], after.columns["income"])


class ConcurrencyAndIsolationTests(unittest.TestCase):
    def test_two_fresh_states_produce_byte_identical_reports(self) -> None:
        experiment, schema = base_inputs(rows=260)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first_path = root / "first.json"
            second_path = root / "second.json"
            first = ScorecardEngine(experiment, schema, root / "first-state").run(report_path=first_path)
            second = ScorecardEngine(experiment, schema, root / "second-state").run(report_path=second_path)
            self.assertEqual(first, second)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

    def test_worker_count_does_not_change_results(self) -> None:
        experiment, schema = base_inputs(rows=360)
        serial_experiment = clone(experiment)
        parallel_experiment = clone(experiment)
        serial_experiment["runtime"]["max_workers"] = 1
        parallel_experiment["runtime"]["max_workers"] = 8
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            serial = ScorecardEngine(serial_experiment, schema, root / "serial").run(
                report_path=root / "serial.json"
            )
            parallel = ScorecardEngine(parallel_experiment, schema, root / "parallel").run(
                report_path=root / "parallel.json"
            )
            self.assertEqual(content_view(serial), content_view(parallel))

    def test_column_exceptions_are_isolated_and_failures_are_sorted(self) -> None:
        experiment, schema = base_inputs(rows=360)

        def side_effect(name: str, *args, **kwargs):
            if name in {"occupation", "income"}:
                raise ValueError(f"injected failure for {name}")
            return real_bin_column(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch("scorecard_app.engine.bin_column", side_effect=side_effect):
                report = ScorecardEngine(experiment, schema, root / "state").run(
                    report_path=root / "report.json"
                )
            self.assertEqual(report["status"], "partial")
            failed_columns = [item["column"] for item in report["failures"]]
            self.assertEqual(failed_columns, ["income", "occupation"])
            self.assertTrue(all(item["code"] == E_COLUMN_BINNING_FAILED for item in report["failures"]))
            self.assertTrue(all(item["exception_type"] == "ValueError" for item in report["failures"]))
            reported_columns = [item["name"] for item in report["columns"]]
            self.assertNotIn("income", reported_columns)
            self.assertNotIn("occupation", reported_columns)
            self.assertEqual(
                report["evaluation"]["scored_columns"],
                ["age", "bureau_score", "channel", "merchant_cluster", "tenure_months"],
            )

    def test_unresolved_monotonic_state_is_preserved_and_excluded_from_evaluation(self) -> None:
        experiment, schema = base_inputs(rows=420)
        experiment["columns"]["age"]["monotonic_direction"] = "descending"
        experiment["monotonic"]["max_rounds"] = 0
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = ScorecardEngine(experiment, schema, root / "state").run(
                report_path=root / "report.json"
            )
            self.assertEqual(report["status"], "partial")
            self.assertTrue(any(item["code"] == E_MONOTONIC_UNRESOLVED for item in report["failures"]))
            age = next(item for item in report["columns"] if item["name"] == "age")
            self.assertEqual(age["status"], "unresolved")
            self.assertTrue(age["bins"])
            self.assertEqual(age["monotonic_correction_count"], 0)
            self.assertNotIn("age", report["evaluation"]["scored_columns"])

    def test_monotonic_exception_is_isolated(self) -> None:
        experiment, schema = base_inputs(rows=360)

        def side_effect(column: str, *args, **kwargs):
            if column == "age":
                raise RuntimeError("injected monotonic failure")
            return real_solve_monotonic(column, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch("scorecard_app.engine.solve_monotonic", side_effect=side_effect):
                report = ScorecardEngine(experiment, schema, root / "state").run(
                    report_path=root / "report.json"
                )
            self.assertEqual(report["status"], "partial")
            failure = next(item for item in report["failures"] if item["column"] == "age")
            self.assertEqual(failure["code"], E_MONOTONIC_FAILED)
            self.assertEqual(failure["stage"], "monotonic")
            self.assertEqual(failure["exception_type"], "RuntimeError")
            age = next(item for item in report["columns"] if item["name"] == "age")
            self.assertEqual(age["status"], "monotonic_failed")
            self.assertNotIn("age", report["evaluation"]["scored_columns"])
            self.assertIn("bureau_score", report["evaluation"]["scored_columns"])


class ShellTests(unittest.TestCase):
    def test_shell_command_sequence_and_save(self) -> None:
        experiment, schema = base_inputs(rows=260)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exp_path = root / "experiment.yaml"
            data_path = root / "data.yaml"
            import yaml

            exp_path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")
            data_path.write_text(yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")
            saved_exp = root / "saved-experiment.yaml"
            saved_data = root / "saved-data.yaml"
            commands = root / "commands.txt"
            commands.write_text(
                "\n".join(
                    [
                        "direction age descending",
                        "max-bins 5",
                        "min-ratio 0.08 income",
                        "priority max_gap,left",
                        "replace-column income 3",
                        f"save {saved_exp} {saved_data}",
                        "quit",
                    ]
                ),
                encoding="utf-8",
            )
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scorecard_app",
                    "shell",
                    "--config",
                    str(exp_path),
                    "--data",
                    str(data_path),
                    "--state-dir",
                    str(root / "state"),
                    "--report",
                    str(root / "report.json"),
                    "--commands",
                    str(commands),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            self.assertTrue(saved_exp.exists())
            self.assertTrue(saved_data.exists())
            report = __import__("json").loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertTrue(report["snapshot"]["version"] > 5)


if __name__ == "__main__":
    unittest.main()
