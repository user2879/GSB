from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scorecard_app.binning import bin_categorical, bin_column
from scorecard_app.config import EffectiveColumn, validate_generated_table, validate_inputs
from scorecard_app.engine import ScorecardEngine
from scorecard_app.errors import E_CONFIG_INVALID
from scorecard_app.synthetic import generate_dataset

from tests.helpers import base_inputs, clone


class ConfigValidationTests(unittest.TestCase):
    def test_invalid_values_are_not_defaulted(self) -> None:
        experiment, schema = base_inputs(rows=100)
        cases = []

        exp = clone(experiment)
        exp["binning"]["max_bins"] = 0
        cases.append((exp, clone(schema), "max_bins"))

        exp = clone(experiment)
        exp["runtime"]["max_workers"] = 0
        cases.append((exp, clone(schema), "max_workers"))

        exp = clone(experiment)
        exp["features"].append("not_a_column")
        cases.append((exp, clone(schema), "unknown feature"))

        exp = clone(experiment)
        exp["columns"]["age"]["type"] = "categorical"
        cases.append((exp, clone(schema), "type mismatch"))

        schema_without_target = clone(schema)
        del schema_without_target["target"]
        cases.append((clone(experiment), schema_without_target, "target"))

        for exp, data, label in cases:
            with self.subTest(label=label):
                validated, failures = validate_inputs(exp, data)
                self.assertIsNone(validated)
                self.assertTrue(failures)
                self.assertTrue(all(item.code == E_CONFIG_INVALID for item in failures))

    def test_target_values_outside_binary_enum_fail(self) -> None:
        experiment, schema = base_inputs(rows=120)
        validated, failures = validate_inputs(experiment, schema)
        self.assertFalse(failures)
        self.assertIsNotNone(validated)
        dataset = generate_dataset(schema)
        dataset.target[0] = 7
        dataset.columns[dataset.target_name][0] = 7
        table_failures = validate_generated_table(validated, dataset.columns, dataset.target)
        self.assertTrue(any("outside binary enum" in item.message for item in table_failures))

    def test_engine_returns_config_error_report_without_snapshot(self) -> None:
        experiment, schema = base_inputs(rows=80)
        experiment["binning"]["max_bins"] = 0
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = ScorecardEngine(experiment, schema, root / "state").run(report_path=root / "report.json")
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["error"], "config_invalid")
            self.assertTrue(any(item["code"] == E_CONFIG_INVALID for item in report["failures"]))
            self.assertFalse((root / "state" / "manifest.json").exists())


def _params(direction: str = "none", max_bins: int = 6, min_ratio: float = 0.05) -> EffectiveColumn:
    return EffectiveColumn(
        name="x",
        column_type="numeric",
        max_bins=max_bins,
        min_bin_ratio=min_ratio,
        direction=direction,
        conflict_priority=["max_gap", "left", "right"],
        max_rounds=20,
    )


class BinningTests(unittest.TestCase):
    def test_numeric_missing_outliers_and_chi_square_tie(self) -> None:
        values = [1, 2, 3, 4, None, 999]
        target = [0, 1, 0, 1, 1, 0]
        result = bin_column("x", "numeric", values, target, 1, _params(max_bins=3, min_ratio=0.2), 4)
        bins = result["bins"]
        self.assertLessEqual(len(bins), 3)
        self.assertTrue(bins[-1]["is_missing"])
        self.assertEqual(sum(item["count"] for item in bins), len(values))
        nonmissing = [item for item in bins if not item["is_missing"]]
        self.assertTrue(all(isinstance(item["lower"], int) and isinstance(item["upper"], int) for item in nonmissing))
        for left, right in zip(nonmissing, nonmissing[1:]):
            self.assertLess(left["upper"], right["lower"])

        tie_result = bin_column("tie", "numeric", [1, 2, 3, 4], [0, 1, 0, 1], 1, _params(max_bins=2, min_ratio=0.1), 4)
        self.assertEqual([(item["lower"], item["upper"]) for item in tie_result["bins"]], [(1, 3), (4, 4)])

    def test_min_ratio_constraint(self) -> None:
        values = list(range(1, 21))
        target = [index % 2 for index in range(20)]
        result = bin_column("x", "numeric", values, target, 1, _params(max_bins=20, min_ratio=0.2), 20)
        for item in result["bins"]:
            if not item["is_missing"]:
                self.assertGreaterEqual(item["count"] / result["total"], 0.2)

    def test_categorical_mapping_and_missing_group(self) -> None:
        values = ["a"] * 5 + ["b"] * 4 + ["c"] * 3 + ["d"] * 2 + [None] * 2
        target = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1]
        result = bin_categorical("cat", values, target, 1, _params(max_bins=3, min_ratio=0.1), 64)
        mapping = result["category_to_group"]
        self.assertEqual(set(mapping), {"a", "b", "c", "d"})
        self.assertEqual(set(mapping.values()), set(range(len(result["bins"]) - 1)))
        self.assertTrue(result["bins"][-1]["is_missing"])
        self.assertEqual(sum(item["count"] for item in result["bins"]), len(values))


if __name__ == "__main__":
    unittest.main()
