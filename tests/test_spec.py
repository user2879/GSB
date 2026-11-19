from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from level_workbench.cli import main
from level_workbench.errors import SpecError
from level_workbench.samples import normal_spec
from level_workbench.spec import parse_spec


class SpecParserTests(unittest.TestCase):
    def test_missing_field_is_stable_error(self) -> None:
        raw = normal_spec()
        del raw["seed"]
        with self.assertRaises(SpecError) as caught:
            parse_spec(raw)
        self.assertEqual(caught.exception.code, "E_SPEC_FIELD")
        self.assertEqual(caught.exception.path, "seed")

    def test_weight_sum_must_equal_one(self) -> None:
        raw = normal_spec()
        raw["terrain"]["weights"]["grass"] = 0.5
        with self.assertRaises(SpecError) as caught:
            parse_spec(raw)
        self.assertEqual(caught.exception.code, "E_SPEC_WEIGHT")

    def test_region_bounds_must_fit_grid(self) -> None:
        raw = normal_spec()
        raw["regions"][0]["bounds"] = [2, 2, 25, 5]
        with self.assertRaises(SpecError) as caught:
            parse_spec(raw)
        self.assertEqual(caught.exception.code, "E_SPEC_BOUNDS")
        self.assertEqual(caught.exception.path, "regions[0].bounds")

    def test_resource_maximum_must_fit_grid_cells(self) -> None:
        raw = normal_spec()
        raw["resources"]["count"] = [300, 320]
        with self.assertRaises(SpecError) as caught:
            parse_spec(raw)
        self.assertEqual(caught.exception.code, "E_SPEC_BOUNDS")
        self.assertEqual(caught.exception.path, "resources.count")

    def test_normalized_spec_is_order_independent(self) -> None:
        raw = normal_spec()
        spec = parse_spec(raw)
        reordered = copy.deepcopy(raw)
        reordered["terrain"]["weights"] = dict(reversed(list(reordered["terrain"]["weights"].items())))
        reordered["regions"] = list(reversed(reordered["regions"]))
        second = parse_spec(reordered)
        self.assertEqual(spec.normalized, second.normalized)
        self.assertEqual([region.id for region in spec.regions], ["grove", "marsh", "ridge"])

    def test_invalid_spec_cli_creates_no_grid_file(self) -> None:
        raw = normal_spec()
        del raw["resources"]["count"]
        with tempfile.TemporaryDirectory() as temp_dir:
            spec_path = Path(temp_dir) / "bad.yaml"
            out_path = Path(temp_dir) / "out"
            spec_path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
            capture = io.StringIO()
            with redirect_stdout(capture):
                code = main(["generate", "--spec", str(spec_path), "--out", str(out_path)])
            self.assertEqual(code, 2)
            payload = json.loads(capture.getvalue())
            self.assertEqual(payload["error"]["code"], "E_SPEC_FIELD")
            self.assertFalse(out_path.exists())


if __name__ == "__main__":
    unittest.main()
