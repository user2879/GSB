from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from level_workbench.canonical import canonical_json_bytes
from level_workbench.runner import compare_directories, run_case, run_demo, write_demo_artifacts
from level_workbench.samples import demo_cases


class RunnerDemoTests(unittest.TestCase):
    def test_normal_case_passes_generator_and_verifier(self) -> None:
        case = next(case for case in demo_cases() if case["id"] == "normal")
        result = run_case(case)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["code"], "OK")
        self.assertIsNone(result["mismatch"])

    def test_unsat_first_violation_has_code_and_input_fragment(self) -> None:
        case = next(case for case in demo_cases() if case["id"] == "connectivity_unsat")
        result = run_case(case)
        self.assertEqual(result["code"], "E_CONNECT_UNSAT")
        first = result["first_violation"]
        self.assertEqual(first["code"], "E_CONNECT_UNSAT")
        self.assertEqual(first["coord"], [13, 5])
        self.assertEqual(first["input_fragment"]["conflict_pair"], ["spawn", "target"])

    def test_spec_patch_overflow_is_rejected_before_generation(self) -> None:
        case = next(case for case in demo_cases() if case["id"] == "resource_spec_overflow")
        result = run_case(case)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "E_SPEC_BOUNDS")
        self.assertIsNone(result["level_document"])

    def test_region_bounds_patch_is_rejected_with_bounds_error(self) -> None:
        case = next(case for case in demo_cases() if case["id"] == "normal")
        case["injection"] = {
            "id": "tamper-region-bounds",
            "spec_patches": [
                {"path": "regions[0].bounds", "value": [2, 2, 99, 5]},
            ],
        }
        result = run_case(case)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "E_SPEC_BOUNDS")
        self.assertEqual(result["error"]["path"], "regions[0].bounds")

    def test_artifact_tamper_reports_verifier_mismatch_with_differences(self) -> None:
        case = next(case for case in demo_cases() if case["id"] == "verifier_mismatch")
        result = run_case(case)
        self.assertEqual(result["code"], "E_VERIFIER_MISMATCH")
        differences = result["mismatch"]["differences"]
        self.assertEqual(differences[0]["constraint"], "forbidden_adjacent")
        self.assertEqual(differences[0]["generator"], "pass")
        self.assertEqual(differences[0]["verifier"], "fail")
        self.assertIsNotNone(result["minimal_counterexample"])

    def test_resource_repair_exhaustion_keeps_last_trace_and_underflow_reason(self) -> None:
        path = {(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)}
        fixed = []
        for y in range(3):
            for x in range(3):
                fixed.append(
                    {
                        "coord": [x, y],
                        "terrain": "grass" if (x, y) in path else "water",
                    }
                )
        raw = {
            "seed": 7,
            "grid": {"width": 3, "height": 3},
            "terrain": {"weights": {"grass": 0.5, "water": 0.5}, "blocked": ["water"]},
            "regions": [],
            "resources": {"count": [8, 8], "allowed_terrain": ["grass"]},
            "spawn": [0, 0],
            "target": [2, 2],
            "forbidden_adjacent": [],
            "fixed_cells": fixed,
        }
        result = run_case({"id": "underflow", "kind": "test", "spec": raw, "injection": None})
        self.assertEqual(result["code"], "E_REPAIR_EXHAUST")
        self.assertEqual(result["generation"]["repair_trace"]["resources"]["status"], "exhausted")
        self.assertIsNotNone(result["error"]["details"]["last_trace"])
        counterexample = result["minimal_counterexample"]
        self.assertEqual(counterexample["violation_code"], "E_RESOURCE_RANGE")
        self.assertFalse(counterexample["minimizable"])

    def test_pipeline_connectivity_sabotage_yields_minimal_cut(self) -> None:
        raw = {
            "seed": 20261119,
            "grid": {"width": 20, "height": 14},
            "terrain": {
                "weights": {"grass": 0.58, "sand": 0.20, "rock": 0.14, "water": 0.08},
                "blocked": ["water"],
            },
            "regions": [
                {"id": "left", "bounds": [2, 2, 5, 5]},
                {"id": "right", "bounds": [13, 2, 17, 5]},
            ],
            "resources": {"count": [0, 0], "allowed_terrain": ["grass", "sand"]},
            "spawn": [1, 7],
            "target": [18, 7],
            "forbidden_adjacent": [["rock", "water"]],
            "fixed_cells": [],
        }
        forced = []
        for y in range(14):
            for x in (8, 12):
                forced.append({"coord": [x, y], "terrain": "grass"})
            for x in (9, 10, 11):
                forced.append({"coord": [x, y], "terrain": "water"})
        injection = {"pipeline": {"post_generation_force_terrain": forced}}
        result = run_case({"id": "connectivity-injected", "kind": "test", "spec": raw, "injection": injection})
        self.assertEqual(result["code"], "E_CONNECT_UNSAT")
        self.assertIsNone(result["mismatch"])
        self.assertEqual(result["first_violation"]["code"], "E_CONNECT_UNSAT")
        counterexample = result["minimal_counterexample"]
        self.assertIsNotNone(counterexample)
        self.assertEqual(counterexample["reproduction"]["kind"], "blocked_cut")
        self.assertTrue(counterexample["checks"]["reproduces"])
        self.assertTrue(counterexample["checks"]["all_single_element_removals_disappear"])

    def test_demo_report_and_files_are_byte_identical_across_runs(self) -> None:
        first = run_demo()
        second = run_demo()
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(first["cases"]["normal"]["status"], "pass")
        self.assertEqual(first["cases"]["connectivity_unsat"]["code"], "E_CONNECT_UNSAT")
        self.assertEqual(first["cases"]["resource_pipeline_overflow"]["code"], "E_RESOURCE_RANGE")
        self.assertEqual(first["cases"]["illegal_adjacent"]["code"], "E_REPAIR_EXHAUST")
        self.assertEqual(first["cases"]["verifier_mismatch"]["code"], "E_VERIFIER_MISMATCH")
        with tempfile.TemporaryDirectory() as temp_a, tempfile.TemporaryDirectory() as temp_b:
            write_demo_artifacts(Path(temp_a), first)
            write_demo_artifacts(Path(temp_b), second)
            comparison = compare_directories(Path(temp_a), Path(temp_b))
            self.assertTrue(comparison["identical"])
            self.assertGreater(comparison["file_count_a"], 20)
            manifest = json.loads((Path(temp_a) / "hash_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("report.json", manifest["files"])
            self.assertIn("report.md", manifest["files"])


if __name__ == "__main__":
    unittest.main()
