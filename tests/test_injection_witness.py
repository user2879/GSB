from __future__ import annotations

import unittest
from unittest import mock

from pickworkbench.canonical import content_hash
from pickworkbench.demo_cases import build_demo_cases
from pickworkbench.engine import execute_case, run_batch, reproduce_witness
from pickworkbench.errors import (
    E_INJECT_CONFIG,
    E_OBJECT_DUP_ID,
    E_PICK_MISMATCH,
    E_PRIORITY_RANGE,
    E_RAY_DEGENERATE,
    E_SNAP_SPACING,
    E_NORMAL_ZERO,
    ERROR_CODES,
)
from pickworkbench.generator import generate_sample_with_config
from pickworkbench.injection import apply_injection, empty_injection


class InjectionAndWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene, cls.queries = generate_sample_with_config()
        cls.cases = {case["name"]: case for case in build_demo_cases(cls.scene, cls.queries)}
        cls.reports = {
            name: execute_case(
                name,
                cls.scene,
                cls.queries,
                case["injection"],
                case["expected_error_codes"],
            )
            for name, case in cls.cases.items()
        }

    def test_all_required_business_codes_exist(self) -> None:
        self.assertEqual(
            set(ERROR_CODES),
            {
                E_RAY_DEGENERATE,
                E_NORMAL_ZERO,
                E_OBJECT_DUP_ID,
                E_PRIORITY_RANGE,
                E_SNAP_SPACING,
                E_PICK_MISMATCH,
                E_INJECT_CONFIG,
            },
        )

    def test_expected_codes_and_first_affected_queries(self) -> None:
        expectations = {
            "00_normal": None,
            "01_degenerate_ray": ("q-000-ray", E_RAY_DEGENERATE),
            "02_zero_normal": ("q-001-ray", E_NORMAL_ZERO),
            "03_duplicate_id": ("q-000-ray", E_OBJECT_DUP_ID),
            "04_priority_range": ("q-000-ray", E_PRIORITY_RANGE),
            "05_snap_spacing": ("q-018-drag", E_SNAP_SPACING),
            "06_invalid_inject_config": None,
        }
        for name, expected in expectations.items():
            report = self.reports[name]
            self.assertEqual(report["status"], "pass", name)
            if expected is None:
                self.assertIsNone(report["first_affected"], name)
                continue
            first = report["first_affected"]
            self.assertEqual((first["query_id"], first["error_code"]), expected, name)
            self.assertIn(
                expected[1],
                report["detected_error_codes"],
                name,
            )

    def test_single_query_failure_does_not_abort_batch(self) -> None:
        report = self.reports["01_degenerate_ray"]
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(report["summary"]["passed"], 21)
        self.assertEqual(report["summary"]["error_counts"], {E_RAY_DEGENERATE: 1})
        error_records = [record for record in report["records"] if record["status"] == "error"]
        self.assertEqual(len(error_records), 1)
        self.assertEqual(error_records[0]["query_id"], "q-000-ray")
        self.assertNotIn("winner", error_records[0])

    def test_unaffected_query_hashes_survive_injection(self) -> None:
        baseline = {
            record["query_id"]: record["content_hash"]
            for record in self.reports["00_normal"]["records"]
        }
        degenerate = self.reports["01_degenerate_ray"]
        for record in degenerate["records"]:
            if record["query_id"] != "q-000-ray":
                self.assertEqual(record["content_hash"], baseline[record["query_id"]])

    def test_each_case_report_is_deterministic(self) -> None:
        for name, report in self.reports.items():
            case = self.cases[name]
            repeated = execute_case(
                name,
                self.scene,
                self.queries,
                case["injection"],
                case["expected_error_codes"],
            )
            self.assertEqual(report["content_hash"], repeated["content_hash"], name)

    def test_witnesses_reproduce_and_are_minimal(self) -> None:
        for name, report in self.reports.items():
            witness = report.get("witness")
            if witness is None:
                continue
            self.assertTrue(witness["minimal"], name)
            outcome = reproduce_witness(witness)
            self.assertTrue(outcome["reproduced"], name)
            self.assertEqual(outcome["error_code"], report["first_affected"]["error_code"], name)
            self.assertTrue(witness["removal_checks"], name)
            for check in witness["removal_checks"]:
                self.assertFalse(check["reproduced"], f"{name}: {check['removed_element']}")

    def test_invalid_injection_is_mapped_and_reported(self) -> None:
        report = self.reports["06_invalid_inject_config"]
        self.assertEqual(report["detected_error_codes"], [E_INJECT_CONFIG])
        self.assertEqual(report["error"]["code"], E_INJECT_CONFIG)
        self.assertEqual(report["execution_status"], "injection_rejected")
        self.assertEqual(report["query_hashes"], [])

    def test_unknown_injection_field_is_rejected(self) -> None:
        bad = empty_injection("bad")
        bad["unknown"] = True
        with self.assertRaises(Exception) as context:
            apply_injection(self.scene, self.queries, bad)
        self.assertEqual(context.exception.code, E_INJECT_CONFIG)

    def test_first_object_order_is_deterministic_for_multiple_anomalies(self) -> None:
        injection = empty_injection("two_priorities")
        injection["priority_overrides"] = [
            {"object_index": 1, "priority": 999},
            {"object_index": 0, "priority": 999},
        ]
        scene, queries, _ = apply_injection(self.scene, self.queries, injection)
        result = run_batch(scene, queries)
        first = result["first_affected"]
        self.assertEqual(first["query_id"], "q-000-ray")
        self.assertEqual(first["trigger"]["object_id"], "obj-000")
        self.assertEqual(first["trigger"]["object_index"], 0)

    def test_pick_mismatch_is_isolated_and_mapped(self) -> None:
        with mock.patch("pickworkbench.engine.screen_candidates", return_value=[]):
            result = run_batch(self.scene, self.queries)
        self.assertEqual(result["summary"]["error_counts"], {E_PICK_MISMATCH: 8})
        self.assertEqual(result["summary"]["failed"], 8)
        self.assertEqual(result["summary"]["passed"], 14)
        self.assertTrue(
            all(
                record["error"]["code"] == E_PICK_MISMATCH
                for record in result["records"]
                if record["status"] == "error"
            )
        )

    def test_injection_hash_covers_effective_inputs(self) -> None:
        report = self.reports["04_priority_range"]
        self.assertEqual(report["injection_hash"], content_hash(report["injection"]))
        injected_scene, _, _ = apply_injection(
            self.scene,
            self.queries,
            self.cases["04_priority_range"]["injection"],
        )
        self.assertEqual(report["scene_hash"], content_hash(injected_scene))


if __name__ == "__main__":
    unittest.main()
