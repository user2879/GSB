from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from pv_ir_ledger.commands import apply_command
from pv_ir_ledger.commands import load_command_stream
from pv_ir_ledger.config import validate_config
from pv_ir_ledger.exchange import (
    downgrade_v2_to_v1,
    logical_comparison_view,
    parse_v1_text,
    parse_v2_document,
    upgrade_v1_to_v2,
)
from pv_ir_ledger.hashing import canonical_json
from pv_ir_ledger.imaging import write_png
from pv_ir_ledger.pipeline import run_pipeline
from pv_ir_ledger.recognition import determine_grade, synthesize_intensity
from pv_ir_ledger.reporting import domain_canonical
from pv_ir_ledger.samplegen import make_scene_heat
from pv_ir_ledger.truth import validate_truth


ROOT = Path(__file__).resolve().parents[1]


class ValidationTests(unittest.TestCase):
    def test_config_rejects_all_named_parameter_classes(self) -> None:
        config, issues = validate_config(
            {
                "colormap_name": "BAD_MAP",
                "grid_rows": 0,
                "grid_cols": True,
                "percentile_low_permille": 900,
                "percentile_high_permille": 100,
                "grade_thresholds_permille": [500, 400],
                "fixed_decimals": 9,
                "quant_levels": 13,
                "max_concurrency": 99,
            }
        )
        self.assertIsNone(config)
        codes = {issue["code"] for issue in issues}
        self.assertEqual(codes, {"E_CONFIG_INVALID"})
        self.assertGreaterEqual(len(issues), 7)

    def test_truth_validates_bbox_and_strict_ids(self) -> None:
        config, _ = validate_config({})
        raw = {
            "arrays": [
                {
                    "image": "A.png",
                    "grid_rows": 1,
                    "grid_cols": 2,
                    "components": [
                        {"component_id": 2, "bbox": [0, 0, 400, 20]},
                        {"component_id": 2, "bbox": [0, 0, 10, 10]},
                    ],
                }
            ]
        }
        truth, issues = validate_truth(raw, config)
        self.assertEqual(truth, {})
        self.assertTrue(any(i["code"] == "E_TRUTH_INVALID" for i in issues))
        self.assertTrue(any("strictly increasing" in i["message"] for i in issues))

    def test_sample_command_stream_is_valid(self) -> None:
        commands, issues = load_command_stream(ROOT / "samples" / "command_stream.json")
        self.assertEqual(len(commands), 7)
        self.assertEqual(issues, [])


class ExchangeTests(unittest.TestCase):
    def test_v1_truncated_and_illegal_fields(self) -> None:
        text = "SHORT\n" + "X" * 20 + "\n"
        judgments, issues = parse_v1_text(text)
        self.assertEqual(len(judgments), 2)
        self.assertTrue(issues)
        self.assertTrue(all(issue["code"] == "E_FIELD_TRUNCATED" for issue in issues))

    def test_float_v2_downgrade_and_sidecar_restore(self) -> None:
        document = {
            "judgments": [
                {
                    "image": "A.png",
                    "component_id": 7,
                    "bbox": [10.25, 20.5, 60.75, 80.4],
                    "intensity": 0.5724,
                    "grade": "MAJOR",
                    "quantization_scheme": "PV_Q4_L32_TEST",
                }
            ]
        }
        judgments, issues = parse_v2_document(document)
        self.assertEqual(issues, [])
        original = judgments[0]
        v1, sidecar = downgrade_v2_to_v1(original)
        self.assertEqual(v1["bbox"], [10, 21, 61, 80])
        self.assertEqual(v1["intensity_permille"], 572)
        self.assertEqual(sidecar["discarded_quantization_scheme"], "PV_Q4_L32_TEST")
        restored = upgrade_v1_to_v2(v1, {"entries": [sidecar]})
        self.assertEqual(
            logical_comparison_view(restored),
            logical_comparison_view(original),
        )
        self.assertEqual(restored["quantization_scheme"], "PV_Q4_L32_TEST")


class RecognitionTests(unittest.TestCase):
    def test_quantization_width_and_levels_change_integers(self) -> None:
        stats = {
            "high_percentile_heat": 900,
            "low_percentile_heat": 300,
            "hot_area_permille": 180,
        }
        regression = {"fill_permille": 640}
        q3, _ = validate_config({"fixed_decimals": 3, "quant_levels": 128})
        q2, _ = validate_config({"fixed_decimals": 2, "quant_levels": 16})
        first = synthesize_intensity(stats, regression, q3)
        second = synthesize_intensity(stats, regression, q2)
        self.assertNotEqual(first["intensity_fixed"], second["intensity_fixed"])
        self.assertNotEqual(first["quant_integer"], second["quant_integer"])
        self.assertEqual(first["fixed_scale"], 1000)
        self.assertEqual(second["fixed_scale"], 100)

    def test_equal_threshold_uses_higher_grade(self) -> None:
        grade = determine_grade(
            {"intensity_permille": 400},
            {"grade_thresholds_permille": [200, 400, 600, 800]},
            False,
        )
        self.assertEqual(grade["grade_code"], 2)
        self.assertTrue(grade["threshold_boundary"])


class PipelineIncrementalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.input_dir = Path(self.temp.name) / "input"
        self.input_dir.mkdir()
        self.config, issues = validate_config({"grid_rows": 2, "grid_cols": 2, "max_concurrency": 2})
        self.assertEqual(issues, [])
        heat, _ = make_scene_heat("TINY.png", rows=2, cols=2, width=320, height=240, variant=1)
        write_png(self.input_dir / "TINY.png", heat, self.config["colormap_name"])

    def test_threshold_reuses_regression_and_matches_full(self) -> None:
        baseline, state = run_pipeline(str(self.input_dir), self.config)
        new_config, issues = validate_config(
            {**self.config, "grade_thresholds_permille": [210, 390, 550, 710]}
        )
        self.assertEqual(issues, [])
        incremental, _ = run_pipeline(
            str(self.input_dir), new_config, previous_state=state
        )
        full, _ = run_pipeline(str(self.input_dir), new_config)
        self.assertEqual(domain_canonical(incremental), domain_canonical(full))
        regression_actions = [
            e["action"] for e in incremental["ledger_events"] if e["stage"] == "regression"
        ]
        self.assertTrue(regression_actions)
        self.assertTrue(all(action == "REUSED" for action in regression_actions))

    def test_budget_pending_resume_matches_full(self) -> None:
        pixel_count = 160 * 120
        cost = pixel_count * int(self.config["heat_bin_count"])
        partial, partial_state = run_pipeline(
            str(self.input_dir), self.config, previous_state=None, budget=cost
        )
        self.assertEqual(len(partial["components"]), 1)
        self.assertEqual(len(partial["pending"]), 3)
        self.assertTrue(any(i["code"] == "E_SAMPLE_BUDGET_PENDING" for i in partial["failures"]))
        resumed, resumed_state = run_pipeline(
            str(self.input_dir), self.config, previous_state=partial_state, budget=10**12
        )
        full, full_state = run_pipeline(str(self.input_dir), self.config, budget=10**12)
        self.assertEqual(domain_canonical(resumed), domain_canonical(full))
        self.assertEqual(
            canonical_json(resumed_state["budget"]["deductions"]),
            canonical_json(full_state["budget"]["deductions"]),
        )

    def test_silent_cell_is_degraded_without_neighbor_pollution(self) -> None:
        heat, _ = make_scene_heat(
            "SILENT.png", rows=2, cols=2, width=320, height=240, silent_component_id=2
        )
        write_png(self.input_dir / "SILENT.png", heat, self.config["colormap_name"])
        ledger, _ = run_pipeline(str(self.input_dir), self.config)
        silent = [c for c in ledger["components"] if c["image"] == "SILENT.png" and c["component_id"] == 2]
        others = [c for c in ledger["components"] if c["image"] == "SILENT.png" and c["component_id"] != 2]
        self.assertEqual(silent[0]["grade_code"], 9)
        self.assertTrue(all(c["grade_code"] != 9 for c in others))


class ArbitrarySequenceTests(unittest.TestCase):
    def test_multiple_arbitrary_command_sequences_match_full(self) -> None:
        for sequence_index in range(3):
            with self.subTest(sequence=sequence_index):
                temp = tempfile.TemporaryDirectory()
                self.addCleanup(temp.cleanup)
                input_dir = Path(temp.name) / "input"
                input_dir.mkdir()
                config, issues = validate_config({"grid_rows": 2, "grid_cols": 2, "max_concurrency": 2})
                self.assertEqual(issues, [])
                heat, _ = make_scene_heat("SEQ.png", rows=2, cols=2, variant=sequence_index)
                write_png(input_dir / "SEQ.png", heat, config["colormap_name"])
                state = None
                version = "v1"
                rng = random.Random(9000 + sequence_index)
                for step in range(8):
                    choice = rng.randrange(6)
                    if choice == 0:
                        command = {
                        "op": "set_grid",
                            "grid_rows": rng.randint(2, 4),
                            "grid_cols": rng.randint(2, 5),
                        }
                    elif choice == 1:
                        low = rng.randint(50, 500)
                        high = rng.randint(low + 100, 1000)
                        command = {
                            "op": "set_percentiles",
                            "percentile_low_permille": low,
                            "percentile_high_permille": min(1000, high),
                        }
                    elif choice == 2:
                        base = sorted(rng.sample(range(120, 880), 3))
                        thresholds = [base[0], max(base[0] + 20, base[1]), max(base[1] + 20, base[2]), max(base[2] + 20, 900)]
                        thresholds = sorted(set(thresholds))
                        while len(thresholds) < 4:
                            thresholds.append(min(980, thresholds[-1] + 20))
                        command = {"op": "set_thresholds", "grade_thresholds_permille": thresholds[:4]}
                    elif choice == 3:
                        command = {
                            "op": "set_quantization",
                            "fixed_decimals": rng.choice([0, 1, 2, 3, 4]),
                            "quant_levels": rng.choice([8, 16, 32, 64, 128, 256]),
                        }
                    elif choice == 4:
                        command = {"op": "set_exchange_version", "version": rng.choice(["v1", "v2"])}
                    else:
                        command = {"op": "replace_image", "image": "SEQ.png", "variant": rng.randint(10, 99)}
                    config, version, apply_issues, _ = apply_command(command, config, version, input_dir)
                    self.assertEqual(apply_issues, [], apply_issues)
                    incremental, state = run_pipeline(
                        str(input_dir), config, previous_state=state, exchange_version=version
                    )
                    full, _ = run_pipeline(str(input_dir), config, exchange_version=version)
                    self.assertEqual(
                        domain_canonical(incremental),
                        domain_canonical(full),
                        f"sequence {sequence_index} step {step} command {command}",
                    )


class SampleArtifactsTests(unittest.TestCase):
    def test_diagnostic_report_is_present_and_passing(self) -> None:
        report_path = ROOT / "reports" / "diagnostic_report.json"
        if not report_path.exists():
            self.skipTest("run python -m pv_ir_ledger verify to generate the report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(report["ok"])
        self.assertEqual(report["summary"]["failed"], 0)


if __name__ == "__main__":
    unittest.main()
