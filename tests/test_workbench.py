from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pe_workbench.cache_keys import stage_keys
from pe_workbench.config import ToleranceConfig, load_config, validate_config_dict
from pe_workbench.encoding import encode_pair, encode_vector
from pe_workbench.engine import WorkbenchEngine
from pe_workbench.error_tracking import track_errors
from pe_workbench.executor import CommandExecutor
from pe_workbench.hashing import stable_json
from pe_workbench.quantization import (
    integer_dot_with_accumulation,
    quantization_profile,
    quantize_vector,
    score_fixed_path,
)
from pe_workbench.reporting import (
    _layer_statistics,
    build_diagnostic_report,
    build_visual_summary,
)
from pe_workbench.sensitivity import SensitivityAnalyzer
from pe_workbench.validation import parse_commands_jsonl, parse_samples_jsonl


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def load_fixture(sample_file: str = "attention_samples.jsonl"):
    config, issues = load_config(SAMPLES / "position_encoding.json")
    assert config is not None and not issues
    loaded = parse_samples_jsonl(SAMPLES / sample_file, config)
    return config, loaded.samples


class ValidationTests(unittest.TestCase):
    def test_invalid_samples_continue_and_report_business_codes(self):
        config, _ = load_fixture()
        loaded = parse_samples_jsonl(SAMPLES / "invalid_samples.jsonl", config)
        codes = {issue.code for issue in loaded.issues}
        self.assertEqual(len(loaded.samples), 2)
        self.assertIn("E_SAMPLE_ID_DUPLICATE", codes)
        self.assertIn("E_SAMPLE_QUERY_POSITION", codes)
        self.assertIn("E_SAMPLE_KEY_POSITION", codes)
        self.assertIn("E_SAMPLE_FEATURE_LENGTH", codes)
        self.assertIn("E_SAMPLE_QUERY_VECTOR", codes)
        self.assertIn("E_SAMPLE_JSON", codes)
        duplicate = next(issue for issue in loaded.issues if issue.code == "E_SAMPLE_ID_DUPLICATE")
        self.assertEqual(duplicate.sample_id, "s0001")
        self.assertEqual(duplicate.field, "sample_id")

    def test_config_enum_and_even_dimension_validation(self):
        raw = json.loads((SAMPLES / "position_encoding.json").read_text(encoding="utf-8"))
        raw["dimension"] = 33
        raw["quantization_level"] = "impossible"
        config, issues = validate_config_dict(raw)
        self.assertIsNone(config)
        codes = {issue.code for issue in issues}
        self.assertIn("E_CONFIG_DIMENSION_EVEN", codes)
        self.assertIn("E_CONFIG_QUANTIZATION_LEVEL", codes)

    def test_invalid_command_stream_is_non_short_circuiting(self):
        config, _ = load_fixture()
        loaded = parse_commands_jsonl(SAMPLES / "invalid_commands.jsonl", config)
        self.assertEqual(len(loaded.commands), 1)
        codes = {issue.code for issue in loaded.issues}
        self.assertIn("E_COMMAND_JSON", codes)
        self.assertIn("E_COMMAND_NAME", codes)
        self.assertIn("E_COMMAND_PARAM", codes)
        self.assertIn("E_CONFIG_DIMENSION_EVEN", codes)


class EncodingAndQuantizationTests(unittest.TestCase):
    def test_three_schemes_are_reproducible_and_numerically_distinct(self):
        config, samples = load_fixture()
        sample = samples[0]
        outputs = {}
        for scheme in ("rope", "linear", "ntk"):
            first = encode_pair(
                sample.sample_id,
                sample.query_position,
                sample.key_positions,
                sample.query_vector,
                sample.key_vectors,
                replace(config, scheme=scheme),
                precision_level="fp64",
            )
            second = encode_pair(
                sample.sample_id,
                sample.query_position,
                sample.key_positions,
                sample.query_vector,
                sample.key_vectors,
                replace(config, scheme=scheme),
                precision_level="fp64",
            )
            self.assertEqual(first["query"]["vector"], second["query"]["vector"])
            self.assertEqual(first["metadata"]["scheme"], scheme)
            self.assertEqual(first["metadata"]["base_frequency"], config.base_frequency)
            self.assertEqual(first["metadata"]["scale_factor"], config.scale_factor)
            outputs[scheme] = tuple(first["query"]["vector"])
        self.assertNotEqual(outputs["rope"], outputs["linear"])
        self.assertNotEqual(outputs["rope"], outputs["ntk"])
        self.assertNotEqual(outputs["linear"], outputs["ntk"])

        linear = encode_vector(
            sample.query_vector,
            sample.query_position,
            scheme="linear",
            base_frequency=config.base_frequency,
            dimension=config.dimension,
            scale_factor=config.scale_factor,
            precision_level="fp64",
        )
        self.assertAlmostEqual(
            linear.effective_position, sample.query_position / config.scale_factor, places=12
        )
        ntk = encode_vector(
            sample.query_vector,
            sample.query_position,
            scheme="ntk",
            base_frequency=config.base_frequency,
            dimension=config.dimension,
            scale_factor=config.scale_factor,
            precision_level="fp64",
        )
        self.assertGreater(ntk.metadata["effective_base_frequency"], config.base_frequency)

    def test_integer_dot_accumulates_in_ascending_dimension_blocks(self):
        values = [0.25, -0.5, 0.75, 1.0, -1.25, 0.5, 0.125, -0.25]
        quantized = quantize_vector(values, level="balanced", fixed_decimal_bits=10)
        result = integer_dot_with_accumulation(quantized["integers"], quantized["integers"], 3)
        self.assertEqual(result["accumulation"][0]["dim_start"], 0)
        self.assertEqual(result["accumulation"][-1]["dim_end"], len(values) - 1)
        self.assertEqual(result["accumulation"][-1]["cumulative_sum"], result["raw_dot"])
        self.assertEqual(len(result["dimension_steps"]), len(values))
        self.assertEqual(result["dimension_steps"][-1]["dimension"], len(values) - 1)
        self.assertEqual(result["dimension_steps"][-1]["cumulative_sum"], result["raw_dot"])
        previous_end = -1
        cumulative = 0
        for block in result["accumulation"]:
            self.assertGreater(block["dim_start"], previous_end)
            previous_end = block["dim_end"]
            cumulative += block["block_sum"]
            self.assertEqual(block["cumulative_sum"], cumulative)

    def test_quantization_tiers_change_effective_precision(self):
        fine = quantization_profile("fine", 10)
        balanced = quantization_profile("balanced", 10)
        coarse = quantization_profile("coarse", 10)
        self.assertGreater(fine["effective_fractional_bits"], balanced["effective_fractional_bits"])
        self.assertGreater(balanced["effective_fractional_bits"], coarse["effective_fractional_bits"])

    def test_progressive_error_first_stage_order(self):
        config, samples = load_fixture()
        sample = next(
            sample
            for sample in samples
            if any(abs(sample.query_position - position) >= 64 for position in sample.key_positions)
        )
        tight = replace(
            config,
            fixed_decimal_bits=6,
            quantization_level="coarse",
            tolerances=ToleranceConfig(1e-15, 1e-15, 1e-15),
        )
        pe = encode_pair(
            sample.sample_id,
            sample.query_position,
            sample.key_positions,
            sample.query_vector,
            sample.key_vectors,
            tight,
        )
        score = score_fixed_path(
            pe["query"],
            pe["keys"],
            sample.query_position,
            sample.key_positions,
            dimension=tight.dimension,
            dimension_block_size=tight.dimension_block_size,
            quantization_level=tight.quantization_level,
            fixed_decimal_bits=tight.fixed_decimal_bits,
        )
        error = track_errors(sample, pe, score, tight, cache_key=stage_keys(sample, tight)["error_tracking"])
        self.assertTrue(error["evaluated"])
        self.assertEqual(error["first_exceeded_stage"], "position_encoding")
        self.assertEqual(error["first_exceeded_value"], error["stage_errors"]["position_encoding"])

        dot_first = replace(
            tight,
            tolerances=ToleranceConfig(1e9, 1e-15, 1e-15),
        )
        error_dot = track_errors(
            sample,
            pe,
            score,
            dot_first,
            cache_key=stage_keys(sample, dot_first)["error_tracking"],
        )
        self.assertEqual(error_dot["first_exceeded_stage"], "dot_score")
        self.assertEqual(
            error_dot["first_exceeded_value"], error_dot["stage_errors"]["dot_score"]
        )

    def test_layer_statistics_use_null_for_empty_layers(self):
        diagnostics = [
            {
                "scores": [
                    {"relative_distance": 0, "score": 0.1},
                    {"relative_distance": 65, "score": 0.2},
                ]
            }
        ]
        rows = _layer_statistics(diagnostics, 64, 128)
        distance_ten = rows[10]
        self.assertTrue(distance_ten["layers"]["in_training"]["empty_layer"])
        self.assertIsNone(distance_ten["layers"]["in_training"]["mean"])
        self.assertIsNone(distance_ten["layers"]["extrapolation"]["max"])
        self.assertFalse(rows[65]["layers"]["extrapolation"]["empty_layer"])


class EngineTests(unittest.TestCase):
    def test_quantization_invalidates_only_downstream_stages(self):
        config, samples = load_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            engine = WorkbenchEngine(config, samples[:4], str(Path(temporary) / "state"))
            first = engine.run()
            self.assertEqual(len(first.completed_samples), 4)
            engine.set_config(replace(config, quantization_level="coarse"))
            second = engine.run()
            self.assertFalse(second.pending_samples)
            reused_stages = {(row["sample_id"], row["stage"]) for row in second.reused}
            recomputed_stages = {(row["sample_id"], row["stage"]) for row in second.recomputed}
            for sample in samples[:4]:
                self.assertIn((sample.sample_id, "position_encoding"), reused_stages)
                self.assertIn((sample.sample_id, "scoring"), recomputed_stages)
                self.assertIn((sample.sample_id, "error_tracking"), recomputed_stages)
                self.assertNotIn((sample.sample_id, "position_encoding"), recomputed_stages)

            engine.set_config(replace(config, scheme="linear", quantization_level="coarse"))
            third = engine.run()
            self.assertFalse(third.pending_samples)
            self.assertTrue(all(row["stage"] != "position_encoding" for row in third.reused))

    def test_point_budget_quota_pending_and_resume(self):
        config, samples = load_fixture()
        selected = samples[:5]
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary) / "state"
            engine = WorkbenchEngine(config, selected, str(state_dir))
            partial = engine.run(point_budget=5, sample_quota=4, snapshot_interval=2)
            self.assertTrue(partial.pending_samples)
            self.assertTrue(partial.resume_cursor)
            self.assertTrue(partial.pending_issues)
            self.assertTrue(
                all(issue.code == "E_POINT_BUDGET_PENDING" for issue in partial.pending_issues)
            )
            self.assertGreater(len(partial.scheduler["checkpoints"]), 0)
            self.assertLessEqual(
                partial.scheduler["max_workers_observed"],
                config.thread_pool_size,
            )
            recovered = WorkbenchEngine(config, selected, str(state_dir))
            self.assertEqual(recovered.resume_state, "exact_resume")
            resumed = recovered.run(point_budget=500, sample_quota=32)
            self.assertFalse(resumed.pending_samples)
            self.assertEqual(len(resumed.completed_samples), len(selected))

            fresh_engine = WorkbenchEngine(config, selected, str(state_dir))
            exact = fresh_engine.run(point_budget=500, sample_quota=32)
            self.assertEqual(exact.resume_state, "exact_resume")
            self.assertEqual(exact.consumed_points, 0)
            self.assertEqual(len(exact.reused), len(selected) * 3)
            self.assertEqual(exact.recomputed, [])

    def test_snapshot_manifest_and_atomic_layout(self):
        config, samples = load_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary) / "state"
            engine = WorkbenchEngine(config, samples[:3], str(state_dir))
            result = engine.run()
            current = (state_dir / "CURRENT").read_text(encoding="utf-8").strip()
            snapshot_dir = state_dir / "snapshots" / current
            manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["config_hash"], result.config_hash)
            self.assertEqual(manifest["completed_schemes"], [config.scheme])
            self.assertEqual(len(manifest["completed_samples"]), 3)
            self.assertIn("consumed_budget", manifest)
            self.assertEqual(manifest["stages"], ["loading", "position_encoding", "scoring", "error_tracking"])
            self.assertEqual(len(list((state_dir / "snapshots").iterdir())), 1)
            staging = state_dir / ".staging"
            self.assertTrue(not staging.exists() or not any(staging.iterdir()))
            for stage in manifest["stages"]:
                self.assertTrue((snapshot_dir / "stages" / f"{stage}.json").exists())

    def test_resume_classification_uses_version_and_config_hash(self):
        config, samples = load_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary) / "state"
            engine = WorkbenchEngine(config, samples[:2], str(state_dir))
            engine.run()

            changed = WorkbenchEngine(
                replace(config, quantization_level="coarse"),
                samples[:2],
                str(state_dir),
            )
            self.assertEqual(changed.resume_state, "incremental")
            self.assertTrue(changed.caches["position_encoding"])

            current = (state_dir / "CURRENT").read_text(encoding="utf-8").strip()
            manifest_path = state_dir / "snapshots" / current / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            incompatible = WorkbenchEngine(config, samples[:2], str(state_dir))
            self.assertEqual(incompatible.resume_state, "rebuild")
            self.assertFalse(incompatible.caches["position_encoding"])
            self.assertTrue(
                any(issue.code == "E_SNAPSHOT_VERSION" for issue in incompatible.loaded_snapshot.issues)
            )

    def test_incremental_and_cold_rebuild_reports_are_identical(self):
        config, samples = load_fixture()
        selected = samples[:4]
        final_config = replace(config, scheme="ntk", quantization_level="coarse")
        with tempfile.TemporaryDirectory() as temporary:
            incremental_state = Path(temporary) / "incremental-state"
            incremental = WorkbenchEngine(config, selected, str(incremental_state))
            incremental.run()
            incremental.set_config(replace(config, quantization_level="coarse"))
            incremental.run()
            incremental.set_config(final_config)
            incremental_result = incremental.run()
            incremental_report = build_diagnostic_report(
                incremental,
                incremental_result,
                sensitivity=None,
                validation_issues=[],
                command_issues=[],
            )

            cold_state = Path(temporary) / "cold-state"
            cold = WorkbenchEngine(final_config, selected, str(cold_state))
            cold_result = cold.run()
            cold_report = build_diagnostic_report(
                cold,
                cold_result,
                sensitivity=None,
                validation_issues=[],
                command_issues=[],
            )
            self.assertEqual(incremental_report["report_id"], cold_report["report_id"])
            self.assertEqual(stable_json(incremental_report), stable_json(cold_report))

    def test_visual_summary_has_stable_stage_and_sensitivity_data(self):
        config, samples = load_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            engine = WorkbenchEngine(config, samples[:3], str(Path(temporary) / "state"))
            result = engine.run()
            sensitivity = SensitivityAnalyzer(
                config,
                samples[:3],
                {
                    "scheme_candidates": ["rope", "linear", "ntk"],
                    "precision_levels": ["fp32", "fp64"],
                    "quantization_levels": ["fine", "coarse"],
                    "fixed_decimal_bits": [10],
                    "fixed_decimal_widths": [8, 10],
                    "data_scales": [0.5, 1.0],
                },
            ).analyze()
            report = build_diagnostic_report(engine, result, sensitivity, [], [])
            visual = build_visual_summary(report)
            self.assertEqual(len(visual["stage_error_bars"]), 3)
            self.assertTrue(visual["sensitivity_ranking"])
            self.assertIn("position_encoding_angle", visual["flip_attribution_counts"])
            self.assertTrue(report["layer_statistics"]["buckets"])
            self.assertTrue(report["boundary_sensitive_samples"])


class CommandExecutorTests(unittest.TestCase):
    def test_command_execution_writes_all_report_artifacts(self):
        config, samples = load_fixture()
        commands = parse_commands_jsonl(SAMPLES / "commands.jsonl", config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executor = CommandExecutor(
                config,
                samples[:3],
                [
                    {"command": "run", "params": {}, "line": 1},
                    {
                        "command": "sensitivity",
                        "params": {
                            "scheme_candidates": ["rope", "linear", "ntk"],
                            "precision_levels": ["fp32", "fp64"],
                            "quantization_levels": ["fine", "coarse"],
                            "fixed_decimal_widths": [8, 10],
                            "data_scales": [0.5, 1.0],
                        },
                        "line": 2,
                    },
                    {"command": "report", "params": {}, "line": 3},
                ],
                state_dir=root / "state",
                reports_dir=root / "reports",
                command_issues=commands.issues,
            )
            result = executor.execute()
            self.assertEqual(result.exit_code, 0)
            for name in (
                "diagnostic_report.json",
                "visual_summary.json",
                "summary.svg",
                "dashboard.html",
            ):
                self.assertTrue((root / "reports" / name).exists(), name)
            report = json.loads((root / "reports" / "diagnostic_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["sensitivity"]["ranking"])
            self.assertEqual(report["metadata"]["completed_sample_count"], 3)
            svg = (root / "reports" / "summary.svg").read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertIn("Boundary-sensitive samples", svg)


if __name__ == "__main__":
    unittest.main()
