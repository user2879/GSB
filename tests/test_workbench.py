from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rope_diag import cli
from rope_diag.attention import aggregate_stratified_statistics, compute_attention
from rope_diag.config import Tolerances, parse_config
from rope_diag.encoding import (
    encode_sample,
    probe_positions,
    reference_encoding,
    validate_scheme_numeric_inputs,
)
from rope_diag.engine import DiagnosticEngine, EngineOptions
from rope_diag.errors import (
    ConfigError,
    E_BASE_ZERO,
    E_CONFIG_MISMATCH,
    E_SCALE_INVALID,
    WorkbenchError,
)
from rope_diag.samples import parse_sample
from rope_diag.tracing import STAGE_ORDER, trace_attention


SAMPLE_SPECS = [
    ("s-near", 5, [2, 4]),
    ("s-boundary", 8, [0, 8]),
    ("s-extrap", 20, [0, 8, 12]),
    ("s-inj-nonfinite", 15, [5]),
    ("s-inj-position", 16, [4]),
    ("s-inj-base", 18, [6]),
    ("s-inj-scale", 19, [7]),
]


def make_config(
    *,
    schemes: list[str] | None = None,
    concurrency: int = 4,
    train_context_length: int = 8,
    base: float = 10000.0,
    scaling_factor: float = 2.0,
    dimension: int = 8,
    base_values: list[float] | None = None,
    scale_values: list[float] | None = None,
) -> dict:
    return {
        "schemes": schemes or ["RoPE", "LinearInterpolation", "NTK-aware"],
        "base": base,
        "train_context_length": train_context_length,
        "extrapolation_target_length": 32,
        "dimension": dimension,
        "scaling_factor": scaling_factor,
        "precision": "float64",
        "concurrency": concurrency,
        "tolerances": {
            "position_vector": 1e-6,
            "dot_score": 1e-6,
            "normalized_attention_weight": 1e-6,
        },
        "sensitivity": {
            "base_values": base_values or [1000.0, 10000.0, 20000.0],
            "scaling_factor_values": scale_values or [1.0, 2.0, 4.0],
        },
    }


def make_sample(sample_id: str, query_position: int, key_positions: list[int]) -> dict:
    # String hash randomization would hurt fixture reproducibility across processes.
    rng = np.random.default_rng(sum(ord(char) for char in sample_id) + query_position)
    return {
        "sample_id": sample_id,
        "query_position": query_position,
        "query_features": rng.normal(0.0, 0.3, size=8).round(12).tolist(),
        "key_positions": key_positions,
        "key_features": [
            rng.normal(0.0, 0.3, size=8).round(12).tolist() for _ in key_positions
        ],
    }


def write_case(
    root: Path,
    *,
    config: dict | None = None,
    injections: list[dict] | None = None,
    samples: list[dict] | None = None,
) -> tuple[Path, Path, Path]:
    config_path = root / "config.json"
    samples_path = root / "samples.jsonl"
    injections_path = root / "injections.json"
    config_path.write_text(json.dumps(config or make_config(), indent=2), encoding="utf-8")
    sample_rows = samples or [make_sample(*spec) for spec in SAMPLE_SPECS]
    samples_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in sample_rows)
        + "\n",
        encoding="utf-8",
    )
    injections_path.write_text(
        json.dumps({"version": 1, "injections": injections or []}, indent=2),
        encoding="utf-8",
    )
    return config_path, samples_path, injections_path


def run_engine(
    root: Path,
    *,
    config: dict | None = None,
    injections: list[dict] | None = None,
    samples: list[dict] | None = None,
    state_name: str = "state",
    report_name: str = "report.json",
    resume: bool = False,
    stop_after: str | None = None,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    config_path, samples_path, injections_path = write_case(
        root,
        config=config,
        injections=injections,
        samples=samples,
    )
    engine = DiagnosticEngine(
        config_path=config_path,
        samples_path=samples_path,
        injections_path=injections_path,
        options=EngineOptions(
            state_dir=root / state_name,
            resume=resume,
            stop_after_stage=stop_after,
        ),
    )
    return engine.run(root / report_name)


class ConfigurationAndEncodingTests(unittest.TestCase):
    def test_three_scheme_interface_is_elementwise_deterministic(self) -> None:
        config = parse_config(make_config())
        sample = parse_sample(make_sample(*SAMPLE_SPECS[2]))
        for scheme in config.schemes:
            first = encode_sample(
                scheme,
                sample,
                config.dimension,
                config.base,
                config.scaling_factor,
                config.precision,
            )
            second = encode_sample(
                scheme,
                sample,
                config.dimension,
                config.base,
                config.scaling_factor,
                config.precision,
            )
            self.assertEqual(first.query_encoded.shape, (config.dimension,))
            self.assertEqual(first.key_encoded.shape, (len(sample.key_positions), config.dimension))
            self.assertEqual(first.cos_values.shape, (1 + len(sample.key_positions), config.dimension // 2))
            np.testing.assert_array_equal(first.query_encoded, second.query_encoded)
            np.testing.assert_array_equal(first.key_encoded, second.key_encoded)
            np.testing.assert_array_equal(first.cos_values, second.cos_values)

    def test_strict_configuration_validation_has_no_silent_defaults(self) -> None:
        missing_base = make_config()
        del missing_base["base"]
        with self.assertRaises(ConfigError) as context:
            parse_config(missing_base)
        self.assertEqual(context.exception.code, "E_CONFIG_INVALID")

        odd = make_config(dimension=7)
        with self.assertRaises(ConfigError) as context:
            parse_config(odd)
        self.assertEqual(context.exception.code, "E_CONFIG_INVALID")

        missing_scale = make_config()
        del missing_scale["scaling_factor"]
        with self.assertRaises(ConfigError) as context:
            parse_config(missing_scale)
        self.assertEqual(context.exception.code, "E_CONFIG_INVALID")

    def test_numeric_business_codes_and_boundary_probe(self) -> None:
        with self.assertRaises(WorkbenchError) as context:
            validate_scheme_numeric_inputs("RoPE", 0.0, 8, 2.0, "float64")
        self.assertEqual(context.exception.code, E_BASE_ZERO)
        with self.assertRaises(WorkbenchError) as context:
            validate_scheme_numeric_inputs("LinearInterpolation", 10000.0, 8, 0.0, "float64")
        self.assertEqual(context.exception.code, E_SCALE_INVALID)

        _, info = probe_positions("LinearInterpolation", [0, 20], 8, 10000.0, 1e300, "float64")
        self.assertIsNotNone(info)
        self.assertEqual(info["criteria"], "dimension_truncation")
        _, info = probe_positions("NTK-aware", [0, 20], 8, 10000.0, 1e300, "float64")
        self.assertIsNotNone(info)
        self.assertEqual(info["criteria"], "scaling_overflow")


class AttentionAndTracingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = parse_config(make_config())
        self.sample = parse_sample(make_sample(*SAMPLE_SPECS[2]))

    def test_boundary_equality_is_in_train_and_marked(self) -> None:
        encoding = encode_sample(
            "LinearInterpolation",
            self.sample,
            self.config.dimension,
            self.config.base,
            self.config.scaling_factor,
            self.config.precision,
        )
        reference = reference_encoding(
            self.sample,
            self.config.dimension,
            self.config.base,
            self.config.precision,
        )
        attention = compute_attention(
            self.sample,
            "LinearInterpolation",
            encoding,
            reference,
            self.config.dimension,
            self.config.train_context_length,
        )
        self.assertEqual(attention.layers, ("out_of_train", "out_of_train", "in_train"))
        self.assertEqual(attention.boundary_flags, (False, False, True))
        self.assertAlmostEqual(float(np.sum(attention.weights)), 1.0, places=14)
        statistics = aggregate_stratified_statistics([attention], self.config.train_context_length)
        self.assertEqual(statistics["boundary_samples"][0]["relative_distance"], 8)
        self.assertTrue(statistics["boundary_samples"][0]["boundary_marked"])
        self.assertEqual(statistics["layer_totals"][0]["layer"], "in_train")
        self.assertIn("p95", statistics["layer_totals"][0]["quantiles"])

    def test_first_exceeded_stage_wins_and_later_values_remain(self) -> None:
        encoding = encode_sample(
            "LinearInterpolation",
            self.sample,
            self.config.dimension,
            self.config.base,
            self.config.scaling_factor,
            self.config.precision,
        )
        reference = reference_encoding(
            self.sample,
            self.config.dimension,
            self.config.base,
            self.config.precision,
        )
        attention = compute_attention(
            self.sample,
            "LinearInterpolation",
            encoding,
            reference,
            self.config.dimension,
            self.config.train_context_length,
        )
        strict_traces = trace_attention(
            attention,
            encoding,
            Tolerances(1e-12, 1e-12, 1e-12),
            self.config.train_context_length,
        )
        self.assertEqual(len(strict_traces), 2)
        self.assertEqual(strict_traces[0]["first_exceeded_stage"], "position_vector")
        self.assertTrue(strict_traces[0]["later_stage_values_recorded"])
        self.assertEqual(strict_traces[0]["stage_order"], list(STAGE_ORDER))

        contradictory = trace_attention(
            attention,
            encoding,
            Tolerances(1e9, 1e-15, 1e9),
            self.config.train_context_length,
        )
        self.assertEqual(contradictory[0]["first_exceeded_stage"], "dot_score")
        self.assertTrue(contradictory[0]["contradictory_tolerance_judgement"])
        self.assertIn("normalized_attention_weight", contradictory[0]["stage_deviations"])
        self.assertGreater(
            contradictory[0]["stage_deviations"]["normalized_attention_weight"]["deviation"],
            0.0,
        )

        rope_encoding = encode_sample(
            "RoPE",
            self.sample,
            self.config.dimension,
            self.config.base,
            self.config.scaling_factor,
            self.config.precision,
        )
        rope_attention = compute_attention(
            self.sample,
            "RoPE",
            rope_encoding,
            reference,
            self.config.dimension,
            self.config.train_context_length,
        )
        rope_traces = trace_attention(
            rope_attention,
            rope_encoding,
            Tolerances(1e-12, 1e-12, 1e-12),
            self.config.train_context_length,
        )
        self.assertTrue(all(item["first_exceeded_stage"] is None for item in rope_traces))


class EndToEndTests(unittest.TestCase):
    def test_injection_hits_and_report_contract(self) -> None:
        injections = [
            {
                "injection_id": "INJ-001",
                "sample_id": "s-inj-nonfinite",
                "stage": "encoding",
                "fault": "nonfinite_feature",
                "target": "query_features",
                "index": 0,
                "value": "nan",
            },
            {
                "injection_id": "INJ-002",
                "sample_id": "s-inj-position",
                "stage": "scoring",
                "fault": "position_out_of_range",
                "target": "key_position",
                "key_index": 0,
                "position": 32,
            },
            {
                "injection_id": "INJ-003",
                "sample_id": "s-inj-base",
                "scheme": "RoPE",
                "stage": "encoding",
                "fault": "base_zero",
            },
            {
                "injection_id": "INJ-004",
                "sample_id": "s-inj-scale",
                "scheme": "LinearInterpolation",
                "stage": "encoding",
                "fault": "scale_nonpositive",
                "value": -1.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = run_engine(Path(tmp), injections=injections)
        codes = {failure["code"] for failure in report["failures"]}
        self.assertEqual(
            codes,
            {
                "E_NONFINITE_INPUT",
                "E_POSITION_OUT_OF_RANGE",
                "E_BASE_ZERO",
                "E_SCALE_INVALID",
            },
        )
        self.assertEqual(len(report["injection_hits"]), 8)
        self.assertEqual(len(report["injection_hash"]), 64)
        by_sample = {item["sample_id"]: item for item in report["injection_hits_by_sample"]}
        self.assertEqual(by_sample["s-inj-base"]["injection_ids"], ["INJ-003"])
        self.assertEqual(by_sample["s-inj-scale"]["injection_ids"], ["INJ-004"])
        self.assertEqual(by_sample["s-inj-position"]["codes"], ["E_POSITION_OUT_OF_RANGE"])
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["diagnostic_digest"])

    def test_sensitivity_ranking_tie_break_and_boundaries(self) -> None:
        config = make_config(
            base_values=[0.0, 10000.0, 20000.0, 1e300],
            scale_values=[0.0, 1.0, 2.0, 4.0, 1e300],
        )
        with tempfile.TemporaryDirectory() as tmp:
            report = run_engine(Path(tmp), config=config)
        base_ranking = next(
            item
            for item in report["sensitivity"]["rankings"]
            if item["scheme"] == "RoPE" and item["parameter"] == "base"
        )
        self.assertNotIn(0.0, [entry["parameter_value"] for entry in base_ranking["ranking"]])
        scale_ranking = next(
            item
            for item in report["sensitivity"]["rankings"]
            if item["scheme"] == "RoPE" and item["parameter"] == "scaling_factor"
        )
        self.assertEqual(
            [entry["parameter_value"] for entry in scale_ranking["ranking"]],
            [1.0, 2.0, 4.0, 1e300],
        )
        boundary_criteria = {
            item["criteria"] for item in report["sensitivity"]["boundary_sensitive_samples"]
        }
        self.assertIn("base_zero", boundary_criteria)
        self.assertIn("scale_nonpositive", boundary_criteria)
        self.assertIn("dimension_truncation", boundary_criteria)
        self.assertIn("scaling_overflow", boundary_criteria)

    def test_concurrency_matches_serial_ordering_and_cold_reports_are_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            serial = run_engine(
                root / "serial",
                config=make_config(concurrency=1),
                state_name="state",
                report_name="serial.json",
            )
            concurrent = run_engine(
                root / "concurrent",
                config=make_config(concurrency=8),
                state_name="state",
                report_name="concurrent.json",
            )
            concurrent_again = run_engine(
                root / "concurrent-again",
                config=make_config(concurrency=8),
                state_name="state",
                report_name="concurrent.json",
            )

        def normalized(report: dict) -> dict:
            value = copy.deepcopy(report)
            value.pop("config_hash", None)
            value.pop("diagnostic_digest", None)
            value["configuration"].pop("concurrency", None)
            value["execution"].pop("concurrency", None)
            return value

        self.assertEqual(normalized(serial), normalized(concurrent))
        self.assertEqual(concurrent, concurrent_again)
        self.assertEqual(
            [(trace["scheme"], trace["sample_id"], trace["key_index"]) for trace in concurrent["error_traces"]],
            [(trace["scheme"], trace["sample_id"], trace["key_index"]) for trace in serial["error_traces"]],
        )


class CacheAndSnapshotTests(unittest.TestCase):
    def test_warm_reuse_and_stage_scoped_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config()
            first = run_engine(root, config=config)
            second = run_engine(root, config=config, report_name="warm.json")
            self.assertEqual(len(first["reuse_manifest"]["recomputed"]), 63)
            self.assertEqual(len(second["reuse_manifest"]["reused"]), 63)
            self.assertEqual(len(second["reuse_manifest"]["recomputed"]), 0)
            self.assertEqual(first["diagnostic_digest"], second["diagnostic_digest"])

            changed_train = copy.deepcopy(config)
            changed_train["train_context_length"] = 9
            train_report = run_engine(root, config=changed_train, report_name="train.json")
            recomputed = train_report["reuse_manifest"]["recomputed"]
            self.assertFalse(any(item["stage"] == "encoding" for item in recomputed))
            affected = {
                (item["stage"], item["sample_id"])
                for item in recomputed
            }
            self.assertEqual(
                affected,
                {
                    (stage, sample_id)
                    for stage in ("scoring", "tracing")
                    for sample_id in ("s-boundary", "s-extrap")
                },
            )

            changed_base = copy.deepcopy(changed_train)
            changed_base["base"] = 20000.0
            base_report = run_engine(root, config=changed_base, report_name="base.json")
            self.assertEqual(len(base_report["reuse_manifest"]["recomputed"]), 63)
            self.assertEqual(len(base_report["reuse_manifest"]["invalidated"]), 63)

            samples = [make_sample(*spec) for spec in SAMPLE_SPECS]
            near = next(item for item in samples if item["sample_id"] == "s-near")
            near["query_features"][0] = float(near["query_features"][0]) + 0.125
            sample_report = run_engine(
                root,
                config=changed_base,
                samples=samples,
                report_name="sample.json",
            )
            recomputed_samples = {
                item["sample_id"] for item in sample_report["reuse_manifest"]["recomputed"]
            }
            self.assertEqual(recomputed_samples, {"s-near"})
            self.assertEqual(len(sample_report["reuse_manifest"]["recomputed"]), 9)

    def test_resume_after_completed_stage_and_reject_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config()
            interrupted = run_engine(root, config=config, stop_after="encoding")
            self.assertEqual(interrupted["status"], "interrupted")
            manifest = json.loads((root / "state" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "interrupted")
            self.assertEqual(manifest["completed_stages"], ["encoding"])
            resumed = run_engine(root, config=config, resume=True, report_name="resumed.json")
            self.assertEqual(resumed["status"], "complete")
            self.assertTrue(
                all(item["stage"] != "encoding" for item in resumed["reuse_manifest"]["recomputed"])
            )
            self.assertTrue(any(item["stage"] == "encoding" for item in resumed["reuse_manifest"]["reused"]))
            self.assertFalse(any(path.suffix == ".tmp" for path in (root / "state").rglob("*")))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config()
            run_engine(root, config=config, stop_after="encoding")
            changed = copy.deepcopy(config)
            changed["base"] = 20000.0
            with self.assertRaises(WorkbenchError) as context:
                run_engine(root, config=changed, resume=True, report_name="bad.json")
            self.assertEqual(context.exception.code, E_CONFIG_MISMATCH)

    def test_cli_stop_signal_hook_exits_zero_and_resume_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, samples_path, injections_path = write_case(root)
            argv = [
                "--config",
                str(config_path),
                "--samples",
                str(samples_path),
                "--injections",
                str(injections_path),
                "--state-dir",
                str(root / "state"),
                "--report",
                str(root / "report.json"),
                "--stop-after-stage",
                "encoding",
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(argv), 0)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(argv[:-2] + ["--resume"]), 0)
            self.assertTrue((root / "report.json").exists())

    def test_cli_invalid_config_writes_failure_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, samples_path, injections_path = write_case(
                root,
                config=make_config(dimension=7),
            )
            argv = [
                "--config",
                str(config_path),
                "--samples",
                str(samples_path),
                "--injections",
                str(injections_path),
                "--state-dir",
                str(root / "state"),
                "--report",
                str(root / "report.json"),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(argv), 2)
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failures"][0]["code"], "E_CONFIG_INVALID")


if __name__ == "__main__":
    unittest.main()
