from __future__ import annotations

import copy
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from vibpipe.config import LoadedConfig, load_and_validate_config, validate_config_object
from vibpipe.engine import run_pipeline
from vibpipe.fixedpoint import fixed_ln, round_div
from vibpipe.generator import write_example_bundle
from vibpipe.precision import classify_disagreements
from vibpipe.samples import load_and_validate_samples
from vibpipe.stats import drift_decision


ROOT = Path(__file__).resolve().parents[1]


class FixedPointTests(unittest.TestCase):
    def test_round_div_half_up(self) -> None:
        self.assertEqual(round_div(5, 2), 3)
        self.assertEqual(round_div(-5, 2), -3)
        self.assertEqual(round_div(4, 2), 2)

    def test_fixed_ln_near_one(self) -> None:
        scale = 10**12
        result = fixed_ln(scale, scale, 14)
        self.assertLess(abs(result), 1000)
        two = fixed_ln(2 * scale, scale, 14)
        self.assertAlmostEqual(two / scale, 0.693147180559, places=6)


class ValidationTests(unittest.TestCase):
    def test_example_config_is_valid(self) -> None:
        config = _raw_config()
        self.assertEqual(validate_config_object(config), [])

    def test_cycle_width_gear_and_threshold_detected(self) -> None:
        config = _raw_config()
        config["stages"][0]["upstream"] = "clip"
        errors = validate_config_object(config)
        self.assertTrue(any("cycle detected" in message for _, message in errors))

        config = _raw_config()
        config["fixed_point"]["decimal_width"] = 0
        self.assertTrue(any("positive integer" in m for _, m in validate_config_object(config)))

        config = _raw_config()
        config["fixed_point"]["quantization_gear"] = "NOPE"
        self.assertTrue(any("quantization_gear" in p for p, _ in validate_config_object(config)))

        config = _raw_config()
        first = next(iter(config["thresholds_bp"]))
        config["thresholds_bp"][first] = 99999
        self.assertTrue(any("threshold" in m for _, m in validate_config_object(config)))

    def test_invalid_samples_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, samples_path = write_example_bundle(root)
            rows = samples_path.read_text(encoding="utf-8").splitlines()
            rows[11] = rows[10]
            bad = root / "bad.jsonl"
            bad.write_text("\n".join(rows) + "\n", encoding="utf-8")
            config = _loaded_config(root)
            samples, errors, _ = load_and_validate_samples(bad, len(config.channels))
            self.assertTrue(errors)
            self.assertTrue(any(e.code == "E_SAMPLE_INVALID" for e in errors))


class DriftBoundaryTests(unittest.TestCase):
    def _stat(self, mean: int) -> dict:
        return {
            "feature_id": "clip:x",
            "feature_name": "x",
            "scale": 10**12,
            "mean_q": mean,
            "sample_set": [1, 2],
            "count": 2,
            "missing_or_imputed_count": 0,
        }

    def test_exact_threshold_equality_is_drift_and_boundary(self) -> None:
        ref = self._stat(10_000)
        obs = self._stat(10_100)
        decision = drift_decision(ref, obs, 100, "Q6", 1)
        self.assertTrue(decision["drift"])
        self.assertTrue(decision["boundary"])
        self.assertEqual(decision["boundary_reason"], "exact_threshold_equality")

    def test_half_step_boundary(self) -> None:
        ref = self._stat(10_000_000)
        obs = self._stat(10_001_400)
        decision = drift_decision(ref, obs, 1, "Q6", 1)
        self.assertTrue(decision["drift"])
        self.assertTrue(decision["boundary"])
        self.assertEqual(decision["boundary_reason"], "within_half_gear_step")


class ClassificationTests(unittest.TestCase):
    def test_classification_categories(self) -> None:
        config = LoadedConfig(ROOT / "examples/local_sample/pipeline_config.json", _raw_config(), "fp")
        products = {
            "clip": {"stage": "clip", "features": [{"id": "clip:x", "source_feature": "interaction:x"}]},
            "interaction": {
                "stage": "interaction",
                "type": "interaction",
                "features": [{"id": "interaction:x", "pair": [0, 1]}],
            },
            "log_stage": {"stage": "log_stage", "type": "log", "features": [{"id": "log:a"}, {"id": "log:b"}]},
            "bucket": {"stage": "bucket", "type": "bucket", "features": [{"id": "bucket:a"}, {"id": "bucket:b"}]},
            "standardize": {
                "stage": "standardize",
                "type": "standardize",
                "features": [{"id": "std:a"}, {"id": "std:b"}],
            },
            "source": {"features": [{"id": "source:a"}, {"id": "source:b"}]},
        }
        normal = [_drift_row(False, False)]
        high = [_drift_row(True, False)]

        boundary = classify_disagreements(
            config,
            products,
            [{**_drift_row(False, True), "feature_id": "clip:x"}],
            [{**_drift_row(True, True), "feature_id": "clip:x"}],
            [],
            ["source", "standardize", "bucket", "log_stage", "interaction", "clip"],
            10,
        )
        self.assertEqual(boundary[0]["classification"], "threshold_boundary_decision")

        log_rows = [
            _stage_diff("standardize", "std:a", 1, False),
            _stage_diff("log_stage", "log:a", 100, True),
        ]
        log_case = classify_disagreements(
            config, products, normal, high, log_rows,
            ["source", "standardize", "bucket", "log_stage", "interaction", "clip"], 10
        )
        self.assertEqual(log_case[0]["classification"], "log_domain_approximation")

        quant_rows = [_stage_diff("standardize", "std:a", 100, True)]
        quant_case = classify_disagreements(
            config, products, normal, high, quant_rows,
            ["source", "standardize", "bucket", "log_stage", "interaction", "clip"], 10
        )
        self.assertEqual(quant_case[0]["classification"], "quantization_truncation")


class PipelineSmokeTests(unittest.TestCase):
    def test_serial_and_concurrent_products_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, samples_path = write_example_bundle(root)
            config, errors = load_and_validate_config(config_path)
            self.assertEqual(errors, [])
            samples, sample_errors, _ = load_and_validate_samples(samples_path, len(config.channels))
            self.assertEqual(sample_errors, [])
            serial = run_pipeline(config, samples, root / "serial", max_workers=1)
            concurrent = run_pipeline(config, samples, root / "concurrent", max_workers=4)
            self.assertEqual(
                serial["report"]["stats"],
                concurrent["report"]["stats"],
            )
            self.assertEqual(serial["products"], concurrent["products"])


def _raw_config() -> dict:
    path = ROOT / "examples/local_sample/pipeline_config.json"
    if path.exists():
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        config_path, _ = write_example_bundle(tmp)
        import json

        return json.loads(config_path.read_text(encoding="utf-8"))


def _loaded_config(root: Path) -> LoadedConfig:
    config, errors = load_and_validate_config(root / "pipeline_config.json")
    assert config is not None
    assert not errors
    return config


def _drift_row(drift: bool, boundary: bool) -> dict:
    return {
        "feature_id": "clip:x",
        "drift": drift,
        "boundary": boundary,
        "threshold_bp": 100,
        "drift_bp_rounded": 90 if not drift else 110,
        "exact_compare": {"left": 9, "threshold_right": 10},
        "distance_to_threshold_q": "1000",
        "half_gear_step_q": "500000",
        "boundary_reason": "within_half_gear_step" if boundary else None,
    }


def _stage_diff(stage: str, feature_id: str, diff: int, exceeds: bool) -> dict:
    return {
        "stage": stage,
        "feature_id": feature_id,
        "diff_canonical_q": diff,
        "exceeds_tolerance": exceeds,
    }


if __name__ == "__main__":
    unittest.main()
