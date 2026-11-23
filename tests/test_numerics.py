from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rope_diag.attention import score_attention, stable_attention
from rope_diag.config import SensitivitySpec, Tolerances, load_config
from rope_diag.errors import EncodingBoundaryError
from rope_diag.models import EncodedSample
from rope_diag.rope import (
    apply_input_precision_and_quantization,
    apply_rotation,
    generate_position_encoding,
)
from rope_diag.samples import load_samples
from rope_diag.sensitivity import scan_scheme_sensitivity
from rope_diag.statistics import compute_statistics
from rope_diag.trace import compute_error_trace

from tests.support import sample_rows, write_case


class NumericsTests(unittest.TestCase):
    def _config(self, temporary: str):
        paths = write_case(Path(temporary), samples=sample_rows(count=2))
        config, failures, _ = load_config(paths["config"])
        self.assertEqual(failures, [])
        return config, paths

    def test_schemes_change_real_angles_and_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _ = self._config(temporary)
            encodings = {
                scheme: generate_position_encoding(scheme, 42, config)
                for scheme in ("rope", "linear_interpolation", "ntk_aware")
            }
            rope_theta = [pair.theta for pair in encodings["rope"].pairs]
            linear_theta = [pair.theta for pair in encodings["linear_interpolation"].pairs]
            ntk_theta = [pair.theta for pair in encodings["ntk_aware"].pairs]
            self.assertNotEqual(rope_theta, linear_theta)
            self.assertNotEqual(rope_theta, ntk_theta)
            self.assertAlmostEqual(encodings["linear_interpolation"].metadata["effective_position"], 10.5)
            self.assertGreater(encodings["ntk_aware"].metadata["effective_base"], config.base)
            repeat = generate_position_encoding("ntk_aware", 42, config)
            np.testing.assert_array_equal(encodings["ntk_aware"].vector, repeat.vector)

            changed_base = generate_position_encoding("rope", 42, config, base_override=20000.0)
            changed_scale = generate_position_encoding(
                "linear_interpolation", 42, config, scaling_override=8.0
            )
            self.assertNotEqual(rope_theta, [pair.theta for pair in changed_base.pairs])
            self.assertNotEqual(linear_theta, [pair.theta for pair in changed_scale.pairs])

    def test_rotation_preserves_norm_and_stable_softmax(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _ = self._config(temporary)
            encoding = generate_position_encoding("rope", 9, config)
            features = np.linspace(-1.0, 1.0, config.dimension)
            rotated = apply_rotation(features, encoding)
            self.assertAlmostEqual(float(np.linalg.norm(features)), float(np.linalg.norm(rotated)))
            scores = np.asarray([1000000.0, 1000001.0, 999999.0], dtype=np.float64)
            weights, maximum = stable_attention(scores)
            shifted_weights, _ = stable_attention(scores + 500000.0)
            self.assertAlmostEqual(float(np.sum(weights)), 1.0)
            self.assertEqual(maximum, 1000001.0)
            np.testing.assert_allclose(weights, shifted_weights, rtol=0.0, atol=1e-15)

    def test_training_window_statistics_and_error_trace_keep_all_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, paths = self._config(temporary)
            dataset = load_samples(paths["samples"], dimension=config.dimension)
            sample = dataset.samples[0]
            query_prepared = apply_input_precision_and_quantization(sample.query_features, config)
            keys_prepared = np.vstack(
                [apply_input_precision_and_quantization(row, config) for row in sample.key_features]
            )
            query_encoding = generate_position_encoding("rope", sample.query_position, config)
            key_encodings = tuple(
                generate_position_encoding("rope", position, config)
                for position in sample.key_positions
            )
            encoded = EncodedSample(
                scheme="rope",
                sample_id=sample.sample_id,
                query_encoding=query_encoding,
                key_encodings=key_encodings,
                query_rotated=apply_rotation(query_prepared, query_encoding),
                key_rotated=np.vstack(
                    [apply_rotation(row, key) for row, key in zip(keys_prepared, key_encodings)]
                ),
            )
            score_result = score_attention(
                encoded.query_rotated,
                encoded.key_rotated,
                list(range(config.dimension)),
                config.dimension_chunk_size,
            )
            statistics = compute_statistics([sample], {sample.sample_id: score_result}, config)
            self.assertGreater(statistics["in_training"]["score_count"], 0)
            self.assertGreater(statistics["extrapolation"]["score_count"], 0)

            strict_config = dataclasses.replace(
                config,
                tolerances=Tolerances(position_vector=0.0, score=0.0, attention=0.0),
            )
            trace = compute_error_trace(
                "rope",
                [sample],
                {sample.sample_id: encoded},
                {sample.sample_id: score_result},
                strict_config,
                list(range(config.dimension)),
                config.dimension_chunk_size,
            )
            self.assertEqual(len(trace["samples"]), 1)
            record = trace["samples"][0]
            self.assertEqual(len(record["stages"]), 3)
            self.assertTrue(all(stage["exceeded"] for stage in record["stages"]))
            self.assertTrue(all(stage["deviation"] is not None for stage in record["stages"]))
            self.assertEqual(record["first_exceeded_stage"], "position_vector")
            self.assertEqual(trace["stage_order"], ["position_vector", "score", "attention"])

    def test_no_extrapolation_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_case(Path(temporary), samples=sample_rows(count=2, no_extrapolation=True))
            config, failures, _ = load_config(paths["config"])
            self.assertEqual(failures, [])
            dataset = load_samples(paths["samples"], dimension=config.dimension)
            statistics = compute_statistics(list(dataset.samples), {}, config)
            self.assertEqual(statistics["status"], "E_NO_EXTRAPOLATION_SAMPLE")

    def test_sensitivity_sorting_and_overflow_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, paths = self._config(temporary)
            dataset = load_samples(paths["samples"], dimension=config.dimension)
            baseline = {}
            for sample in dataset.samples:
                query_prepared = apply_input_precision_and_quantization(sample.query_features, config)
                keys_prepared = np.vstack(
                    [apply_input_precision_and_quantization(row, config) for row in sample.key_features]
                )
                query_encoding = generate_position_encoding("ntk_aware", sample.query_position, config)
                key_encodings = tuple(
                    generate_position_encoding("ntk_aware", position, config)
                    for position in sample.key_positions
                )
                baseline[sample.sample_id] = score_attention(
                    apply_rotation(query_prepared, query_encoding),
                    np.vstack([apply_rotation(row, enc) for row, enc in zip(keys_prepared, key_encodings)]),
                    list(range(config.dimension)),
                    config.dimension_chunk_size,
                )
            extreme_config = dataclasses.replace(
                config,
                sensitivity=SensitivitySpec(base_step=1000.0, scale_step=1e308, radius=1),
            )
            report = scan_scheme_sensitivity(
                "ntk_aware",
                list(dataset.samples),
                baseline,
                extreme_config,
                list(range(config.dimension)),
                config.dimension_chunk_size,
            )
            self.assertTrue(report["entries"][0]["boundary_triggered"])
            conditions = [
                item["condition"]
                for entry in report["entries"]
                for item in entry["boundary_samples"]
            ]
            self.assertTrue(any("scaling_overflow" in condition for condition in conditions))
            self.assertEqual(report["neighbor_steps"]["base"], 1000.0)

            rope_baseline = {}
            for sample in dataset.samples:
                query_prepared = apply_input_precision_and_quantization(sample.query_features, config)
                keys_prepared = np.vstack(
                    [apply_input_precision_and_quantization(row, config) for row in sample.key_features]
                )
                query_encoding = generate_position_encoding("rope", sample.query_position, config)
                key_encodings = tuple(
                    generate_position_encoding("rope", position, config)
                    for position in sample.key_positions
                )
                rope_baseline[sample.sample_id] = score_attention(
                    apply_rotation(query_prepared, query_encoding),
                    np.vstack([apply_rotation(row, enc) for row, enc in zip(keys_prepared, key_encodings)]),
                    list(range(config.dimension)),
                    config.dimension_chunk_size,
                )
            rope_report = scan_scheme_sensitivity(
                "rope",
                list(dataset.samples),
                rope_baseline,
                config,
                list(range(config.dimension)),
                config.dimension_chunk_size,
            )
            scale_entries = [entry for entry in rope_report["entries"] if entry["parameter"] == "scaling_factor"]
            values = [entry["value"] for entry in scale_entries]
            self.assertEqual(values, sorted(values))
            self.assertTrue(all(entry["deviation_abs"] == 0.0 for entry in scale_entries))

    def test_dimension_truncation_boundary_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _ = self._config(temporary)
            large = dataclasses.replace(config, dimension=2048)
            with self.assertRaises(EncodingBoundaryError) as context:
                generate_position_encoding("rope", 5, large, base_override=1e308)
            self.assertIn("dimension_truncation", context.exception.details["condition"])


if __name__ == "__main__":
    unittest.main()
