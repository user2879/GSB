from __future__ import annotations

import unittest

import numpy as np

from mixedprec_diag.deterministic import generate_artifacts
from mixedprec_diag.engine import InferenceEngine
from mixedprec_diag.inference import analyze_sample, fixed_forward, float_forward
from mixedprec_diag.quantization import INT32_MAX, INT32_MIN, quantize_array
from tests.helpers import config_payload
from mixedprec_diag.config import parse_config


class QuantizationTests(unittest.TestCase):
    def test_rounding_is_nearest_with_ties_away_from_zero(self) -> None:
        result = quantize_array(np.asarray([0.5, -0.5, 1.5, -1.5, 0.49, -0.49]), 0)
        self.assertEqual(result.as_list, [1, -1, 2, -2, 0, 0])

    def test_overflow_saturates_explicitly(self) -> None:
        result = quantize_array(np.asarray([3.0e9, -3.0e9, 1.0]), 0)
        self.assertEqual(result.values.tolist(), [INT32_MAX, INT32_MIN, 1])
        self.assertEqual(result.saturations, 2)

    def test_negative_power_is_exact_decimal_scale(self) -> None:
        result = quantize_array(np.asarray([15.0, -5.0, 4.9]), -1)
        self.assertEqual(result.as_list, [2, -1, 0])


class ForwardPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = generate_artifacts(31, 12, 7, 11)

    def test_fixed_and_float_paths_are_repeatable(self) -> None:
        sample = self.artifacts.dataset.inputs[3]
        first_float = float_forward(self.artifacts.model, sample)
        second_float = float_forward(self.artifacts.model, sample)
        np.testing.assert_array_equal(first_float.output, second_float.output)
        first_fixed = fixed_forward(self.artifacts.model, sample, 4)
        second_fixed = fixed_forward(self.artifacts.model, sample, 4)
        np.testing.assert_array_equal(first_fixed.output_q, second_fixed.output_q)
        np.testing.assert_array_equal(first_fixed.output_float, second_fixed.output_float)

    def test_layer_diagnostics_have_integer_bounds_and_shares(self) -> None:
        result = analyze_sample(self.artifacts.model, self.artifacts.dataset.inputs[0], 4)
        self.assertEqual(len(result["layer_diagnostics"]), 3)
        self.assertIn(result["dominant_layer"], (0, 1, 2))
        cumulative = result["layer_diagnostics"][-1]["cumulative_error_share_ppm"]
        self.assertEqual(cumulative, 1_000_000)
        for layer in result["layer_diagnostics"]:
            self.assertEqual(layer["rounding_quantization_limit_int_milli"], 500)
            self.assertGreaterEqual(layer["error_bound_int"], layer["observed_max_abs_error_int"])
            self.assertTrue(layer["error_bound_valid"])

    def test_attribution_is_batch_and_position_independent(self) -> None:
        config2 = parse_config(config_payload(batch_size_limit=2, precision="fixed"))
        config5 = parse_config(config_payload(batch_size_limit=5, precision="fixed"))
        report2 = InferenceEngine().run(config2, self.artifacts, config_version=1)
        report5 = InferenceEngine().run(config5, self.artifacts, config_version=1)
        by_id2 = {sample["sample_id"]: sample for sample in report2["samples"]}
        by_id5 = {sample["sample_id"]: sample for sample in report5["samples"]}
        self.assertEqual(sorted(by_id2), sorted(by_id5))
        for sample_id in by_id2:
            left = by_id2[sample_id]
            right = by_id5[sample_id]
            self.assertEqual(left["input_sha256"], right["input_sha256"])
            self.assertEqual(left["dominant_layer"], right["dominant_layer"])
            self.assertEqual(left["fixed_output_q"], right["fixed_output_q"])
            self.assertEqual(left["float_output"], right["float_output"])
            self.assertEqual(
                [layer["observed_max_abs_error_int"] for layer in left["layer_diagnostics"]],
                [layer["observed_max_abs_error_int"] for layer in right["layer_diagnostics"]],
            )

    def test_repeated_engine_run_has_identical_output_digests(self) -> None:
        config = parse_config(config_payload(batch_size_limit=3, precision="fixed"))
        left = InferenceEngine().run(config, self.artifacts, config_version=1)
        right = InferenceEngine().run(config, self.artifacts, config_version=1)
        self.assertEqual(left["path_output_sha256"], right["path_output_sha256"])
        self.assertEqual(
            [batch["output_sha256"] for batch in left["batches"]],
            [batch["output_sha256"] for batch in right["batches"]],
        )
        self.assertEqual(
            [sample["fixed_output_q"] for sample in left["samples"]],
            [sample["fixed_output_q"] for sample in right["samples"]],
        )


if __name__ == "__main__":
    unittest.main()
