from __future__ import annotations

import unittest

from quantcalib.calibration import calibrate_group, percentile_value
from quantcalib.model import Layer, LayerQuantConfig
from quantcalib.quantization import quantize_group
from quantcalib.utils import canonical_json_bytes


def make_layer(weights: list[float], name: str = "layer") -> Layer:
    return Layer(
        name=name,
        shape=(len(weights),),
        weights=tuple(weights),
        content_version=1,
        group_granularity=4,
        source_line=1,
    )


def make_cfg(**overrides) -> LayerQuantConfig:
    values = {
        "scheme": "asymmetric",
        "bit_width": 8,
        "group_granularity": 4,
        "lower_permille": 100,
        "upper_permille": 900,
        "fractional_bits": 8,
        "quantization_levels": 1,
        "scale_effective_digits": 15,
    }
    values.update(overrides)
    return LayerQuantConfig(**values)


class CalibrationQuantizationTests(unittest.TestCase):
    def test_linear_percentile_rule(self) -> None:
        values = [0.0, 10.0, 20.0, 30.0]
        self.assertEqual(percentile_value(values, 0), 0.0)
        self.assertEqual(percentile_value(values, 1000), 30.0)
        self.assertAlmostEqual(percentile_value(values, 333), 9.99)

    def test_symmetric_zero_point_is_zero(self) -> None:
        layer = make_layer([-2.0, -0.5, 0.5, 2.0])
        cfg = make_cfg(scheme="symmetric", lower_permille=0, upper_permille=1000)
        group = calibrate_group(layer, cfg, 0, calibration_sample_ids=["s0"])
        self.assertEqual(group["zero"], 0)
        self.assertGreater(group["scale"], 0.0)
        self.assertEqual(group["percentile_metadata"]["interpolation"], "linear_n_minus_1")

    def test_calibration_artifact_is_byte_deterministic(self) -> None:
        layer = make_layer([0.25, -0.5, 1.25, 2.5, -2.0, 0.1, 0.2, 0.3])
        cfg = make_cfg()
        first = calibrate_group(layer, cfg, 0, calibration_sample_ids=["s0", "s1"])
        second = calibrate_group(layer, cfg, 0, calibration_sample_ids=["s0", "s1"])
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))

    def test_profile_changes_integer_and_dequantized_values(self) -> None:
        layer = make_layer([2.5, -0.98, -0.09, -0.51])
        cfg_a = make_cfg(fractional_bits=8, quantization_levels=1)
        cfg_b = make_cfg(fractional_bits=4, quantization_levels=2)
        group_a = calibrate_group(layer, cfg_a, 0, calibration_sample_ids=["s0"])
        group_b = calibrate_group(layer, cfg_b, 0, calibration_sample_ids=["s0"])
        quant_a = quantize_group(
            layer, cfg_a, group_a, batch_point_budget=2, calibration_sample_ids=["s0"]
        )
        quant_b = quantize_group(
            layer, cfg_b, group_b, batch_point_budget=2, calibration_sample_ids=["s0"]
        )
        self.assertNotEqual(quant_a["fixed_integers"], quant_b["fixed_integers"])
        self.assertNotEqual(quant_a["dequantized"], quant_b["dequantized"])
        self.assertNotEqual(quant_a["role_fingerprint"], quant_b["role_fingerprint"])
        self.assertEqual(quant_a["role"]["retained_fractional_bits"], 8)
        self.assertEqual(quant_b["role"]["retained_fractional_bits"], 4)

    def test_saturation_records_direction_and_pre_value(self) -> None:
        layer = make_layer([-100.0, -50.0, 50.0, 100.0])
        cfg = make_cfg(bit_width=4, lower_permille=250, upper_permille=750)
        group = calibrate_group(layer, cfg, 0, calibration_sample_ids=["s0"])
        quant = quantize_group(
            layer, cfg, group, batch_point_budget=2, calibration_sample_ids=["s0"]
        )
        directions = {item["direction"] for item in quant["truncations"]}
        self.assertEqual(directions, {"low", "high"})
        for item in quant["truncations"]:
            self.assertIn("pre_saturation_fixed", item)
            self.assertIn("pre_saturation_integer", item)
        self.assertEqual(len(quant["fixed_integers"]), 4)
        self.assertEqual(
            sum(1 for start, end in ((0, 2), (2, 4)) if end - start <= 2), 2
        )


if __name__ == "__main__":
    unittest.main()
