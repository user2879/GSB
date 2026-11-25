import unittest

import numpy as np

from quantcalib.calib.quantiles import calibrate_group, fixed_quantile
from quantcalib.quant.kernel import quantize_dequantize, quantize_group
from quantcalib.data.config import QuantConfig
from quantcalib.distill.correction import correct
from quantcalib.data.samples import generate_calibration, generate_weights


class QuantileTests(unittest.TestCase):
    def test_fixed_quantile_determinism_and_tie(self):
        values = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
        ordered = np.sort(values)
        first, rank_a = fixed_quantile(ordered, 250)
        second, rank_b = fixed_quantile(ordered, 250)
        self.assertEqual(first, second)
        self.assertEqual(rank_a, rank_b)
        # Lower index wins the tie.
        self.assertLessEqual(rank_a, 1)

    def test_scale_zero_and_bit_switch_change_integers(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=16)
        calib_8 = calibrate_group("l", 0, values, 8, 6, 10, 990, "asymmetric")
        calib_4 = calibrate_group("l", 0, values, 4, 6, 10, 990, "asymmetric")
        q8 = quantize_dequantize(values, calib_8.scale, calib_8.zero_point,
                                 8, "half_even", "saturate")
        q4 = quantize_dequantize(values, calib_4.scale, calib_4.zero_point,
                                 4, "half_even", "saturate")
        self.assertFalse(np.array_equal(q8["codes"], q4["codes"]))
        self.assertGreaterEqual(q4["codes"].min(), -(1 << 3))
        self.assertLessEqual(q4["codes"].max(), (1 << 3) - 1)

    def test_saturation_records_direction_and_pre_clip(self):
        values = np.array([0.0, 0.1, 5.0, -5.0])
        calib = calibrate_group("l", 0, values, 4, 4, 100, 900, "asymmetric")
        result = quantize_group("l", 0, values, calib, 4, 0.01,
                                "half_even", "saturate")
        directions = {item["direction"] for item in result.clipped}
        self.assertIn("up", directions)
        self.assertTrue(all("pre_clip_value" in item for item in result.clipped))
        self.assertTrue(any(result.over_tolerance))
        self.assertIn("saturation", result.error_sources)

    def test_distillation_changes_outputs_and_is_deterministic(self):
        layers = generate_weights()
        calib = generate_calibration(layers)
        full = [layer.weights for layer in layers]
        cfg = QuantConfig()
        quant = []
        sat = []
        for layer in layers:
            group = layer.weights.reshape(-1)[:layer.group_granularity]
            c = calibrate_group(layer.name, 0, group, 8, 6, 10, 990, "asymmetric")
            q = quantize_group(layer.name, 0, group, c, 8, 0.05,
                               "half_even", "saturate")
            rebuilt = np.array(q.dequantized)
            whole = layer.weights.reshape(-1).copy()
            whole[:rebuilt.size] = rebuilt
            quant.append(whole.reshape(layer.weights.shape))
            sat.append(q.saturation_count)
        names = [layer.name for layer in layers]
        r1 = correct(full, quant, calib.inputs, calib.reference_outputs,
                     names, 2.0, 4, 0.5, 0.1, sat)
        r2 = correct(full, quant, calib.inputs, calib.reference_outputs,
                     names, 2.0, 4, 0.5, 0.1, sat)
        self.assertEqual(
            r1.sample_deviations_after, r2.sample_deviations_after)
        self.assertNotEqual(
            r1.sample_deviations_before, r1.sample_deviations_after)


if __name__ == "__main__":
    unittest.main()
