import unittest

from arbworkbench.errors import GridError
from arbworkbench.fixedpoint import (
    MAX_EXPONENT,
    MIN_EXPONENT,
    SAT_HIGH,
    SAT_LOW,
    FixedPoint,
)


class FixedPointTests(unittest.TestCase):
    def test_quantize_half_away_from_zero(self):
        fp = FixedPoint(2)
        self.assertEqual(fp.quantize(100.25), 10025)
        self.assertEqual(fp.quantize(100.005), 10001)  # half rounds away
        self.assertEqual(fp.quantize(-100.005), -10001)
        self.assertEqual(fp.quantize(7), 700)

    def test_negative_exponent_scale(self):
        fp = FixedPoint(-3)
        # q-units: real 1500 -> 1500 * 10^-3 = 1.5 -> half rounds away -> 2
        self.assertEqual(fp.quantize(1500), 2)
        self.assertEqual(fp.quantize(1000), 1)
        self.assertEqual(fp.to_real(2), 2000.0)
        self.assertEqual(fp.quantum_real, 1000.0)

    def test_out_of_grid_exponent_raises_grid(self):
        with self.assertRaises(GridError) as ctx:
            FixedPoint(MAX_EXPONENT + 1)
        self.assertEqual(ctx.exception.code, "E_GRID")
        with self.assertRaises(GridError):
            FixedPoint(MIN_EXPONENT - 1)

    def test_weighted_mean_floor_and_saturation(self):
        fp = FixedPoint(0)
        self.assertEqual(fp.weighted_mean([(7, 5), (9, 3)]), 7)  # 62//8
        saturated = fp.add(SAT_HIGH, 1)
        self.assertEqual(saturated, SAT_HIGH)
        self.assertGreaterEqual(fp.saturation_events, 1)
        self.assertEqual(fp.add(SAT_LOW, -1), SAT_LOW)

    def test_midpoint_floor_picks_lower_on_odd_span(self):
        fp = FixedPoint(0)
        self.assertEqual(fp.midpoint_floor(6, 9), 7)   # 15//2
        self.assertEqual(fp.midpoint_floor(6, 8), 7)

    def test_error_bounds_grow_only_in_intermediate_term(self):
        fp = FixedPoint(2)
        bounds = fp.describe_bounds()
        self.assertEqual(bounds["single_observation_quantization_error"], 0.005)
        self.assertEqual(bounds["bound_majority_real"], 0.02)
        self.assertIn("linearly", bounds["note"])

    def test_division_by_zero_raises_grid(self):
        fp = FixedPoint(0)
        with self.assertRaises(GridError):
            fp.div_floor(10, 0)


if __name__ == "__main__":
    unittest.main()
