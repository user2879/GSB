import unittest
from fractions import Fraction

from cfgdist.config.grid import Grid
from cfgdist.config import fixedpoint as fp


class FixedPointTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid.from_q(2)  # scale 100

    def test_quantize_truncates_toward_zero(self):
        positive = self.grid.quantize_int(Fraction(1234, 1000))  # 1.234 -> 1.23
        self.assertEqual(positive, 123)
        negative = self.grid.quantize_int(Fraction(-1234, 1000))  # -1.234 -> -1.23
        self.assertEqual(negative, -123)

    def test_quantization_error_bound(self):
        qv = fp.quantize_value(2.345, self.grid)
        self.assertEqual(qv.fixed.mantissa, 234)
        self.assertEqual(qv.trunc_error, Fraction(5, 1000))  # 0.005

    def test_integer_only_derivation(self):
        # u = 0.5 + 2*base_x - base_y ; base_x=1.25, base_y=2.345
        qx = fp.quantize_value(1.25, self.grid).fixed
        qy = fp.quantize_value(2.345, self.grid).fixed
        cx = fp.quantize_coefficient(2, self.grid)
        cy = fp.quantize_coefficient(-1, self.grid)
        fixed, local = fp.derive_with_values(
            "u", ["base_x", "base_y"],
            {"base_x": qx, "base_y": qy},
            {"base_x": cx, "base_y": cy},
            Fraction(1, 2), self.grid,
        )
        # 0.5 + 2*1.25 - 2.34 = 0.66
        self.assertEqual(fixed.mantissa, 66)
        self.assertIsInstance(fixed.mantissa, int)

    def test_error_synthesis_weighted(self):
        leaf = fp.ErrorResult(key="leaf", total=Fraction(5, 1000))
        coeff = fp.quantize_coefficient(3, self.grid)
        result = fp.synthesize_error(
            "out", ["leaf"], {"leaf": coeff},
            {"leaf": leaf}, {}, [],
        )
        # |3| * 0.005 = 0.015
        self.assertEqual(result.total, Fraction(15, 1000))
        self.assertEqual(result.paths[0].path, ["leaf", "out"])
        self.assertEqual(result.paths[0].level, 1)


if __name__ == "__main__":
    unittest.main()
