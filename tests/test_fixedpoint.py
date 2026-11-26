import unittest

from intent_router.fixedpoint import (
    ALLOWED_FRACTION_BITS,
    QuantGear,
    round_half_up,
)


class FixedPointTest(unittest.TestCase):
    def test_round_half_up_always_toward_positive(self):
        self.assertEqual(round_half_up(5, 2), 6)
        self.assertEqual(round_half_up(4, 2), 4)
        self.assertEqual(round_half_up(7, 4), 8)
        self.assertEqual(round_half_up(6, 4), 8)
        self.assertEqual(round_half_up(1, 1), 1)

    def test_gear_quantize_and_half_step(self):
        gear = QuantGear(8, 4)
        self.assertEqual(gear.quantize(10), 12)
        self.assertEqual(gear.quantize(9), 8)
        self.assertTrue(gear.at_half_step(10))
        self.assertFalse(gear.at_half_step(9))

    def test_enum_validation(self):
        with self.assertRaises(ValueError):
            QuantGear(3, 1)
        with self.assertRaises(ValueError):
            QuantGear(8, 3)
        self.assertIn(8, ALLOWED_FRACTION_BITS)

    def test_scale_lift(self):
        gear = QuantGear(8, 1)
        self.assertEqual(gear.from_int(3), 3 * 256)


if __name__ == "__main__":
    unittest.main()
