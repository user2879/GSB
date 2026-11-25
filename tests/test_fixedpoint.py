from __future__ import annotations

import unittest

from evidence_diag.fixedpoint import FixedDomain


class FixedPointTests(unittest.TestCase):
    def test_deterministic_mul_div_and_quantization(self) -> None:
        domain = FixedDomain(8, 4, "half_up")
        self.assertEqual(domain.value_to_units("0.57"), 144)
        self.assertEqual(domain.mul(128, 128), 64)
        self.assertEqual(domain.div(64, 128), 128)
        self.assertEqual(domain.pow_ratio(domain.value_to_units("0.9"), 2), 212)

    def test_rounding_modes_can_differ(self) -> None:
        half_up = FixedDomain(8, 1, "half_up")
        half_even = FixedDomain(8, 1, "half_even")
        down = FixedDomain(8, 1, "down")
        value = "0.001953125"  # exactly half a unit at 8 fractional bits
        self.assertEqual(half_up.value_to_units(value), 1)
        self.assertEqual(half_even.value_to_units(value), 0)
        self.assertEqual(down.value_to_units(value), 0)

    def test_quantization_step_changes_intermediate_integer(self) -> None:
        fine = FixedDomain(8, 1, "half_up")
        coarse = FixedDomain(8, 8, "half_up")
        self.assertNotEqual(
            fine.value_to_units("0.57"), coarse.value_to_units("0.57")
        )


if __name__ == "__main__":
    unittest.main()
