from __future__ import annotations

import unittest

from colmv.fixed import (
    INT64_MAX,
    INT64_MIN,
    FixedPointContract,
    saturate_int64,
    trunc_div_toward_zero,
)


class FixedPointTests(unittest.TestCase):
    def test_quantize_truncates_toward_zero(self) -> None:
        contract = FixedPointContract(3)
        self.assertEqual(contract.quantize(1.2349), 1234)
        self.assertEqual(contract.quantize(-1.2349), -1234)
        self.assertEqual(contract.quantize(7), 7000)
        self.assertEqual(contract.quantize(-7), -7000)
        self.assertIsNone(contract.quantize(None))

    def test_division_truncates_toward_zero(self) -> None:
        self.assertEqual(trunc_div_toward_zero(7, 2), 3)
        self.assertEqual(trunc_div_toward_zero(-7, 2), -3)
        self.assertEqual(trunc_div_toward_zero(7, -2), -3)
        self.assertEqual(trunc_div_toward_zero(0, -2), 0)

    def test_saturation(self) -> None:
        self.assertEqual(saturate_int64(INT64_MAX + 1), INT64_MAX)
        self.assertEqual(saturate_int64(INT64_MIN - 1), INT64_MIN)
        self.assertEqual(saturate_int64(42), 42)

    def test_decimal_rendering(self) -> None:
        contract = FixedPointContract(3)
        self.assertEqual(contract.decimal_from_units(1234), "1.234")
        self.assertEqual(contract.decimal_from_units(-1), "-0.001")
        self.assertEqual(FixedPointContract(0).decimal_from_units(9), "9")


if __name__ == "__main__":
    unittest.main()
