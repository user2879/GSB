import unittest

from ancient_ocr.fixedpoint import FixedPoint, QUANT_LEVELS, validate_quant
from ancient_ocr.errors import PipelineValidationError


class FixedPointTest(unittest.TestCase):
    def test_round_half_up_and_directions(self):
        fp = FixedPoint(4)
        self.assertEqual(fp.round_to_int(fp.from_int(3)), (3, "exact"))
        self.assertEqual(fp.round_to_int(fp.from_int(2) + 8), (3, "up"))
        self.assertEqual(fp.round_to_int(fp.from_int(2) + 7), (2, "down"))
        self.assertEqual(fp.round_to_int(-(fp.from_int(2) + 8)), (-3, "up"))

    def test_mul_div_are_q_consistent(self):
        fp = FixedPoint(4)
        self.assertEqual(fp.mul(fp.from_int(2), fp.from_int(3)), fp.from_int(6))
        self.assertEqual(fp.div(10, 4), fp.from_int(2) + 8)
        self.assertEqual(fp.mean(10, 4), fp.div(10, 4))

    def test_quant_enum_validation(self):
        validate_quant(4, "standard")
        with self.assertRaises(PipelineValidationError):
            validate_quant(3, "standard")
        with self.assertRaises(PipelineValidationError):
            validate_quant(4, "unknown")
        self.assertEqual(QUANT_LEVELS["standard"], 4)


if __name__ == "__main__":
    unittest.main()
